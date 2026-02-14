"""Identity sync manager -- keeps local face/gait/speaker .npy databases
in sync with the Atlas Brain master registry over WebSocket."""

import asyncio
import logging
from pathlib import Path

import numpy as np

from . import config

log = logging.getLogger(__name__)

MODALITIES = ("face", "gait", "speaker")

# Maps modality -> config DB directory
_DB_DIRS = {
    "face": config.FACE_DB_DIR,
    "gait": config.GAIT_DB_DIR,
    "speaker": config.SPEAKER_DB_DIR,
}


class IdentitySyncManager:
    """Synchronizes identity embeddings between edge node and Brain.

    On WS connect: sends a manifest of local identities -> Brain responds
    with missing/updated/deleted entries.  When a new identity is registered
    locally, pushes it to Brain so other nodes receive it.
    """

    def __init__(self, ws_client, face_db=None, gait_db=None, speaker_db=None):
        self._ws = ws_client
        self._dbs = {
            "face": face_db,
            "gait": gait_db,
            "speaker": speaker_db,
        }
        self._sync_interval = config.IDENTITY_SYNC_INTERVAL
        self._watch_interval = config.IDENTITY_WATCH_INTERVAL

        # Snapshot of (modality, name) pairs the sync manager knows about.
        # Anything that appears on disk but NOT here was created by an
        # external process (registration script) and needs pushing to Brain.
        self._known_files: set[tuple[str, str]] = self._scan_all_files()

        # Register WS handlers
        ws_client.add_handler("identity_sync", self._handle_sync)
        ws_client.add_handler("identity_update", self._handle_update)
        ws_client.add_handler("identity_delete", self._handle_delete)
        ws_client.on_connect(self._on_connected)

    # --- WS event handlers ---

    async def _on_connected(self):
        """Called when WS connects -- send sync request with current manifest."""
        manifest = self._current_manifest()
        msg = {"type": "identity_sync_request", "current": manifest}
        await self._ws.send(msg)
        counts = ", ".join(
            f"{len(names)} {mod}" for mod, names in manifest.items()
        )
        log.info("Identity sync: sent request with %s", counts)

    async def _handle_sync(self, msg: dict):
        """Handle full sync response from Brain."""
        identities = msg.get("identities", {})
        deletes = msg.get("delete", {})

        saved = 0
        for modality in MODALITIES:
            entries = identities.get(modality, {})
            for name, embedding_list in entries.items():
                embedding = np.array(embedding_list, dtype=np.float32)
                self._save_embedding(name, modality, embedding)
                saved += 1

            del_names = deletes.get(modality, [])
            for name in del_names:
                self._delete_embedding(name, modality)

            if entries or del_names:
                self._reload_db(modality)

        log.info(
            "Identity sync: received %d embeddings, %d deletions",
            saved,
            sum(len(v) for v in deletes.values()),
        )

        # Push embeddings that Brain is missing (edge-only identities)
        need_from_edge = msg.get("need_from_edge", {})
        pushed = 0
        for modality in MODALITIES:
            for name in need_from_edge.get(modality, []):
                path = Path(_DB_DIRS[modality]) / f"{name}.npy"
                if not path.exists():
                    log.warning("Brain requested %s/%s but file missing locally", modality, name)
                    continue
                try:
                    embedding = np.load(path)
                    await self.push_registration(name, modality, embedding)
                    pushed += 1
                except Exception:
                    log.exception("Failed to push %s/%s to Brain", modality, name)
        if pushed:
            log.info("Identity sync: pushed %d embeddings Brain was missing", pushed)

    async def _handle_update(self, msg: dict):
        """Handle single identity update pushed from Brain."""
        name = msg.get("name")
        modality = msg.get("modality")
        embedding_list = msg.get("embedding")

        if not name or not modality or embedding_list is None:
            log.warning("Identity update missing fields: %s", msg)
            return
        if modality not in MODALITIES:
            log.warning("Identity update unknown modality '%s'", modality)
            return

        # Don't re-save if it came from this node
        if msg.get("source_node") == config.NODE_ID:
            return

        embedding = np.array(embedding_list, dtype=np.float32)
        self._save_embedding(name, modality, embedding)
        self._reload_db(modality)
        log.info("Identity update: saved %s/%s from %s", modality, name, msg.get("source_node", "?"))

    async def _handle_delete(self, msg: dict):
        """Handle identity deletion pushed from Brain."""
        name = msg.get("name")
        modality = msg.get("modality")

        if not name or not modality:
            log.warning("Identity delete missing fields: %s", msg)
            return
        if modality not in MODALITIES:
            log.warning("Identity delete unknown modality '%s'", modality)
            return

        self._delete_embedding(name, modality)
        self._reload_db(modality)
        log.info("Identity delete: removed %s/%s", modality, name)

    # --- Push local registrations to Brain ---

    async def push_registration(self, name: str, modality: str, embedding: np.ndarray):
        """Push a locally registered embedding to Brain for distribution."""
        if not self._ws.is_connected:
            log.debug("WS not connected, skipping push for %s/%s", modality, name)
            return

        msg = {
            "type": "identity_register",
            "name": name,
            "modality": modality,
            "embedding": embedding.tolist(),
        }
        await self._ws.send(msg)
        log.info("Identity push: sent %s/%s to Brain", modality, name)

    # --- Periodic re-sync + local registration watcher ---

    async def periodic_sync(self):
        """Re-send sync request every IDENTITY_SYNC_INTERVAL seconds."""
        while True:
            await asyncio.sleep(self._sync_interval)
            if self._ws.is_connected:
                await self._on_connected()

    async def watch_local_registrations(self):
        """Poll DB directories for .npy files created by external scripts.

        Any file not in _known_files was written by a registration CLI
        (a separate process) and needs to be pushed to Brain.
        """
        while True:
            await asyncio.sleep(self._watch_interval)
            if not self._ws.is_connected:
                continue

            current = self._scan_all_files()
            new_entries = current - self._known_files
            if not new_entries:
                # Still sync _known_files with disk to track deletions
                self._known_files = current
                continue

            for modality, name in new_entries:
                path = Path(_DB_DIRS[modality]) / f"{name}.npy"
                try:
                    embedding = np.load(path)
                    await self.push_registration(name, modality, embedding)
                except Exception:
                    log.exception("Failed to push local registration %s/%s", modality, name)

            # Update known set so we don't push again (also removes deleted entries)
            self._known_files = current

    # --- Internal helpers ---

    def _current_manifest(self) -> dict:
        """Build {modality: [names]} from local .npy files."""
        manifest = {}
        for modality in MODALITIES:
            db_dir = Path(_DB_DIRS[modality])
            if db_dir.exists():
                manifest[modality] = sorted(
                    p.stem for p in db_dir.glob("*.npy")
                )
            else:
                manifest[modality] = []
        return manifest

    def _scan_all_files(self) -> set[tuple[str, str]]:
        """Return the set of (modality, name) pairs currently on disk."""
        files = set()
        for modality in MODALITIES:
            db_dir = Path(_DB_DIRS[modality])
            if db_dir.exists():
                for p in db_dir.glob("*.npy"):
                    files.add((modality, p.stem))
        return files

    def _save_embedding(self, name: str, modality: str, embedding: np.ndarray):
        """Write a .npy file for the given identity (Brain-originated)."""
        db_dir = Path(_DB_DIRS[modality])
        db_dir.mkdir(parents=True, exist_ok=True)
        path = db_dir / f"{name}.npy"
        np.save(path, embedding)
        self._known_files.add((modality, name))

    def _delete_embedding(self, name: str, modality: str):
        """Remove a .npy file for the given identity."""
        path = Path(_DB_DIRS[modality]) / f"{name}.npy"
        if path.exists():
            path.unlink()
        self._known_files.discard((modality, name))

    def _reload_db(self, modality: str):
        """Hot-reload the in-memory database for the given modality."""
        db = self._dbs.get(modality)
        if db is None:
            return
        try:
            db.load()
            log.debug("Reloaded %s database", modality)
        except Exception:
            log.exception("Failed to reload %s database", modality)
