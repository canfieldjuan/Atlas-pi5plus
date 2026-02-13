"""Local LLM client -- queries llama-server (Phi-3) as Brain fallback.

Uses the OpenAI-compatible /v1/chat/completions endpoint exposed by
llama-server on localhost.  Designed for RK3588 CPU inference (~2.4s/token
with Phi-3-mini Q4).  The 120s default timeout accommodates worst-case
generation of ~200 tokens at that speed.
"""

import asyncio
import logging

import aiohttp

from . import config

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are Atlas, a helpful home assistant running on an edge device. "
    "Give brief, conversational answers. Keep responses under 2 sentences "
    "unless the user asks for detail."
)


def _build_url(path: str) -> str:
    return f"http://127.0.0.1:{config.LOCAL_LLM_PORT}{path}"


class LocalLLM:
    """Async client for the local llama-server OpenAI-compatible API."""

    def __init__(self):
        self._session: aiohttp.ClientSession | None = None
        self._available = False
        self._busy = False

    async def start(self):
        connector = aiohttp.TCPConnector(limit_per_host=1, limit=2)
        self._session = aiohttp.ClientSession(connector=connector)
        try:
            await self._check_health()
        except Exception:
            await self._session.close()
            self._session = None
            raise

    async def stop(self):
        if self._session:
            try:
                await self._session.close()
            except Exception:
                log.exception("Error closing local LLM session")
            self._session = None

    async def _check_health(self):
        """Check if llama-server is responding."""
        try:
            async with self._session.get(
                _build_url("/health"),
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                self._available = resp.status == 200
                if self._available:
                    log.info("Local LLM available at port %d", config.LOCAL_LLM_PORT)
                else:
                    log.warning("Local LLM health check returned %d", resp.status)
        except Exception:
            self._available = False
            log.warning("Local LLM not reachable at port %d", config.LOCAL_LLM_PORT)

    @property
    def available(self) -> bool:
        return self._available

    async def query(self, text: str, speaker: str | None = None) -> str | None:
        """Send a query to the local LLM. Returns response text or None on failure.

        Only one query runs at a time.  If another query is in flight,
        returns None immediately to prevent pileup on slow CPU inference.
        If _available is False (previous failure), re-probes health first.
        """
        if not self._session:
            return None

        if self._busy:
            log.debug("Local LLM busy -- dropping query: %s", text[:40])
            return None

        # Re-probe health if previously marked unavailable
        if not self._available:
            await self._check_health()
            if not self._available:
                return None

        self._busy = True
        try:
            return await self._do_query(text, speaker)
        finally:
            self._busy = False

    async def _do_query(self, text: str, speaker: str | None) -> str | None:
        user_msg = text
        if speaker:
            user_msg = f"[{speaker} says]: {text}"

        payload = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": config.LOCAL_LLM_MAX_TOKENS,
            "temperature": config.LOCAL_LLM_TEMPERATURE,
        }

        try:
            async with self._session.post(
                _build_url("/v1/chat/completions"),
                json=payload,
                timeout=aiohttp.ClientTimeout(total=config.LOCAL_LLM_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    log.warning("Local LLM returned %d", resp.status)
                    self._available = False
                    return None

                data = await resp.json()

        except asyncio.TimeoutError:
            log.warning("Local LLM timed out (%.0fs limit)", config.LOCAL_LLM_TIMEOUT)
            self._available = False
            return None
        except Exception:
            log.exception("Local LLM query failed")
            self._available = False
            return None

        # Parse response -- validate structure instead of bare KeyError
        try:
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                log.warning("Local LLM returned non-string content: %s", type(content))
                return None
            content = content.strip()
        except (KeyError, IndexError, TypeError) as exc:
            log.warning("Local LLM malformed response: %s -- %s", exc, str(data)[:200])
            return None

        self._available = True
        log.info("Local LLM response: %s", content[:80])
        return content if content else None
