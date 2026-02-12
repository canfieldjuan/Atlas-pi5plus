"""Buffer that accumulates streaming tokens and yields complete sentences.

Ported from atlas_brain/voice/pipeline.py SentenceBuffer.
Used for progressive TTS: speak each sentence as soon as it completes
instead of waiting for the full LLM response.
"""

from typing import Optional


SENTENCE_ENDINGS = ".!?"


class SentenceBuffer:
    """Accumulate tokens, yield complete sentences on punctuation boundaries."""

    def __init__(self):
        self._tokens: list[str] = []

    def add_token(self, token: str) -> Optional[str]:
        """Add a token. Returns a complete sentence if one is ready, else None."""
        self._tokens.append(token)
        combined = "".join(self._tokens).strip()
        if combined and combined[-1] in SENTENCE_ENDINGS:
            self._tokens.clear()
            return combined
        return None

    def flush(self) -> Optional[str]:
        """Flush any remaining buffered content as a final sentence."""
        combined = "".join(self._tokens).strip()
        if combined:
            self._tokens.clear()
            return combined
        return None

    def clear(self):
        """Discard all buffered tokens."""
        self._tokens.clear()
