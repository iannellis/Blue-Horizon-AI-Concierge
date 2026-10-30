"""Moderation check on guest messages and agent replies.

Whether a message is refused otherwise depends entirely on the router model's
judgement, which a determined jailbreak can talk around. OpenAI's moderation
endpoint is a separate classifier with no instructions to argue with, and it
is free to call.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import openai

if TYPE_CHECKING:
    from blue_horizon.config import OrchestrationModerationConfig

logger = logging.getLogger(__name__)


class Moderator:
    """Asks OpenAI's moderation endpoint whether a text is harmful.

    The text itself is never logged, only which categories flagged it.
    """

    __slots__ = ("_client", "_model")

    _client: openai.AsyncOpenAI
    _model: str

    def __init__(self, config: OrchestrationModerationConfig) -> None:
        """Build the moderation client.

        Args:
            config: Moderation model, timeout, and retry settings.

        Raises:
            openai.OpenAIError: If ``OPENAI_API_KEY`` is not set.

        """
        self._client = openai.AsyncOpenAI(
            timeout=config.timeout_s,
            max_retries=config.max_retries,
        )
        self._model = config.model

    async def flagged(self, text: str, *, source: str) -> bool:
        """Return whether the moderation endpoint flags a text.

        A flagged text is logged as one warning line naming its categories,
        which are also attached to the record as ``moderation_categories``.

        Args:
            text: The text to check.
            source: What the text is, such as ``"guest message"``, for the
                log line.

        Returns:
            True if any moderation category flagged the text.

        Raises:
            openai.APIError: If the request fails after the client's retries.

        """
        response = await self._client.moderations.create(
            model=self._model,
            input=text,
        )
        result = response.results[0]
        if result.flagged:
            categories = sorted(
                name
                for name, hit in result.categories.model_dump(by_alias=True).items()
                if hit
            )
            logger.warning(
                "Moderation flagged the %s: categories=%s",
                source,
                ",".join(categories),
                extra={"moderation_categories": categories},
            )
        return result.flagged

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()
