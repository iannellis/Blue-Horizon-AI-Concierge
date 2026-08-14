"""Custom Ragas prompt overrides for Blue Horizon RAG metrics.

Ragas builds its metric prompts from plain class attributes (``instruction``
and ``examples``) on a :class:`~ragas.prompt.metrics.base_prompt.BasePrompt`
subclass, so a metric's prompt can be replaced by assigning a different prompt
instance to ``metric.prompt`` after construction.

This module supplies a context-precision prompt tuned for the hotel concierge
dataset, whose queries frequently bundle several unrelated questions into one
turn ("Do you have a pool, and what time is checkout?"). The stock Ragas
prompt gives no guidance on partial coverage, and weaker judge models resolve
that ambiguity by marking a chunk irrelevant unless it covers the whole
question -- which collapses context precision on multi-part turns.
"""

from __future__ import annotations

from ragas.metrics.collections.context_precision.util import (
    ContextPrecisionInput,
    ContextPrecisionOutput,
    ContextPrecisionPrompt,
)


class HotelContextPrecisionPrompt(ContextPrecisionPrompt):
    """Context-precision prompt that scores multi-part questions per clause.

    Overrides only ``instruction`` and ``examples``; ``input_model`` and
    ``output_model`` are inherited unchanged, so the JSON schema Ragas
    generates for the judge is identical to the stock prompt's.

    The instruction makes two things explicit that the stock prompt leaves
    ambiguous: a chunk covering one clause of a multi-part question is useful,
    and each chunk is judged independently of what the other chunks cover. The
    examples demonstrate both halves of a two-part question scoring 1, plus a
    same-domain chunk that answers neither clause scoring 0, so the added
    leniency does not degrade into "any hotel content counts".
    """

    instruction = (
        "Given a question, an answer, and a single retrieved context chunk, "
        "decide whether the chunk was useful in arriving at the answer.\n\n"
        "The question may ask about several distinct things at once. A chunk is "
        "useful if it supports at least one part of the question. It does not "
        "need to address every part. Judge the chunk on its own and do not "
        "consider whether other chunks cover the remaining parts.\n\n"
        "A chunk that is merely on the same topic, but does not supply "
        "information needed for any part of the answer, is not useful.\n\n"
        'Give verdict "1" if the chunk supports any part of the question and '
        '"0" if it supports none. Output JSON with a short reason and the verdict.'
    )

    examples = [  # noqa: RUF012
        # Compound question, chunk covers one clause -> useful.
        (
            ContextPrecisionInput(
                question="Do you have a pool, and what time is checkout?",
                context=(
                    "The rooftop pool is open daily from 7:00 AM to 10:00 PM and "
                    "is available to all registered guests."
                ),
                answer=(
                    "Yes, we have a rooftop pool open 7:00 AM to 10:00 PM. "
                    "Checkout is at 11:00 AM."
                ),
            ),
            ContextPrecisionOutput(
                reason=(
                    "The chunk supplies the pool information, which is one of the "
                    "two things asked about. Partial coverage is sufficient."
                ),
                verdict=1,
            ),
        ),
        # Same compound question, chunk covers the other clause -> also useful.
        (
            ContextPrecisionInput(
                question="Do you have a pool, and what time is checkout?",
                context=(
                    "Standard checkout is 11:00 AM. Late checkout until 2:00 PM "
                    "may be arranged at the front desk subject to availability."
                ),
                answer=(
                    "Yes, we have a rooftop pool open 7:00 AM to 10:00 PM. "
                    "Checkout is at 11:00 AM."
                ),
            ),
            ContextPrecisionOutput(
                reason=(
                    "The chunk supplies the checkout time, which is the second "
                    "part of the question."
                ),
                verdict=1,
            ),
        ),
        # Topically adjacent but supports neither clause -> not useful.
        (
            ContextPrecisionInput(
                question="Do you have a pool, and what time is checkout?",
                context=(
                    "The fitness center is located on the second floor and is "
                    "accessible with your room key at any hour."
                ),
                answer=(
                    "Yes, we have a rooftop pool open 7:00 AM to 10:00 PM. "
                    "Checkout is at 11:00 AM."
                ),
            ),
            ContextPrecisionOutput(
                reason=(
                    "Amenity information, but it addresses neither the pool nor "
                    "the checkout time."
                ),
                verdict=0,
            ),
        ),
    ]
