"""Custom Ragas prompt overrides for Blue Horizon RAG metrics.

Ragas builds its metric prompts from plain class attributes (``instruction``
and ``examples``) on a :class:`~ragas.prompt.metrics.base_prompt.BasePrompt`
subclass, so a metric's prompt can be replaced by assigning a different prompt
instance to ``metric.prompt`` after construction.

This module supplies a context-precision prompt tuned for the Blue Horizon
concierge dataset, whose queries frequently bundle several unrelated
questions into one turn ("Do you have a pool, and what time is checkout?").
The stock Ragas prompt gives no guidance on partial coverage, and weaker
judge models resolve that ambiguity by marking a chunk irrelevant unless it
covers the whole question -- which collapses context precision on multi-part
turns. The examples below are deliberately domain-neutral (a software
product FAQ) rather than hotel-related, so the judge learns the general
partial-coverage rule instead of pattern-matching on cases that resemble the
eval dataset itself.
"""

from __future__ import annotations

from ragas.metrics.collections.context_precision.util import (
    ContextPrecisionInput,
    ContextPrecisionOutput,
    ContextPrecisionPrompt,
)


class PartialCoverageContextPrecisionPrompt(ContextPrecisionPrompt):
    """Context-precision prompt that scores multi-part questions per clause.

    Overrides only ``instruction`` and ``examples``; ``input_model`` and
    ``output_model`` are inherited unchanged, so the JSON schema Ragas
    generates for the judge is identical to the stock prompt's.

    The instruction makes two things explicit that the stock prompt leaves
    ambiguous: a chunk covering one clause of a multi-part question is useful,
    and each chunk is judged independently of what the other chunks cover. The
    examples open with an ordinary single-fact question scoring 1, to anchor
    that the added leniency only changes how multi-part questions are judged
    and leaves single-part questions at the stock standard. They go on to
    demonstrate both halves of a two-part question scoring 1, plus a
    same-domain chunk that answers neither clause scoring 0, so the added
    leniency does not degrade into "any content on the same topic counts".
    They use a software-product FAQ rather than a hotel scenario so the
    judge learns the general rule instead of anchoring on cases that look
    like the eval dataset.
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
        # Single-fact question, chunk directly answers it -> useful.
        (
            ContextPrecisionInput(
                question="What is the maximum file size for uploads?",
                context=(
                    "Uploaded files may be up to 2 GB each. Larger files must be "
                    "split into parts before uploading."
                ),
                answer="The maximum upload size is 2 GB per file.",
            ),
            ContextPrecisionOutput(
                reason=(
                    "The question asks for one fact and the chunk supplies it; "
                    "no partial-coverage reasoning is needed."
                ),
                verdict=1,
            ),
        ),
        # Compound question, chunk covers one clause -> useful.
        (
            ContextPrecisionInput(
                question=(
                    "Does the app support two-factor authentication, and how do "
                    "I export my data?"
                ),
                context=(
                    "Two-factor authentication can be enabled from Settings > "
                    "Security by scanning a QR code with an authenticator app."
                ),
                answer=(
                    "Yes, two-factor authentication is supported and can be "
                    "enabled under Settings > Security. You can export your data "
                    "as a CSV from the Data tab."
                ),
            ),
            ContextPrecisionOutput(
                reason=(
                    "The chunk supplies the two-factor authentication "
                    "information, which is one of the two things asked about. "
                    "Partial coverage is sufficient."
                ),
                verdict=1,
            ),
        ),
        # Same compound question, chunk covers the other clause -> also useful.
        (
            ContextPrecisionInput(
                question=(
                    "Does the app support two-factor authentication, and how do "
                    "I export my data?"
                ),
                context=(
                    "Data can be exported as a CSV file from the Data tab. Exports "
                    "larger than 10,000 rows are split into multiple files."
                ),
                answer=(
                    "Yes, two-factor authentication is supported and can be "
                    "enabled under Settings > Security. You can export your data "
                    "as a CSV from the Data tab."
                ),
            ),
            ContextPrecisionOutput(
                reason=(
                    "The chunk supplies the data export information, which is "
                    "the second part of the question."
                ),
                verdict=1,
            ),
        ),
        # Topically adjacent but supports neither clause -> not useful.
        (
            ContextPrecisionInput(
                question=(
                    "Does the app support two-factor authentication, and how do "
                    "I export my data?"
                ),
                context=(
                    "Password reset links are valid for 30 minutes and can be "
                    "requested from the login screen."
                ),
                answer=(
                    "Yes, two-factor authentication is supported and can be "
                    "enabled under Settings > Security. You can export your data "
                    "as a CSV from the Data tab."
                ),
            ),
            ContextPrecisionOutput(
                reason=(
                    "Account-security-adjacent information, but it addresses "
                    "neither two-factor authentication nor data export."
                ),
                verdict=0,
            ),
        ),
    ]
