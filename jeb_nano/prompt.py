"""Prompt contract for Jeb-Omni-Nano.

A *prompt contract* is a fixed, versioned byte-exact rendering of a decision request
into text. Every published reimplementation of a Jev-class model pins one with a hash
and asserts at test time that training, evaluation and serving all use the same bytes.
Jebadiah stores the source commit and a ``prompt_source_sha256``; Kev ships parity
tests. We do the same: the renderer lives here alone, and
``tests/test_prompt_identity.py`` checks the hash is unchanged.

The rendering below is deliberately the Jev-Omni shape, not the Jev shape:

* Jev-Omni numbers the options and asks for a number, then reads *slot* logits::

      {state}

      ---

      QUESTION: {question}

      OPTIONS:
      1. {opt}
      2. {opt}

      Reply with only the number of the correct option (1-{K}).
      Output a single number and nothing else.

* Jev / Kev / Jebadiah label the options with single tokens and read the
  distribution over those label tokens at the answer position.

Slot logits win here for one reason: they are **positional**, so the head is a plain
``Linear(hidden, 256)`` and a brand-new question with new option words needs no new
parameters. That property is exactly what makes one checkpoint generalise to unseen
question schemas, which is the whole point of a generalist decision model.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

QuestionType = Literal["noul", "choice", "score"]

SYSTEM_PROMPT = (
    "You are a decision classifier. Given a state, a question and a numbered list of "
    "options, reply with only the number of the correct option. Output a single number "
    "and nothing else."
)

PROMPT_CONTRACT_VERSION = "1.0.0"


@dataclass(frozen=True)
class Question:
    """One typed question over a shared state.

    ``options`` is the ordered option list. Order is part of the contract: the head
    reads *slots*, so permuting the list permutes the label semantics. Option-order
    robustness is a real, measured failure mode in this class of model (JevBench
    documents a 72% -> 21% collapse on one reverse-order run), so evaluate under
    permutation rather than trusting a single ordering.
    """

    key: str
    instructions: str
    type: QuestionType = "choice"
    options: tuple[str, ...] = ()
    #: Optional human-readable criteria per option. Included in the prompt because the
    #: typed-decisions benchmark found that stripping criteria to a bare label list
    #: makes the task measurably easier; keep them if your callers supply them.
    criteria: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in ("noul", "choice", "score"):
            raise ValueError(f"unknown question type {self.type!r}")
        if self.type == "noul":
            if not self.options:
                object.__setattr__(self, "options", ("Yes", "No"))
        elif not self.options:
            raise ValueError(f"question {self.key!r} of type {self.type!r} has no options")
        if len(self.options) < 2:
            raise ValueError(f"question {self.key!r} needs at least 2 options")
        if len(self.options) > 256:
            raise ValueError(
                f"question {self.key!r} has {len(self.options)} options; the head caps "
                "at 256. Jev-Omni is documented as best-supported at <=20."
            )

    @property
    def n_options(self) -> int:
        return len(self.options)


@dataclass(frozen=True)
class DecisionRequest:
    """A state plus one or more typed questions.

    ``state`` may be a string or any JSON-serialisable object. Objects are rendered
    as sorted ``key: value`` lines for determinism, matching how Kev renders
    object/array states ("Objects and arrays are converted to labeled text").
    """

    state: str | dict[str, Any] | list[Any]
    questions: tuple[Question, ...] | list[Question]

    def __post_init__(self) -> None:
        qs = tuple(self.questions)
        if not qs:
            raise ValueError("a DecisionRequest needs at least one question")
        object.__setattr__(self, "questions", qs)

    @property
    def state_text(self) -> str:
        if isinstance(self.state, str):
            return self.state
        if isinstance(self.state, dict):
            return "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False, sort_keys=True)}"
                             for k, v in sorted(self.state.items()))
        return json.dumps(self.state, ensure_ascii=False, sort_keys=True)


def render_question(question: Question) -> str:
    """Render one question to the exact text fed to the backbone."""
    options = "\n".join(f"{i + 1}. {opt}" for i, opt in enumerate(question.options))
    if question.criteria:
        criteria = "\n".join(
            f"  - {opt}: {question.criteria[opt]}"
            for opt in question.options
            if question.criteria.get(opt)
        )
        options = f"{options}\nCRITERIA:\n{criteria}"
    return (
        f"QUESTION: {question.instructions}\n\n"
        f"OPTIONS:\n{options}\n\n"
        f"Reply with only the number of the correct option (1-{question.n_options}).\n"
        "Output a single number and nothing else."
    )


def render_request(request: DecisionRequest, joiner: str = "\n\n---\n\n") -> str:
    """Render a whole request to text.

    The default joins questions with a delimiter so each question is a visibly
    separate branch. Whether a question can attend to its siblings is an *architecture*
    property (a mask, or one row per question), not a prompt property — this delimiter
    only marks boundaries. If you need hard isolation, use
    :func:`render_single_question`, which emits exactly one question per prompt.
    """
    body = joiner.join(render_question(q) for q in request.questions)
    return f"{request.state_text}\n\n---\n\n{body}"


def render_single_question(state: str, question: Question) -> str:
    """Render state + exactly one question. The isolation-safe path.

    Use this when you need the guarantee that a question cannot see its siblings: one
    prompt per question means no sibling tokens exist to attend to. Costs one state
    re-encode per question unless you reuse a shared KV cache, which is the
    optimisation JevBench's cost model is built around.
    """
    return f"{state}\n\n---\n\n{render_question(question)}"


def prompt_sha256(request: DecisionRequest) -> str:
    """Stable hash of the rendered prompt, for run records and regression tests."""
    return hashlib.sha256(render_request(request).encode("utf-8")).hexdigest()


def to_chat_messages(request: DecisionRequest) -> list[dict[str, str]]:
    """Wrap the rendered request in the LFM2.5 ChatML message list.

    LFM2.5 uses a ChatML-like template: ``<|im_start|>system`` / ``<|im_start|>user``
    / ``<|im_start|>assistant`` with ``<|im_end|>`` terminators. Feed this to
    ``tokenizer.apply_chat_template(..., add_generation_prompt=True, tokenize=True)``
    and read the head at the final position.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_request(request)},
    ]


def permute(question: Question, order: list[int]) -> Question:
    """Return a copy of ``question`` with options reordered by ``order``.

    Used to run option-order robustness checks. A model whose accuracy collapses
    under permutation is not safe to gate on.
    """
    if sorted(order) != list(range(question.n_options)):
        raise ValueError("order must be a permutation of range(n_options)")
    return Question(
        key=question.key,
        instructions=question.instructions,
        type=question.type,
        options=tuple(question.options[i] for i in order),
        criteria={question.options[i]: question.criteria.get(question.options[i], "")
                  for i in order},
    )
