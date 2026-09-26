"""Jeb-Omni-Nano: a small, calibrated, typed-decision model on Liquid AI LFM2.5.

Read ``GUIDE.txt`` in the repo root for the full build guide, and
``.deep-research/research-report.md`` for the source-verified research report that
motivates every design choice here.
"""
from .head import (
    SHRINKAGE_K,
    DecisionHead,
    TemperatureFit,
    brier_score,
    expected_calibration_error,
    fit_temperature,
    option_bucket,
    probabilities,
    shrink_temperature,
)
from .prompt import (
    PROMPT_CONTRACT_VERSION,
    DecisionRequest,
    Question,
    permute,
    prompt_sha256,
    render_request,
    render_single_question,
    to_chat_messages,
)

__all__ = [
    "DecisionHead",
    "TemperatureFit",
    "brier_score",
    "expected_calibration_error",
    "fit_temperature",
    "probabilities",
    "PROMPT_CONTRACT_VERSION",
    "DecisionRequest",
    "Question",
    "permute",
    "prompt_sha256",
    "render_request",
    "render_single_question",
    "to_chat_messages",
]

__version__ = "0.1.0"
