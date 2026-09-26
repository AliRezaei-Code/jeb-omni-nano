"""Jeb-Omni-Nano decision head.

Design lineage
--------------
This head is a faithful, deliberately small reimplementation of the readout used by
``akhilaaa3/Jev-Omni`` (12B, Gemma-4-12B-IT backbone), ported to a ~350M-class
Liquid AI LFM2.5 backbone. See ``.deep-research/research-report.md`` for the full
source-verified derivation.

The original, verbatim from the Jev-Omni checkpoint's ``jev_omni.py``::

    class _Head256(torch.nn.Module):
        \"\"\"Decision head: normalise the last hidden state, one logit per option slot.\"\"\"
        def __init__(self, hidden):
            super().__init__()
            self.register_buffer("mu", torch.zeros(1, hidden))
            self.register_buffer("sd", torch.ones(1, hidden))
            self.linear = torch.nn.Linear(hidden, 256, dtype=torch.float32)

        def forward(self, features, counts):
            z = self.linear((features.float() - self.mu) / self.sd)
            return z.masked_fill(torch.arange(256, device=z.device)[None] >= counts[:, None], -1e30)

Three properties carry over unchanged and matter:

1. **Slot logits, not token logits.** The 256 outputs are *positional* — "option 0",
   "option 1", ... — not the vocabulary. This is what lets one checkpoint answer a
   question it has never seen, with an arbitrary number of options, without retraining.
2. **Frozen normalisation statistics.** ``mu``/``sd`` are buffers, not parameters:
   the backbone's final hidden state is standardised before the linear map. This
   keeps the head numerically well-conditioned across backbones and dtypes.
3. **fp32 head on a bf16 backbone.** The linear map and the softmax are computed in
   float32 while the backbone runs in bfloat16, so probabilities do not inherit
   bf16's ~3 decimal digits of mantissa.

The one deliberate addition over the original is :func:`DecisionHead.forward` returning
the masked logits *and* an optional per-type temperature, which is the mechanism every
open Jev-style reimplementation converged on for calibration (see report S5).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

MAX_OPTIONS = 256
NEG_INF = -1e30


class DecisionHead(nn.Module):
    """Standardise the final hidden state, then score 256 fixed option slots.

    Parameters
    ----------
    hidden_size:
        Backbone hidden width. 3840 for Gemma-4-12B, 1024 for LFM2.5-350M,
        2048 for LFM2.5-2.6B.
    max_options:
        Hard cap on the number of options. ``MAX_OPTIONS`` (256) is the value the
        whole ecosystem converged on: TypeSafe's Jev caps at 255 (2**8 - 1), and
        Jev-Omni's head is literally ``Linear(hidden, 256)``. Lowering this shrinks
        the head and is the single cheapest way to shrink the model further.
    mu, sd:
        Optional initial/loaded standardisation statistics, shape ``(1, hidden_size)``.
        Left as the identity when ``None``, matching Jev-Omni's initial state.
    """

    def __init__(
        self,
        hidden_size: int,
        max_options: int = MAX_OPTIONS,
        mu: torch.Tensor | None = None,
        sd: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if not 2 <= max_options <= MAX_OPTIONS:
            raise ValueError(f"max_options must be in [2, {MAX_OPTIONS}], got {max_options}")
        self.hidden_size = hidden_size
        self.max_options = max_options
        self.register_buffer("mu", torch.zeros(1, hidden_size) if mu is None else mu.float())
        self.register_buffer("sd", torch.ones(1, hidden_size) if sd is None else sd.float())
        # fp32 head on purpose. Matches Jev-Omni's `dtype=torch.float32`.
        self.linear = nn.Linear(hidden_size, max_options, dtype=torch.float32)

    def forward(self, features: torch.Tensor, n_options: torch.Tensor) -> torch.Tensor:
        """Score option slots.

        Parameters
        ----------
        features:
            ``(batch, hidden_size)`` final-layer hidden states, typically the last
            token position. Cast to float32 internally.
        n_options:
            ``(batch,)`` int tensor of the *real* number of options per row. Rows
            padded to ``max_options`` get ``NEG_INF`` so they cannot win a softmax.

        Returns
        -------
        ``(batch, max_options)`` float32 logits, with padded slots at ``NEG_INF``.
        """
        if features.shape[-1] != self.hidden_size:
            raise ValueError(
                f"expected hidden_size={self.hidden_size}, got {features.shape[-1]}"
            )
        z = self.linear((features.float() - self.mu) / self.sd)
        idx = torch.arange(self.max_options, device=z.device)
        return z.masked_fill(idx[None, :] >= n_options.to(z.device)[:, None], NEG_INF)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"hidden_size={self.hidden_size}, max_options={self.max_options}, "
            f"params={self.numel():,}"
        )


#: Default shrinkage constant for grouped temperature fits, k = 100.
#: A group's weight toward its own fitted value is ``n / (n + k)``, so a group with
#: 100 rows gets half weight and shrinks halfway to the global temperature. Verified
#: arithmetically against the published ``Jev-Style-0.8B-Decision-v3`` calibrator,
#: whose 20 groups all satisfy ``weight == n / (n + 100)`` exactly.
SHRINKAGE_K = 100.0


def option_bucket(n_options: int) -> str:
    """Bucket an option count. Mirrors the published convention.

    ``"2", "3-5", "6-10", "11-20", "21+"`` -- the same buckets used by
    ``Jev-Style-0.8B-Decision-v3``, so the fitted temperatures are comparable.
    """
    if n_options <= 2:
        return "2"
    if n_options <= 5:
        return "3-5"
    if n_options <= 10:
        return "6-10"
    if n_options <= 20:
        return "11-20"
    return "21+"


def shrink_temperature(t_raw: float, t_global: float, n: int, k: float = SHRINKAGE_K) -> float:
    """Shrink a group's raw temperature toward the global one, in LOG space.

    ``w = n / (n + k)``; ``T = exp(w * ln T_raw + (1 - w) * ln T_global)``.

    The log-space (geometric) blend is not a stylistic choice. Reproducing the
    published v3 calibrator, the geometric blend matches its stored values to ~5
    decimal places while an arithmetic blend is off by up to 8% on the same groups.

    Why shrink at all: a group fitted on 15 rows is a textbook overfit. In the
    published calibrator one such group wanted T = 5.0 and was saved by the clamp
    alone. Shrinkage is the principled version of that rescue, and it degrades
    gracefully to the global value as ``n`` goes to zero.
    """
    w = n / (n + k) if (n + k) > 0 else 0.0
    if w <= 0.0:
        return float(t_global)
    if w >= 1.0:
        return float(t_raw)
    import math
    return float(math.exp(w * math.log(max(t_raw, 1e-9))
                          + (1.0 - w) * math.log(max(t_global, 1e-9))))


@dataclass
class TemperatureFit:
    """Temperatures fitted on a held-out calibration split, with shrinkage.

    Two axes, because the evidence says both matter:

    *BY QUESTION TYPE.* noul / choice / score produce differently-shaped logit
    distributions. Jebadiah ships three (1.12 / 1.33 / 1.20).

    *BY OPTION COUNT.* This one is easy to miss and matters more. A published 151M
    calibrator fitted temperature per cardinality and found a **3.3x spread** --
    k=2 -> 5.01, k=6 -> 2.39, k=9 -> 1.67, k=25 -> 1.51. A single global T is 3.3x
    too wrong at the ends of the range.

    Any group with fewer than ~100 rows is shrunk toward ``global`` in log space,
    because small-group fits are noise.

    ``T`` is applied to logits *before* the softmax: ``p = softmax(z / T)``. This
    cannot change the argmax, only the shape around it -- which is what makes it a
    safe post-hoc knob.

    ``T`` may be **below** 1.0. A published 0.8B calibrator ships a global T of
    0.880, meaning that model was *under*confident and calibration sharpened it.
    Do not assume T >= 1.
    """

    global_: float = 1.0
    by_type: dict[str, float] = field(default_factory=dict)
    by_bucket: dict[str, float] = field(default_factory=dict)
    #: Raw (unshrunk) per-bucket fits, kept so a larger calibration set can be
    #: re-shrunk later without refitting.
    bucket_raw: dict[str, float] = field(default_factory=dict)
    bucket_n: dict[str, int] = field(default_factory=dict)
    n_rows: int = 0
    fitted_on: str | None = field(default=None, compare=False)

    def get(self, question_type: str = "choice", n_options: int | None = None) -> float:
        """Look up T: per-bucket first, then per-type, then global.

        Precedence matters. A bucket is the most specific and most data-hungry, so
        it wins when present; a type fit is the fallback when a bucket has no rows.
        """
        if question_type not in ("choice", "noul", "score"):
            raise ValueError(
                f"unknown question type {question_type!r}; expected choice, noul or score"
            )
        if n_options is not None:
            t = self.by_bucket.get(option_bucket(n_options))
            if t is not None:
                return float(t)
        t = self.by_type.get(question_type)
        if t is not None:
            return float(t)
        return float(self.global_)

    def to_dict(self) -> dict:
        return {
            "global": self.global_,
            "by_type": self.by_type,
            "by_bucket": self.by_bucket,
            "bucket_raw": self.bucket_raw,
            "bucket_n": self.bucket_n,
            "n_rows": self.n_rows,
            "fitted_on": self.fitted_on,
            "shrinkage_k": SHRINKAGE_K,
            "option_buckets": ["2", "3-5", "6-10", "11-20", "21+"],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TemperatureFit":
        return cls(
            global_=float(d.get("global", 1.0)),
            by_type={k: float(v) for k, v in (d.get("by_type") or {}).items()},
            by_bucket={k: float(v) for k, v in (d.get("by_bucket") or {}).items()},
            bucket_raw={k: float(v) for k, v in (d.get("bucket_raw") or {}).items()},
            bucket_n={k: int(v) for k, v in (d.get("bucket_n") or {}).items()},
            n_rows=int(d.get("n_rows", 0)),
            fitted_on=d.get("fitted_on"),
        )


def probabilities(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Softmax with temperature scaling, in float32.

    Padded slots hold ``NEG_INF`` and therefore receive exactly 0.0, so a caller can
    zip the result against the real option list without slicing.
    """
    return torch.softmax(logits.float() / max(temperature, 1e-6), dim=-1)


def fit_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    grid: torch.Tensor | None = None,
) -> float:
    """Fit a single temperature by minimising NLL on a calibration split.

    Parameters
    ----------
    logits:
        ``(n, K)`` *unmasked-for-real-options* logits from the head. Padded slots must
        already be ``NEG_INF`` (or be sliced off) so the softmax only sees real options.
    targets:
        ``(n,)`` int64 index of the correct option per row. Use a **soft** target
        distribution when you have one — Jebadiah fits against the training target
        (soft/ordinal) because that is the distribution the model was taught; fitting
        against the argmax is the "hard" fit and sharpens, which measurably hurt its
        rubric calibration.
    grid:
        Search grid. Defaults to 0.25..5.0, 96 points. A grid search beats LBFGS here
        because NLL in T is smooth but nearly flat near the optimum; the grid makes
        the fit reproducible and dependency-free, and 96 points is well under a
        millisecond.

    Returns
    -------
    The temperature minimising mean negative log-likelihood.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be (n, K), got shape {tuple(logits.shape)}")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets must agree on the number of rows")
    if grid is None:
        grid = torch.linspace(0.25, 5.0, 96)
    logits = logits.detach().float()
    targets = targets.detach()

    best_t, best_nll = 1.0, float("inf")
    for t in grid:
        logp = torch.log_softmax(logits / t, dim=-1)
        nll = -logp.gather(1, targets[:, None]).mean().item()
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error over ``n_bins`` equal-width confidence bins.

    ECE = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

    ``probs`` is ``(n, K)`` predicted distributions, ``labels`` is ``(n,)``. Uses the
    **top-1** probability as the confidence, matching how DecisionBench and JevBench
    both report it. 10 bins is the DecisionBench convention; its chart uses 5 for
    readability, so numbers quoted from the chart and the table are not comparable.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    conf, pred = probs.detach().float().max(dim=-1)
    correct = (pred == labels).float()
    n = conf.numel()
    if n == 0:
        return float("nan")
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=conf.device)
    # bucketise with right=True so a confidence of exactly 1.0 lands in the last bin
    idx = torch.bucketize(conf, edges[1:-1], right=True)
    ece = torch.zeros((), device=conf.device)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece = ece + mask.float().mean() * (correct[mask].mean() - conf[mask].mean()).abs()
    return float(ece.item())


def brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Multiclass Brier score: mean squared error between the distribution and one-hot.

    ``sum_k (p_k - y_k)**2 / K`` averaged over rows, with ``K`` the option count.
    Lower is better. Reported alongside ECE by the typed-decisions benchmark, which
    argues for reading Brier over ECE because a base-rate-only predictor is perfectly
    calibrated by construction yet useless. The ``1/K`` normalisation is what puts a
    uniform guess near 0.25 for a 2-way question, matching the published uniform row
    of 0.238 on that benchmark.
    """
    n, k = probs.shape
    onehot = torch.zeros_like(probs, dtype=torch.float32)
    onehot.scatter_(1, labels[:, None], 1.0)
    return float(((probs.float() - onehot) ** 2).sum(dim=-1).mean().item() / k)
