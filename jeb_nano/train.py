"""Train a Jeb-Omni-Nano decision head with LoRA on a Liquid AI LFM2.5 backbone.

What is trained
---------------
Exactly two things, and nothing else:

1. A **LoRA adapter** on the backbone's linear projections.
2. The **decision head** (:class:`~jeb_nano.head.DecisionHead`).

The base weights are frozen. The adapter is ~1-2% of the model; the head is 262,400
parameters. This is the same split every open Jev-class implementation converged on
(Jebadiah: 32.5M LoRA params on 4B plus a pointer head; Kev: rank-16 LoRA plus a
pointer head; Jev-Omni: LoRA plus a 3840x256 linear head).

The objective
-------------
Cross-entropy over the option-slot logits **at the answer position only**. Nothing is
generated, so there are no target tokens to mask. For each training example you need:

* the rendered prompt, and
* the index of the correct option.

Two refinements that the published work shows are worth real accuracy, and both are
optional flags here:

``--soft-targets``
    Train toward a *distribution* over options rather than a one-hot vector. The
    typed-decisions benchmark ships gold distributions (the mean of three teacher
    samples), and the benchmark notes that carrying the soft target into a learner cut
    KL by a third and score MAE by 15% while barely moving accuracy. If your labels are
    genuinely ambiguous, this is how you tell the model so.

``--ordinal-score``
    For ``score`` questions, put 20% of the label's mass on the adjacent level instead
    of a one-hot target. Jebadiah measured this on human helpfulness ratings: the 9B's
    Decision Score went from -21.4 to +11.9 and its calibration error from 0.39 to
    0.045, with rubric accuracy unchanged. The model stopped being confidently wrong
    about things it cannot judge.

Data
----
A JSONL file, one request per line, in the shape Kev documents and DecisionBench uses::

    {"state": "...",
     "questions": {"team": {"type": "choice",
                            "instructions": "Which team should take this?",
                            "options": ["billing", "support"],
                            "label": 0},
                   "urgent": {"type": "noul", "instructions": "Is this urgent?",
                              "label": 1}}}

``label`` is the **index** of the correct option, or for ``score`` questions an object
``{"index": 2, "soft": [0.0, 0.0, 0.8, 0.2, 0.0]}``.

The cheapest serious starting set is ``LocalLLaMA/typed-decisions`` (Apache-2.0,
1,600 cases x 5 questions, gold distributions included) plus ``akhilaaa3/decision-bench``
(Apache-2.0, 293 questions x 2 subsets) as a held-out eval. See GUIDE.txt section 6.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .head import DecisionHead, TemperatureFit, fit_temperature
from .prompt import DecisionRequest, Question, render_request

ORDINAL_NEIGHBOUR_WEIGHT = 0.2


@dataclass
class Example:
    """One rendered question with its target distribution."""

    prompt: str
    target: list[float]
    question_type: str
    n_options: int


def _ordinal_target(index: int, n: int, neighbour: float = ORDINAL_NEIGHBOUR_WEIGHT) -> list[float]:
    """One-hot with ``neighbour`` of the mass on each adjacent level.

    Keeps 1 - 2*neighbour on the label (or 1 - neighbour at an edge). This encodes
    "level 2 is right, but level 1 and 3 are not far off", which is the truth about an
    ordinal judgement.
    """
    target = [0.0] * n
    target[index] = 1.0
    for delta in (-1, 1):
        j = index + delta
        if 0 <= j < n:
            take = min(neighbour, target[index])
            target[index] -= take
            target[j] += take
    return target


def build_example(
    state: Any,
    question: Question,
    label: int | dict[str, Any],
    *,
    soft_targets: bool = False,
    ordinal_score: bool = True,
) -> Example:
    """Render one training example."""
    n = question.n_options
    if isinstance(label, dict):
        if "soft" in label and soft_targets:
            target = [float(x) for x in label["soft"]]
        else:
            target = _ordinal_target(int(label["index"]), n) if question.type == "score" \
                else [1.0 if i == int(label["index"]) else 0.0 for i in range(n)]
    else:
        target = ([1.0 if i == int(label) else 0.0 for i in range(n)]
                  if question.type != "score" or not ordinal_score
                  else _ordinal_target(int(label), n))
    total = sum(target)
    if total <= 0:
        raise ValueError("target distribution sums to zero")
    target = [t / total for t in target]
    prompt = render_request(DecisionRequest(state=state, questions=[question]))
    return Example(prompt=prompt, target=target, question_type=question.type, n_options=n)


def collate(batch: list[Example], tokenizer, max_length: int = 2048, device: str = "cpu"):
    """Tokenise, pad, and build the padded-slot count tensor the head needs."""
    encoded = [
        tokenizer(e.prompt, add_special_tokens=False, truncation=True, max_length=max_length).input_ids
        for e in batch
    ]
    width = max(len(ids) for ids in encoded)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
    # Left-pad so the last position is always real, matching how we read the hidden state.
    for row, ids in enumerate(encoded):
        input_ids[row, width - len(ids):] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, width - len(ids):] = 1
    # Pad targets out to the head's slot width, zeroing the padded slots. Every batch
    # has a different true option count, so a ragged target tensor would not broadcast
    # against the (batch, max_options) logits. Zeros are safe: the padded logits are
    # -1e30, so those terms contribute exactly 0 to the loss.
    slot_width = max(e.n_options for e in batch)
    targets = torch.zeros((len(batch), slot_width), dtype=torch.float32)
    for row, e in enumerate(batch):
        targets[row, : e.n_options] = torch.tensor(e.target, dtype=torch.float32)
    counts = torch.tensor([e.n_options for e in batch], dtype=torch.long)
    return (input_ids.to(device), attention_mask.to(device), targets.to(device), counts.to(device))


def parse_jsonl(path: str) -> list[Example]:
    """Read a JSONL training file into examples. See the module docstring for the shape."""
    examples: list[Example] = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            state = row.get("state")
            for key, spec in row.get("questions", {}).items():
                if "label" not in spec:
                    continue
                options = tuple(spec.get("options") or ())
                q = Question(
                    key=key,
                    instructions=spec.get("instructions", ""),
                    type=spec.get("type", "choice"),
                    options=options,
                    criteria=spec.get("criteria", {}) or {},
                )
                examples.append(build_example(state, q, spec["label"]))
    if not examples:
        raise ValueError(f"{path} produced no examples; every question needs a `label`")
    return examples


class DecisionTrainer:
    """Minimal, explicit training loop.

    Deliberately not built on TRL's ``SFTTrainer``: that trainer computes a causal-LM
    loss over target tokens, and this task has no target tokens -- the loss is a
    cross-entropy over option-slot logits at one position. Writing the loop out keeps
    the loss, the masking and the temperature handling visible, which is where this
    class of model actually goes wrong.
    """

    def __init__(
        self,
        model,
        head: DecisionHead,
        *,
        lora_root=None,
        lr: float = 1e-4,
        head_lr: float = 1e-4,
        weight_decay: float = 0.0,
        warmup_ratio: float = 0.1,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.head = head
        self.device = device
        self._lora_root = lora_root

        # `model` is the text backbone we forward through. When a LoRA adapter is
        # attached it wraps a parent, so the trainable parameters live on that parent,
        # not on the inner module -- enumerate from the parent when one is given.
        param_root = lora_root if lora_root is not None else model

        lora_params = [p for p in param_root.parameters() if p.requires_grad]
        head_params = list(head.parameters())
        n_lora = sum(p.numel() for p in lora_params)
        n_head = sum(p.numel() for p in head_params)
        print(f"  trainable: LoRA {n_lora:,} + head {n_head:,} "
              f"= {n_lora + n_head:,} parameters")
        if n_lora == 0:
            print("  WARNING: no LoRA parameters are trainable -- the head alone will "
                  "train, which usually underfits. Check --lora-r.")

        self.optim = torch.optim.AdamW(
            [
                {"params": lora_params, "lr": lr},
                {"params": head_params, "lr": head_lr},
            ],
            weight_decay=weight_decay,
        )
        self.lr, self.head_lr = lr, head_lr
        self.warmup_ratio = warmup_ratio
        self._step = 0
        self._total = 0

    def _param_root(self):
        """Module that owns the trainable adapter parameters."""
        return self._lora_root if self._lora_root is not None else self.model

    def _set_lr(self, step: int) -> None:
        warmup = max(1, int(self._total * self.warmup_ratio))
        if step < warmup:
            scale = step / warmup
        else:
            progress = (step - warmup) / max(1, self._total - warmup)
            scale = max(0.1, 1.0 - progress * 0.9)  # linear decay to 10% of peak
        for group in self.optim.param_groups:
            base = self.lr if group is self.optim.param_groups[0] else self.head_lr
            group["lr"] = base * scale

    def train(
        self,
        examples: list[Example],
        *,
        epochs: int = 1,
        batch_size: int = 8,
        accum: int = 1,
        max_length: int = 2048,
        seed: int = 17,
        log_every: int = 20,
        tokenizer=None,
    ) -> dict[str, Any]:
        from transformers import AutoTokenizer

        tokenizer = tokenizer or AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-350M")
        rng = random.Random(seed)
        torch.manual_seed(seed)
        self._total = max(1, math.ceil(len(examples) / batch_size) * epochs // accum)

        self.model.train()
        self.head.train()
        history: list[dict[str, float]] = []
        for epoch in range(epochs):
            order = list(range(len(examples)))
            rng.shuffle(order)
            running, seen = 0.0, 0
            for start in range(0, len(order), batch_size):
                batch = [examples[i] for i in order[start:start + batch_size]]
                input_ids, mask, targets, counts = collate(
                    batch, tokenizer, max_length, self.device)
                hidden = _last_hidden(self.model, input_ids, mask, self.head.hidden_size)
                logits = self.head(hidden, counts)
                # Narrow to this batch's widest real option count. The head always
                # emits `max_options` slots; the targets only span the options this
                # batch actually has, and the padded logits are -1e30 so dropping them
                # changes nothing numerically.
                logits = logits[:, : targets.shape[1]]
                logp = torch.log_softmax(logits, dim=-1)
                # Soft-target cross-entropy: -sum_k target_k * log p_k. Works for both
                # one-hot labels and a soft/ordinal distribution.
                loss = -(targets.to(logp.device) * logp).sum(dim=-1).mean() / accum
                loss.backward()
                running += float(loss.item()) * accum
                seen += 1
                if (start // batch_size + 1) % accum == 0:
                    trainable = [p for p in self._param_root().parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(
                        trainable + list(self.head.parameters()), 1.0)
                    self._set_lr(self._step)
                    self.optim.step()
                    self.optim.zero_grad(set_to_none=True)
                    self._step += 1
                    if self._step % log_every == 0:
                        print(f"  epoch {epoch + 1}/{epochs} step {self._step}/{self._total} "
                              f"loss {running / max(seen, 1):.4f} "
                              f"lr {self.optim.param_groups[0]['lr']:.2e}")
                        history.append({"step": self._step, "loss": running / max(seen, 1)})
                        running, seen = 0.0, 0
        self.model.eval()
        self.head.eval()
        return {"history": history, "steps": self._step}


def resolve_backbone(model) -> Any:
    """Return the module whose ``forward`` yields ``last_hidden_state``.

    ``Lfm2ForCausalLM`` (and every causal-LM wrapper) returns ``logits`` of width
    ``vocab_size``. The decision head needs a hidden vector of width ``hidden_size``,
    which lives on the inner text stack. This unwraps the PEFT wrapper if one is
    attached, so the returned module is still inside the LoRA graph and the adapter
    still receives gradients.

    Tries, in order: the PEFT base-model wrapper, then ``.model`` / ``.backbone`` /
    ``.transformer`` / ``.text_model``, then the model itself.
    """
    inner = getattr(model, "base_model", None)
    if inner is not None and hasattr(inner, "model") and not hasattr(inner, "layers"):
        inner = inner.model
    candidates = []
    for base in (model, inner):
        if base is None:
            continue
        for attr in ("model", "backbone", "transformer", "text_model"):
            cand = getattr(base, attr, None)
            if cand is not None:
                candidates.append(cand)
        candidates.append(base)
    for cand in candidates:
        if cand is not None and hasattr(cand, "layers"):
            return cand
    raise RuntimeError(
        f"could not find a text backbone with a `.layers` attribute on "
        f"{type(model).__name__}; the decision head has nothing to read"
    )

def _last_hidden(model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 expected_width: int) -> torch.Tensor:
    """Run ``model`` and return its last-position hidden state, validating the width.

    A causal-LM wrapper can return either ``last_hidden_state`` (width ``hidden_size``)
    or ``logits`` (width ``vocab_size``). The decision head wants the former. Selecting
    the wrong one is the most common way to end up with a silently broken decision
    model, so this checks the width and names the fix instead of letting it broadcast.
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    hidden = getattr(out, "last_hidden_state", None)
    if hidden is None:
        hidden = out[0] if isinstance(out, (tuple, list)) else None
    if hidden is None:
        if getattr(out, "logits", None) is not None:
            raise RuntimeError(
                "the backbone returned logits, not hidden states. Read the hidden state "
                "from the text backbone (e.g. model.model or model.backbone), not from "
                "the causal-LM wrapper."
            )
        raise RuntimeError(f"cannot find a hidden state in output of type {type(out)}")
    if hidden.shape[-1] != expected_width:
        raise RuntimeError(
            f"hidden width {hidden.shape[-1]} does not match the head's {expected_width}; "
            "the head is attached to the wrong module"
        )
    return hidden[:, -1].float()



@torch.inference_mode()
def fit_temperatures(
    model,
    head: DecisionHead,
    examples: list[Example],
    *,
    tokenizer=None,
    max_length: int = 2048,
    device: str = "cuda",
) -> TemperatureFit:
    """Fit one temperature per question type on a held-out calibration split.

    Fit per type, not globally: the three question types produce differently-shaped logit
    distributions and every implementation that reports calibration fits them separately
    (Jebadiah ships choice 1.12 / noul 1.33 / score 1.20).
    """
    from transformers import AutoTokenizer

    tokenizer = tokenizer or AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-350M")
    model.eval()
    head.eval()
    buckets: dict[str, list[tuple[torch.Tensor, int]]] = {}
    for start in range(0, len(examples), 8):
        batch = examples[start:start + 8]
        input_ids, mask, targets, counts = collate(batch, tokenizer, max_length, device)
        hidden = _last_hidden(model, input_ids, mask, head.hidden_size)
        logits = head(hidden, counts)
        valid = torch.arange(logits.shape[1])[None, :] < counts[:, None]
        logits = logits.masked_fill(~valid, -1e30)
        for row, ex in enumerate(batch):
            buckets.setdefault(ex.question_type, []).append(
                (logits[row, : ex.n_options].cpu(), int(targets[row].argmax().item()))
            )
    fit = TemperatureFit()
    for qtype, rows in buckets.items():
        logits = torch.stack([r[0] for r in rows])
        labels = torch.tensor([r[1] for r in rows])
        t = fit_temperature(logits, labels)
        setattr(fit, qtype, t)
        print(f"  fitted T[{qtype}] = {t:.3f}  (n={len(rows)})")
    fit.fitted_on = f"{len(examples)} held-out examples"
    return fit


def attach_lora(model, *, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    """Wrap the backbone in a LoRA adapter.

    ``target_modules`` is deliberately ``all-linear``: on hybrid LFM2.5 backbones the
    short-convolution blocks have their own projections, and an adapter that only
    touches the attention layers leaves most of the model frozen. Kev's trainer picks
    the right targets from the model config for the same reason.
    """
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules="all-linear",
        bias="none",
    )
    return get_peft_model(model, config)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backbone", default="LiquidAI/LFM2.5-350M")
    p.add_argument("--revision", default=None, help="pin a revision; strongly recommended")
    p.add_argument("--data", required=True, help="training JSONL")
    p.add_argument("--calibration", default=None,
                   help="held-out JSONL for temperature fitting (never train on this)")
    p.add_argument("--out", default="runs/jeb-nano")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-options", type=int, default=256)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--limit", type=int, default=None, help="cap examples (smoke runs)")
    args = p.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 72)
    print("Jeb-Omni-Nano trainer")
    print("=" * 72)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.backbone, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.backbone, revision=args.revision, dtype=torch.bfloat16
        if args.device == "cuda" else torch.float32)
    model.config.use_cache = False

    examples = parse_jsonl(args.data)
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    if args.calibration:
        cal = parse_jsonl(args.calibration)
    else:
        cut = max(1, int(len(examples) * 0.05))
        cal, examples = examples[:cut], examples[cut:]
        print(f"  no --calibration given: holding out {cut} examples ({cut / len(examples):.1%})")
    if args.limit:
        examples = examples[: args.limit]
    print(f"  train examples : {len(examples):,}")
    print(f"  calibration    : {len(cal):,} (never trained on)")

    # Attach LoRA to the full model, then train against the INNER text backbone.
    # Lfm2ForCausalLM.forward returns logits (vocab-width), not hidden states, so
    # training through the causal-LM wrapper cannot work. The inner `model` attribute
    # of a LFM2 causal LM is the text stack that returns last_hidden_state, and it is
    # inside the LoRA wrapper, so the adapter still receives gradients.
    if args.lora_r > 0:
        model = attach_lora(model, r=args.lora_r, alpha=args.lora_alpha,
                            dropout=args.lora_dropout)
        model.to(args.device)
    hidden_size = int(getattr(model.config, "hidden_size",
                              getattr(model.config, "text_config", model.config).hidden_size))
    backbone = resolve_backbone(model)
    print(f"  backbone       : {args.backbone}")
    print(f"  training on    : {type(backbone).__name__} (returns last_hidden_state)")
    print(f"  hidden_size    : {hidden_size}")
    print(f"  max_options    : {args.max_options}")

    trainer = DecisionTrainer(backbone, head, lora_root=model, lr=args.lr,
                              head_lr=args.head_lr, device=args.device)
    print("\nTraining:")
    trainer.train(examples, epochs=args.epochs, batch_size=args.batch_size,
                  accum=args.accum, max_length=args.max_length, seed=args.seed,
                  tokenizer=tokenizer)

    print("\nFitting temperatures on the calibration split:")
    temperatures = fit_temperatures(backbone, head, cal, tokenizer=tokenizer,
                                    max_length=args.max_length, device=args.device)

    os.makedirs(args.out, exist_ok=True)
    torch.save({"head": head.state_dict(), "temperatures": temperatures.to_dict(),
                "config": vars(args)}, os.path.join(args.out, "head.pt"))
    if args.lora_r > 0:
        model.save_pretrained(os.path.join(args.out, "adapter"))
    tokenizer.save_pretrained(os.path.join(args.out, "tokenizer"))
    with open(os.path.join(args.out, "temperatures.json"), "w") as f:
        json.dump(temperatures.to_dict(), f, indent=2)
    print(f"\nSaved head + temperatures + adapter to {args.out}/")
    print("Next: evaluate BEFORE reporting any number. See GUIDE.txt section 8.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
