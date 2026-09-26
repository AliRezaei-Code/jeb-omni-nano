"""The readout-layer sweep -- the experiment this project has been deferring.

The question, from Nokia's AnyJev:

    "Cutting Qwen2.5-7B from 28 blocks to 18 left accuracy slightly *higher* and
     calibration better... a middle block is a better feature space for a linear head
     than the last one, where the remaining blocks are busy turning the answer into
     tokens."

Every implementation we studied reads the LAST layer -- including Jev-Omni, whose code
we ported. If a middle layer is better, that is a free accuracy and latency win, and it
also means the layers above the read-out point can be truncated away.

This script measures it on real data:
  * real decision data (LocalLLaMA/typed-decisions, Apache-2.0, soft gold distributions)
  * a real LFM2.5-350M backbone, pinned revision
  * a fitted head per layer, trained only on TRAIN
  * a temperature fitted on a held-out CALIBRATION split
  * every metric reported on a held-out EVAL split the head never saw

Run:  python experiments/layer_sweep.py --out runs/layer-sweep
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jeb_nano import (DecisionHead, Question, brier_score,  # noqa: E402
                      expected_calibration_error, fit_temperature, probabilities)
from jeb_nano.model import JebNanoModel  # noqa: E402
from jeb_nano.prompt import DecisionRequest, render_request  # noqa: E402
from jeb_nano.train import resolve_backbone  # noqa: E402

BACKBONE = "LiquidAI/LFM2.5-350M"
REVISION = "9e6c6ccf47cd318696e137d381a7ded8fe4df09f"


def load_examples(workflow: str, max_cases: int | None = None):
    """Load typed-decisions into (prompt, gold_distribution, n_options) tuples.

    Uses the GOLD DISTRIBUTION, not the argmax. That matters: the benchmark's own
    finding is that carrying the soft target cuts KL by a third, and its Prior row shows
    that argmax alone hides everything about distribution shape.
    """
    from datasets import load_dataset

    out = []
    for split in ("train", "test"):
        ds = load_dataset("LocalLLaMA/typed-decisions", workflow, split=split)
        rows = list(ds)
        if max_cases:
            rows = rows[:max_cases]
        for row in rows:
            state = row["state"]
            questions = json.loads(row["questions"]) if isinstance(row["questions"], str) \
                else row["questions"]
            gold = json.loads(row["gold"]) if isinstance(row["gold"], str) else row["gold"]
            for key, spec in questions.items():
                g = gold.get(key)
                if not isinstance(g, dict) or "probabilities" not in g:
                    continue
                # Real schema, verified against the Hub rather than assumed:
                #   questions[key] = {type, instructions, criteria:{option: text}}
                #   gold[key]     = {type, label, confidence, probabilities:{opt: p}}
                # `criteria` is a DICT for choice/noul ({option: description}) and a
                # LIST for score (ordered level names). Gold keys levels as "0".."4",
                # so a list is stringified to line up with the gold's keys.
                crit = spec.get("criteria")
                if isinstance(crit, dict):
                    options = list(crit.keys())
                elif isinstance(crit, list):
                    options = [str(o) for o in crit]
                else:
                    options = []
                if len(options) < 2:
                    continue
                probs = [float(g["probabilities"].get(o, 0.0)) for o in options]
                total = sum(probs)
                if total <= 0:
                    continue
                probs = [p / total for p in probs]
                q = Question(key=key, instructions=spec.get("instructions", ""),
                             type=spec.get("type", "choice"), options=tuple(options),
                             criteria=(crit if isinstance(crit, dict)
                                        else {o: "" for o in options}))
                out.append((render_request(DecisionRequest(state=state, questions=[q])),
                            probs, split, spec.get("type", "choice")))
    return out


def split_data(examples, calib_frac=0.2, eval_frac=0.25, seed=17):
    """Split by CASE id, not by question, so sibling questions never straddle a split.

    Every case contributes several questions about the same state. Splitting per question
    would leak the state across train and eval.
    """
    by_case: dict[str, list] = {}
    for prompt, probs, split, qtype in examples:
        # Key on the STATE, not the prompt: sibling questions about the same
        # case share a state prefix, and the prompt tail (question + options) differs.
        # Using the full prompt collapsed every sibling into its own bucket, and
        # using prompt[:160] collapsed ALL questions of a case into one.
        key = prompt.split("\n\n---\n\n", 1)[0]
        by_case.setdefault(key, []).append((prompt, probs, qtype))
    cases = list(by_case.values())
    rng = random.Random(seed)
    rng.shuffle(cases)
    n = len(cases)
    n_eval = int(n * eval_frac)
    n_calib = int(n * eval_frac + n * calib_frac)
    return (cases[n_calib:n], cases[:n_calib], cases[:n_eval])


def train_head_on_layer(backbone, hidden, train_cases, layer, epochs=3, batch=8,
                        lr=1e-3, head_lr=1e-3, device="cpu", seed=17):
    """Train a fresh head at a given read-out layer. LoRA is NOT used here.

    Deliberately: this isolates the *read-out layer* as the only variable. Adding LoRA
    would confound the comparison with an adapter-quality difference. A frozen backbone
    plus a trained head is the cleanest possible test of "which layer's representation
    is linearly separable".
    """
    torch.manual_seed(seed)
    head = DecisionHead(hidden, max_options=32).to(device)
    opt = torch.optim.AdamW(
        [{"params": list(backbone.parameters()), "lr": 0.0},  # frozen
         {"params": list(head.parameters()), "lr": head_lr}],
        weight_decay=0.0)
    # only the head actually gets gradients
    for p in backbone.parameters():
        p.requires_grad_(False)
    n = sum(len(c) for c in train_cases)
    for ep in range(epochs):
        idx = [(i, j) for i, c in enumerate(train_cases) for j in range(len(c))]
        random.Random(seed + ep).shuffle(idx)
        total, seen = 0.0, 0
        for start in range(0, len(idx), batch):
            rows = [idx[k] for k in range(start, min(start + batch, len(idx)))]
            prompts = [train_cases[i][j][0] for i, j in rows]
            # Options are ragged across questions (2 for noul, 5 for category),
            # so pad the gold rows to the batch max -- same fix as train.collate.
            raw = [train_cases[i][j][1] for i, j in rows]
            width = max(len(r) for r in raw)
            probs = torch.zeros(len(raw), width, dtype=torch.float32)
            for r, v in enumerate(raw):
                probs[r, : len(v)] = torch.tensor(v, dtype=torch.float32)
            counts = torch.tensor([len(v) for v in raw], device=device)
            enc = [tokenize(p) for p in prompts]
            width = max(len(e) for e in enc)
            pad = TOKENIZER.pad_token_id or 0
            ids = torch.full((len(enc), width), pad, dtype=torch.long)
            mask = torch.zeros((len(enc), width), dtype=torch.long)
            for r, e in enumerate(enc):
                ids[r, width - len(e):] = torch.tensor(e)
                mask[r, width - len(e):] = 1
            h = _last_hidden_at(backbone, ids.to(device), mask.to(device), hidden, layer)
            logits = head(h, counts)
            valid = torch.arange(logits.shape[1])[None, :] < counts[:, None]
            logp = torch.log_softmax(logits.masked_fill(~valid, -1e30), dim=-1)
            # soft-target cross-entropy against the GOLD DISTRIBUTION
            target = torch.zeros_like(logp)
            target[:, : probs.shape[1]] = probs.to(logp.device)  # padded slots are 0
            loss = -(target * logp).sum(-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            total += float(loss.item())
            seen += 1
        print(f"    epoch {ep + 1}/{epochs}  loss {total / max(seen, 1):.4f}  (n={n})")
    return head


def _last_hidden_at(backbone, ids, mask, hidden, layer):
    """Forward the stack and return the last-position hidden state at `layer`.

    Uses a **forward hook on the target layer** rather than iterating
    `backbone.layers` by hand. Manual iteration looks cheaper but is wrong here: LFM2's
    attention layers require `position_embeddings` (cos, sin) to be passed in, and a
    hand-rolled call omits them, so every attention layer raises
    `cannot unpack non-iterable NoneType`. A hook is signature-agnostic and costs one
    full forward instead of a partial one -- the right trade for a correctness
    experiment.
    """
    captured: dict[str, torch.Tensor] = {}

    def hook(_m, _a, out):
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else (
            out[0] if isinstance(out, (tuple, list)) else out)
        captured["h"] = h[:, -1].float()

    handle = backbone.layers[layer].register_forward_hook(hook)
    try:
        backbone(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        handle.remove()
    if "h" not in captured:
        raise RuntimeError(f"the hook on layer {layer} did not fire")
    return captured["h"]


@torch.inference_mode()
def evaluate(backbone, head, cases, layer, hidden, device="cpu", max_len=1024):
    """Accuracy, Brier and ECE on a held-out split, with a temperature fitted on a
    *different* half of that split.

    Rows stay separate rather than being concatenated: option counts are ragged (2 for
    `noul`, 5 for `category`), so `torch.cat` over per-row logits of different widths
    fails. Each row is scored, padded to the batch width locally, and evaluated.
    """
    rows = []  # (logits[K], gold[K]) per question
    for c in cases:
        for prompt, probs, qtype in c:
            enc = tokenize(prompt)[:max_len]
            ids = torch.tensor([enc], device=device)
            mask = torch.ones_like(ids)
            h = _last_hidden_at(backbone, ids, mask, hidden, layer)
            lg = head(h, torch.tensor([len(probs)], device=device))[0, : len(probs)]
            rows.append((lg.cpu().float(), torch.tensor(probs, dtype=torch.float32)))
    if not rows:
        return {}

    half = len(rows) // 2

    def _stack(subset):
        k = max(r[0].numel() for r in subset)
        lg = torch.full((len(subset), k), -1e30)
        tg = torch.zeros(len(subset), k)
        for i, (a, b) in enumerate(subset):
            lg[i, : a.numel()] = a
            tg[i, : b.numel()] = b
        return lg, tg

    cal_lg, cal_tg = _stack(rows[:half])
    t = fit_temperature(cal_lg, cal_tg.argmax(-1)) if half > 1 else 1.0
    ev_lg, ev_tg = _stack(rows[half:])
    if ev_lg.shape[0] == 0:
        return {}
    p = probabilities(ev_lg, t)
    labels = ev_tg.argmax(-1)
    return {
        "layer": layer,
        "n_eval": int(ev_lg.shape[0]),
        "temperature": round(float(t), 4),
        "accuracy": round(float((p.argmax(-1) == labels).float().mean()), 4),
        "brier": round(brier_score(p, labels), 4),
        "ece": round(expected_calibration_error(p, labels), 4),
        "mean_confidence": round(float(p.max(-1).values.mean()), 4),
    }


TOKENIZER = None


def tokenize(text: str):
    return TOKENIZER(text, add_special_tokens=False).input_ids


def main() -> int:
    global TOKENIZER
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", default="customer_service")
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--layers", default="-16,-12,-8,-4,-1",
                    help="negative indices from the end; -1 is the final layer")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--out", default="runs/layer-sweep")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    TOKENIZER = AutoTokenizer.from_pretrained(BACKBONE, revision=REVISION)

    print("=" * 74)
    print("READOUT-LAYER SWEEP  --  does a middle layer beat the last?")
    print("=" * 74)
    print(f"  backbone : {BACKBONE} @ {REVISION[:12]}")
    print(f"  data     : LocalLLaMA/typed-decisions / {args.workflow}")
    print(f"  protocol : frozen backbone, head-only training, so the read-out layer is")
    print(f"             the ONLY variable. T fitted on one half of a held-out set and")
    print(f"             reported on the other half.")

    ex = load_examples(args.workflow, args.max_cases)
    train_cases, calib_cases, eval_cases = split_data(ex, seed=args.seed)
    print(f"\n  cases    : {len(train_cases)} train / {len(calib_cases)} calib / "
          f"{len(eval_cases)} eval")
    print(f"  questions: {sum(len(c) for c in train_cases)} / "
          f"{sum(len(c) for c in calib_cases)} / {sum(len(c) for c in eval_cases)}")

    model = JebNanoModel(BACKBONE, revision=REVISION, device="cpu",
                         dtype=torch.float32, readout_layer=-1)
    # resolve_backbone walks .model/.backbone/... from a raw HF module.
    # JebNanoModel is a wrapper around it, so resolve from the inner model.
    backbone = resolve_backbone(model.model)
    hidden = model.hidden_size
    n_layers = len(backbone.layers)
    print(f"  layers    : {n_layers}")

    layers = [int(x) for x in args.layers.split(",")]
    results = []
    for layer in layers:
        idx = layer if layer >= 0 else n_layers + layer
        label = f"L{idx} (from end {layer})"
        print(f"\n  --- {label} ---")
        t0 = time.time()
        head = train_head_on_layer(backbone, hidden, train_cases, idx,
                                   epochs=args.epochs, seed=args.seed)
        res = evaluate(backbone, head, eval_cases, idx, hidden)
        res["seconds"] = round(time.time() - t0, 1)
        res["label"] = label
        results.append(res)
        print(f"    -> acc {res['accuracy']:.4f}  brier {res['brier']:.4f}  "
              f"ece {res['ece']:.4f}  T {res['temperature']:.3f}  "
              f"({res['seconds']}s, n={res['n_eval']})")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.workflow}.json"), "w") as f:
        json.dump({"backbone": BACKBONE, "revision": REVISION,
                   "workflow": args.workflow, "frozen_backbone": True,
                   "epochs": args.epochs, "results": results}, f, indent=2)

    print("\n" + "=" * 74)
    print("SUMMARY  (frozen backbone, head-only)")
    print(f"  {'layer':<20} {'acc':>8} {'brier':>8} {'ece':>8} {'T':>7} {'sec':>7}")
    for r in sorted(results, key=lambda x: x["layer"]):
        print(f"  {r['label']:<20} {r['accuracy']:>8.4f} {r['brier']:>8.4f} "
              f"{r['ece']:>8.4f} {r['temperature']:>7.3f} {r['seconds']:>7.1f}")
    best_acc = max(results, key=lambda x: x["accuracy"])
    best_ece = min(results, key=lambda x: x["ece"])
    last = [r for r in results if r["layer"] == n_layers - 1]
    print(f"\n  best accuracy : {best_acc['label']}")
    print(f"  best ECE      : {best_ece['label']}")
    if last:
        la = last[0]
        print(f"  last layer    : acc {la['accuracy']:.4f}  brier {la['brier']:.4f}  "
              f"ece {la['ece']:.4f}")
        print(f"  vs last       : best-acc is {best_acc['accuracy'] - la['accuracy']:+.4f} "
              f"accuracy, {best_ece['ece'] - la['ece']:+.4f} ECE")
    print(f"\n  wrote {args.out}/{args.workflow}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
