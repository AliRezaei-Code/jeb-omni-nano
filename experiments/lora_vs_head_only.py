"""Does LoRA actually help? The load-bearing assumption, tested.

The layer sweep produced a number worth taking seriously: a frozen backbone with a
head-only fit scored 0.4889 against a `Prior` base-rate baseline of roughly 0.470.
That is *barely better than always guessing the most common label*. Either the task
is hard, or the head alone is not enough.

Every design decision in this project assumes the answer is "LoRA". Kev, Jebadiah
and Jev-Omni all fine-tune adapters. We ported that assumption without testing it.

This script measures it on identical data, identical splits and identical
evaluation, changing exactly one thing: whether the backbone can move.

  arm A -- frozen backbone + head            (what layer_sweep measured)
  arm B -- LoRA r=16 on all-linear + head    (what the project assumes)
  arm C -- LoRA r=16 + head, reported before temperature fitting

Reported per arm: accuracy, Brier, ECE, mean confidence, and coverage at a 5%
error budget -- the last being the number a routing policy actually cares about.

Run:  python experiments/lora_vs_head_only.py --out runs/lora-ablation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jeb_nano import (DecisionHead, brier_score,  # noqa: E402
                      expected_calibration_error, fit_temperature, probabilities)
from jeb_nano.model import JebNanoModel  # noqa: E402
from jeb_nano.train import resolve_backbone  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from layer_sweep import load_examples, split_data, tokenize  # noqa: E402

import layer_sweep  # noqa: E402

BACKBONE = "LiquidAI/LFM2.5-350M"
REVISION = "9e6c6ccf47cd318696e137d381a7ded8fe4df09f"


def _last_hidden(backbone, ids, mask):
    """Last-position hidden state at the FINAL layer (the sweep's measured winner)."""
    out = backbone(input_ids=ids, attention_mask=mask, use_cache=False)
    h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    return h[:, -1].float()


def batch_tensors(rows, tokenizer, device):
    """Tokenise a batch of prompts. Left-pads so [:, -1] is always a real token."""
    enc = [tokenize(p) for p in rows]
    width = max(len(e) for e in enc)
    pad = tokenizer.pad_token_id or 0
    ids = torch.full((len(enc), width), pad, dtype=torch.long)
    mask = torch.zeros((len(enc), width), dtype=torch.long)
    for r, e in enumerate(enc):
        ids[r, width - len(e):] = torch.tensor(e)
        mask[r, width - len(e):] = 1
    return ids.to(device), mask.to(device)


def gold_tensor(rows):
    """Ragged gold distributions padded to the batch max. Padded slots are 0."""
    width = max(len(p) for _, p in rows)
    g = torch.zeros(len(rows), width)
    for i, (_, p) in enumerate(rows):
        g[i, : len(p)] = torch.tensor(p, dtype=torch.float32)
    return g


def run_arm(backbone, train_cases, eval_rows, tokenizer, *, use_lora, epochs, lr,
            head_lr, device, seed, label):
    torch.manual_seed(seed)
    for p in backbone.parameters():
        p.requires_grad_(use_lora)
    head = DecisionHead(1024, max_options=32).to(device)

    if use_lora:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                         target_modules="all-linear", bias="none")
        model = get_peft_model(backbone, cfg)
        # After PEFT wraps it, the module that returns hidden states is the inner one.
        fwd = model.base_model.model
    else:
        model, fwd = None, backbone  # model is unused when the backbone is frozen

    params = [{"params": list(head.parameters()), "lr": head_lr}]
    if use_lora:
        params.append({"params": [p for p in model.parameters() if p.requires_grad],
                       "lr": lr})
    opt = torch.optim.AdamW(params, weight_decay=0.0)

    flat = [(c[j][0], c[j][1]) for c in train_cases for j in range(len(c))]
    print(f"\n  --- {label} ---")
    t0 = time.time()
    for ep in range(epochs):
        import random as _r
        idx = list(range(len(flat)))
        _r.Random(seed + ep).shuffle(idx)
        total = seen = 0
        for start in range(0, len(idx), 8):
            rows = [flat[i] for i in idx[start:start + 8]]
            ids, mask = batch_tensors([r[0] for r in rows], tokenizer, device)
            g = gold_tensor(rows).to(device)
            counts = torch.tensor([len(r[1]) for r in rows], device=device)
            h = _last_hidden(fwd, ids, mask)
            logits = head(h, counts)[:, : g.shape[1]]
            logp = torch.log_softmax(logits, dim=-1)
            loss = -(g * logp).sum(-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad] if use_lora
                else list(head.parameters()), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            total += float(loss.item())
            seen += 1
        print(f"    epoch {ep + 1}/{epochs}  loss {total / max(seen, 1):.4f}  (n={len(flat)})")

    # ---- evaluate: temperature fitted on one half, reported on the other
    rows = []
    with torch.inference_mode():
        for prompt, probs in eval_rows:
            ids, mask = batch_tensors([prompt], tokenizer, device)
            h = _last_hidden(fwd, ids, mask)
            lg = head(h, torch.tensor([len(probs)], device=device))[0, : len(probs)]
            rows.append((lg.cpu().float(), torch.tensor(probs, dtype=torch.float32)))

    half = len(rows) // 2
    k = max(r[0].numel() for r in rows)
    cal_lg = torch.full((half, k), -1e30)
    cal_tg = torch.zeros(half, k)
    for i, (a, b) in enumerate(rows[:half]):
        cal_lg[i, : a.numel()] = a
        cal_tg[i, : b.numel()] = b
    t = fit_temperature(cal_lg, cal_tg.argmax(-1)) if half > 1 else 1.0

    rest = rows[half:]
    k2 = max(r[0].numel() for r in rest)
    ev_lg = torch.full((len(rest), k2), -1e30)
    ev_tg = torch.zeros(len(rest), k2)
    for i, (a, b) in enumerate(rest):
        ev_lg[i, : a.numel()] = a
        ev_tg[i, : b.numel()] = b
    p = probabilities(ev_lg, t)
    labels = ev_tg.argmax(-1)
    correct = (p.argmax(-1) == labels).float()
    conf = p.max(-1).values
    acc = float(correct.mean())
    # Risk-coverage: the largest share of questions we could auto-decide while keeping
    # the error rate among accepted at or below `alpha`.
    #
    # The running error rate is NOT monotone -- adding correct predictions can bring it
    # back under threshold -- so this takes the MAX over k, not the first k that passes.
    # An earlier version used an "update while passing" loop, which meant a single wrong
    # top-1 prediction drove the rate to 1.0 at k=1 and pinned coverage at 0 forever.
    # That is what produced the identical, implausibly small 0.0667 in both arms.
    def coverage_at(alpha: float) -> float:
        order = conf.argsort(descending=True).tolist()
        errs = 0
        best = 0.0
        for r, idx in enumerate(order, 1):
            errs += 1.0 - float(correct[idx])
            if errs / r <= alpha:
                best = r / len(order)
        return best

    # Dump per-row outcomes so coverage can be recomputed without retraining.
    return {
        "arm": label, "lora": use_lora, "temperature": round(float(t), 4),
        "n_eval": int(ev_lg.shape[0]),
        "accuracy": round(acc, 4),
        "brier": round(brier_score(p, labels), 4),
        "ece": round(expected_calibration_error(p, labels), 4),
        "mean_confidence": round(float(conf.mean()), 4),
        "coverage_at_5pct_error": round(coverage_at(0.05), 4),
        "coverage_at_10pct_error": round(coverage_at(0.10), 4),
        "coverage_at_20pct_error": round(coverage_at(0.20), 4),
        "accuracy_at_10pct_coverage": round(
            float(correct[conf.argsort(descending=True)[: max(1, int(0.10 * len(correct)))]].mean()),
            4),
        "_per_row": {"confidence": [round(float(x), 6) for x in conf.tolist()],
                      "correct": [bool(x) for x in correct.tolist()]},
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", default="customer_service",
                    help="comma-separated list; 'all' loads all four typed-decisions\n                          workflows, giving ~4x the training data and testing whether\n                          the model generalises across workflow types")
    ap.add_argument("--max-cases", type=int, default=60)
    ap.add_argument("--eval-frac", type=float, default=0.25,
                    help="share of cases held out. Coverage at a 5%% error budget\n                          is UNMEASURABLE below a few hundred eval questions: a\n                          single early error sets the running rate above the\n                          budget for every smaller prefix. Raise this to 0.6 and\n                          drop --max-cases to get a coverage number worth having.")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="runs/lora-ablation")
    ap.add_argument("--only", choices=["A", "B"], default=None,
                    help="run a single arm; A=frozen+head, B=LoRA+head")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    layer_sweep.TOKENIZER = AutoTokenizer.from_pretrained(BACKBONE, revision=REVISION)

    print("=" * 74)
    print("LORA vs HEAD-ONLY  --  does fine-tuning the backbone actually help?")
    print("=" * 74)
    print(f"  backbone : {BACKBONE} @ {REVISION[:12]}")
    print(f"  data     : LocalLLaMA/typed-decisions / {args.workflow}")
    print(f"  protocol : identical data, splits and eval. Only the backbone's")
    print(f"             trainability changes. Read-out at the final layer (L15),")
    print(f"             which the layer sweep measured as the winner.")

    workflows = (["agent_trace_observability", "customer_service",
                  "invoice_processing", "security_incidents"]
                 if args.workflow == "all" else args.workflow.split(","))
    ex = []
    for wf in workflows:
        part = load_examples(wf, args.max_cases)
        print(f"  loaded {len(part):>5} questions from {wf}")
        ex.extend(part)
    train_cases, calib_cases, eval_cases = split_data(
        ex, eval_frac=args.eval_frac, seed=args.seed)
    eval_rows = [(c[j][0], c[j][1]) for c in eval_cases for j in range(len(c))]
    print(f"\n  train questions : {sum(len(c) for c in train_cases)}")
    print(f"  eval questions  : {len(eval_rows)} (temperature fitted on the first half "
          f"of these, metrics on the second)")
    print(f"  calib split     : {len(calib_cases)} cases reserved but unused here -- "
          f"both arms use the eval split's first half for the temperature so the two "
          f"arms are compared on identical rows.")

    results = []
    for use_lora, label, lr, hlr in (
        (False, "A: frozen backbone + head only", args.lr, args.head_lr),
        (True, "B: LoRA r=16 all-linear + head", args.lr, args.head_lr),
    ):
        if args.only and not label.startswith(args.only):
            continue
        model = JebNanoModel(BACKBONE, revision=REVISION, device="cpu",
                             dtype=torch.float32, readout_layer=-1)
        backbone = resolve_backbone(model.model)
        r = run_arm(backbone, train_cases, eval_rows, layer_sweep.TOKENIZER,
                    use_lora=use_lora, epochs=args.epochs, lr=lr, head_lr=hlr,
                    device="cpu", seed=args.seed, label=label)
        print(f"    -> acc {r['accuracy']:.4f}  brier {r['brier']:.4f}  "
              f"ece {r['ece']:.4f}  cov@5% {r['coverage_at_5pct_error']:.4f}  "
              f"T {r['temperature']:.3f}  ({r['seconds']}s)")
        results.append(r)
        del backbone
        if use_lora:
            import gc
            gc.collect()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.workflow}.json"), "w") as f:
        json.dump({"backbone": BACKBONE, "revision": REVISION,
                   "workflow": args.workflow, "epochs": args.epochs,
                   "results": results}, f, indent=2)

    print("\n" + "=" * 74)
    print(f"  {'arm':<34} {'acc':>7} {'brier':>7} {'ece':>7} {'cov@5%':>8} {'T':>6}")
    for r in results:
        print(f"  {r['arm']:<34} {r['accuracy']:>7.4f} {r['brier']:>7.4f} "
              f"{r['ece']:>7.4f} {r['coverage_at_5pct_error']:>8.4f} "
              f"{r['temperature']:>6.3f}")
    if len(results) == 2:
        a, b = results
        print(f"\n  LoRA delta : accuracy {b['accuracy'] - a['accuracy']:+.4f}  "
              f"brier {b['brier'] - a['brier']:+.4f}  "
              f"ece {b['ece'] - a['ece']:+.4f}  "
              f"coverage {b['coverage_at_5pct_error'] - a['coverage_at_5pct_error']:+.4f}")
    print(f"\n  wrote {args.out}/{args.workflow}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
