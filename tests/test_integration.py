"""Integration test against the real LiquidAI/LFM2.5-350M backbone.

Skips cleanly if torch/transformers 5.x or the weights are unavailable, so it can live
in the normal test run without being a hard dependency.

What it actually proves:
  1. The forward hook fires on a real LFM2.5 text backbone and returns a usable
     ``(batch, hidden_size)`` last-position hidden state.
  2. The head attaches at the right width (1024) and is negligible in size.
  3. Inference is deterministic, which every published implementation in this class
     treats as a hard requirement ("the same question must get the same answer").
  4. A single forward pass returns a full distribution -- no generation, no parsing.

What it does NOT prove: accuracy. The head here is randomly initialised, so the
probabilities are meaningless. Training is the subject of ``GUIDE.txt``.

Run: python tests/test_integration.py
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

BACKBONE = "LiquidAI/LFM2.5-350M"
# Pinned. RCLD benchmarked against exactly this revision; never benchmark a moving tag.
REVISION = "9e6c6ccf47cd318696e137d381a7ded8fe4df09f"

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILED.append(name)


def main() -> int:
    try:
        import torch
        import transformers
    except ImportError as exc:
        print(f"SKIP: {exc}")
        return 0

    if not hasattr(transformers, "Lfm2ForCausalLM") and "lfm2" not in dir(transformers):
        print(f"SKIP: this transformers ({transformers.__version__}) has no lfm2 model type; "
              "LFM2.5 needs transformers>=5.0.0")
        return 0

    from jeb_nano import DecisionRequest, Question, render_request
    from jeb_nano.model import JebNanoModel

    print("=" * 72)
    print(f"Jeb-Omni-Nano integration test :: {BACKBONE} @ {REVISION[:12]}")
    print("=" * 72)

    t0 = time.time()
    model = JebNanoModel(BACKBONE, revision=REVISION, device="cpu", dtype=torch.float32)
    print(f"\n  loaded in {time.time() - t0:.1f}s")
    print(f"  hidden_size={model.hidden_size}  (expect 1024)")
    print(f"  head params={sum(p.numel() for p in model.head.parameters()):,}")
    print(f"  backbone params={sum(p.numel() for p in model.model.parameters()):,}")

    check("hidden size is 1024", model.hidden_size == 1024, str(model.hidden_size))
    head_p = sum(p.numel() for p in model.head.parameters())
    check("head is 262,400 params (1024*256 + 256)", head_p == 262_400, f"{head_p:,}")
    frac = head_p / sum(p.numel() for p in model.model.parameters())
    check("head is under 0.1% of the backbone", frac < 0.001, f"{frac:.4%}")

    state = "The meeting starts at 10 AM. It is now 9 AM."
    q = Question(key="started", instructions="Has the meeting started?", type="noul")
    request = DecisionRequest(state=state, questions=[q])
    n_tokens = model._encode(render_request(request)).shape[1]
    check("prompt tokenises to a sane length", 10 <= n_tokens <= 200, f"{n_tokens} tokens")

    t1 = time.time()
    a = model.decide(state, q)
    ms = (time.time() - t1) * 1000
    check("decide() returns without raising", True)
    check("a distribution is returned for every option",
          set(a.probabilities) == set(q.options), str(sorted(a.probabilities)))
    check("probabilities sum to 1", abs(sum(a.probabilities.values()) - 1.0) < 1e-5,
          f"{sum(a.probabilities.values()):.6f}")
    check("noul answer is one of the options", a.prediction in q.options, a.prediction)
    check("confidence is a probability", 0.0 <= a.confidence <= 1.0, f"{a.confidence:.4f}")

    b = model.decide(state, q)
    check("inference is deterministic across calls", b.probabilities == a.probabilities)

    # a 3-way choice exercises the masking path with a different K
    q3 = Question(key="route", instructions="Which team should take this ticket?",
                  options=("billing", "support", "sales"))
    c = model.decide({"subject": "Charged twice", "body": "Two charges on my card."}, q3)
    check("3-way choice sums to 1", abs(sum(c.probabilities.values()) - 1.0) < 1e-5)
    check("choice confidence is in [0,1]", 0.0 <= c.confidence <= 1.0, f"{c.confidence:.4f}")

    # score type
    qs = Question(key="urgency", instructions="How urgent is this ticket?", type="score",
                  options=("can wait", "this week", "today"))
    s = model.decide("Server is down for everyone.", qs)
    check("score type returns expected_score", s.expected_score is not None,
          f"{s.expected_score:.4f}")
    check("expected_score is within the level range",
          s.expected_score is not None and 0 <= s.expected_score <= 2,
          f"{s.expected_score:.4f}")

    print(f"\n  single-question latency: {ms:.0f} ms "
          f"(fp32 CPU, untrained head, no causal_conv1d kernel installed)")


    # Readout-layer sweep. Nokia's AnyJev reports a middle layer beats the last one
    # for a linear head; this proves the option is wired up and actually reads a
    # different layer on real weights.
    print("\n  readout-layer sweep (Nokia/AnyJev finding: try a middle layer):")
    seen = {}
    for layer in (-1, -8, -12):
        m2 = JebNanoModel(BACKBONE, revision=REVISION, device="cpu",
                          dtype=torch.float32, readout_layer=layer)
        check(f"readout_layer={layer} hooks a real decoder layer",
              m2._readout_module is m2._find_text_backbone().layers[layer])
        check(f"readout_layer={layer} registered its hook",
              len(m2._readout_module._forward_hooks) == 1)
        a2 = m2.decide("The server is down for everyone.",
                       Question(key="u", instructions="Urgent?", type="noul"))
        check(f"readout_layer={layer} returns a valid distribution",
              abs(sum(a2.probabilities.values()) - 1.0) < 1e-5)
        seen[layer] = tuple(round(v, 6) for v in a2.probabilities.values())
    check("different readout layers give different activations",
          len(set(seen.values())) == len(seen), str(seen))
    try:
        JebNanoModel(BACKBONE, revision=REVISION, device="cpu",
                     dtype=torch.float32, readout_layer=99)
        check("out-of-range readout_layer rejected", False)
    except ValueError:
        check("out-of-range readout_layer rejected", True)

    print("\n  NOTE: the head is randomly initialised here, so these probabilities are")
    print("  meaningless as predictions. This test proves the wiring, the shapes, the")
    print("  determinism and the cost -- not the accuracy. See GUIDE.txt for training.")

    print("\n" + "=" * 72)
    if FAILED:
        print(f"FAILED ({len(FAILED)}): " + ", ".join(FAILED))
        return 1
    print("All integration checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
