"""Smoke test: proves the head, prompt contract and calibration code actually run.

No model weights and no network needed. Runs the DecisionHead on random features to
verify masking, temperature fitting, ECE and Brier behave as documented, and locks the
prompt contract with a hash.

Run: python -m tests.test_smoke
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, ".")

from jeb_nano import (  # noqa: E402
    DecisionHead,
    Question,
    DecisionRequest,
    brier_score,
    expected_calibration_error,
    fit_temperature,
    permute,
    probabilities,
    prompt_sha256,
    render_request,
    render_single_question,
)

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILED.append(name)


def test_head_shapes_and_masking() -> None:
    print("\n1. Head shapes, masking and padding")
    h = DecisionHead(hidden_size=1024)
    n_params = sum(p.numel() for p in h.parameters())
    check("head param count is hidden*max_options + max_options",
          n_params == 1024 * 256 + 256, f"{n_params:,} params")
    check("head is ~0.25% of a 350M backbone", n_params / 354_500_000 < 0.001,
          f"{n_params / 354_500_000:.4%}")

    feats = torch.randn(4, 1024)
    counts = torch.tensor([2, 5, 256, 3])
    logits = h(feats, counts)
    check("logits shape is (batch, max_options)", logits.shape == (4, 256), str(tuple(logits.shape)))
    check("logits are float32", logits.dtype == torch.float32, str(logits.dtype))

    for row, k in enumerate(counts.tolist()):
        p = probabilities(logits[row : row + 1], 1.0)[0]
        check(f"row {row}: probs sum to 1", abs(float(p.sum()) - 1.0) < 1e-5, f"{float(p.sum()):.6f}")
        check(f"row {row}: padded slots are exactly 0", bool((p[k:] == 0).all()))
        check(f"row {row}: real slots are > 0", bool((p[:k] > 0).all()))

    # smaller head still works
    small = DecisionHead(hidden_size=64, max_options=8)
    lg = small(torch.randn(2, 64), torch.tensor([3, 8]))
    check("max_options=8 head masks at 8", bool((lg[0, 3:] == -1e30).all()))


def test_temperature_monotonicity() -> None:
    print("\n2. Temperature scaling behaviour")
    feats = torch.randn(1, 1024)
    logits = DecisionHead(hidden_size=1024)(feats, torch.tensor([4]))[0]
    p_low = probabilities(logits[None], 0.5)[0]
    p_mid = probabilities(logits[None], 1.0)[0]
    p_high = probabilities(logits[None], 4.0)[0]
    spread = lambda p: float(p.max() - p.min())  # noqa: E731
    check("lower T sharpens", spread(p_low) > spread(p_mid), f"{spread(p_low):.4f} > {spread(p_mid):.4f}")
    check("higher T softens", spread(p_high) < spread(p_mid), f"{spread(p_high):.4f} < {spread(p_mid):.4f}")
    check("argmax is invariant to T", int(p_low.argmax()) == int(p_high.argmax()))


def test_fit_temperature_recovers_known_value() -> None:
    print("\n3. Temperature fitting recovers a planted value")
    torch.manual_seed(0)
    n, k = 20000, 5
    true_t = 2.5
    # A model whose true predictive distribution IS softmax(z / T). If the labels are
    # sampled from that distribution, NLL is minimised at exactly T -- that is the
    # whole theorem behind temperature scaling, so this is a real test, not a
    # tautology. Sampling is what makes it non-trivial: argmax-only labels would
    # drive NLL monotonically to the grid edge.
    z = torch.randn(n, k) * 1.5
    p_true = torch.softmax(z / true_t, dim=-1)
    labels = torch.multinomial(p_true, 1).squeeze(1)
    fitted = fit_temperature(z, labels)
    check("recovers a planted T=2.5 within 10%",
          abs(fitted - true_t) / true_t < 0.10, f"fitted={fitted:.3f} vs true={true_t}")

    # Over-confident model: the logits imply softmax(z/1) but the truth is a softer
    # softmax(z/2), so the model claims more certainty than it has and the fit must
    # land above 1. (Sampling truth at T=1 would by construction be *correctly*
    # calibrated, which is why the first case above is the real test.)
    soft_labels = torch.multinomial(torch.softmax(z / 2.0, dim=-1), 1).squeeze(1)
    over = fit_temperature(z, soft_labels)
    check("over-confident logits fit T > 1", over > 1.0, f"T={over:.3f}")


def test_answer_confidence_formulas() -> None:
    """Confidence must match TypeSafe's reference adapter, which Kev reproduces.

    The Kev README publishes a worked response, so these are checked against real
    published numbers rather than against our own implementation.
    """
    print("\n7. Choice / score / noul answer formulas vs published examples")
    from jeb_nano.model import _mean_confidence, _score_confidence, _to_answer

    # Kev README: probabilities {returns: 0.47, shipping: 0.28, billing: 0.25} -> 0.21
    c = _mean_confidence(torch.tensor([0.47, 0.28, 0.25]))
    check("choice confidence matches Kev's worked example (0.21)",
          abs(c - 0.21) < 0.01, f"{c:.4f}; exact formula gives 0.2050, doc rounds to 0.21")
    check("uniform choice confidence is 0",
          abs(_mean_confidence(torch.tensor([1 / 3, 1 / 3, 1 / 3]))) < 1e-6)
    check("one-hot choice confidence is 1",
          abs(_mean_confidence(torch.tensor([1.0, 0.0, 0.0])) - 1.0) < 1e-6)

    # Kev README: score probs {0: 0.00, 1: 0.56, 2: 0.44} -> score 1.44, confidence 0.34
    sq = Question(key="frustration", instructions="How frustrated is the customer?",
                  type="score", options=("Calm", "Frustrated", "Very angry"))
    a = _to_answer(sq, torch.tensor([0.00, 0.56, 0.44]))
    check("expected score matches Kev's worked example (1.44)",
          abs(a.expected_score - 1.44) < 0.01, f"{a.expected_score:.4f}")
    check("score confidence matches Kev's worked example (0.34)",
          abs(a.confidence - 0.34) < 0.01, f"{a.confidence:.4f}")
    check("all mass on one level -> confidence 1",
          abs(_score_confidence(torch.tensor([0.0, 0.0, 1.0])) - 1.0) < 1e-6)
    check("uniform spread -> confidence 0",
          abs(_score_confidence(torch.tensor([1 / 3, 1 / 3, 1 / 3]))) < 1e-6)

    nq = Question(key="escalate", instructions="Does this need urgent attention?", type="noul")
    na = _to_answer(nq, torch.tensor([0.93, 0.07]))
    check("noul returns P(true) as confidence", abs(na.confidence - 0.93) < 1e-6)
    check("noul prediction is Yes", na.prediction == "Yes")
    check("noul probabilities sum to 1", abs(sum(na.probabilities.values()) - 1.0) < 1e-6)


def test_option_buckets_and_shrinkage() -> None:
    """The bucket axes and the shrinkage rule, checked against published values.

    Every expected number here is copied from a real, published calibrator rather
    than invented: `chaoliangUNSW/Jev-Style-0.8B-Decision-v3`'s readout_config.json.
    """
    print("\n8. Option buckets and temperature shrinkage")
    from jeb_nano import SHRINKAGE_K, TemperatureFit, option_bucket, shrink_temperature

    check("bucket boundaries match the published convention",
          [option_bucket(k) for k in (2, 3, 5, 6, 10, 11, 20, 21, 99)]
          == ["2", "3-5", "3-5", "6-10", "6-10", "11-20", "11-20", "21+", "21+"],
          str([option_bucket(k) for k in (2, 3, 5, 6, 10, 11, 20, 21, 99)]))
    check("shrinkage k is 100", SHRINKAGE_K == 100.0, str(SHRINKAGE_K))

    # The published calibrator stores weight = n / (n + 100) per group. Verify that
    # reproduces its stored, already-shrunk temperatures from its raw ones.
    # Group general|choice|3-5: T_raw 0.85249625, stored T 0.856380, n 600
    #   (600/700 = 0.8571428... which is the weight this file records for it)
    got = shrink_temperature(0.85249625, 0.8800546821789332, 600)
    check("log-space blend reproduces a published group (n=600)",
          abs(got - 0.856380) < 1e-4, f"got {got:.6f}, published 0.856380")
    # Group intent|choice|11-20: T_raw 0.85304, stored T 0.853659, n 4233
    got2 = shrink_temperature(0.85304, 0.8800546821789332, 4233)
    check("log-space blend reproduces a large group (n=4233)",
          abs(got2 - 0.853659) < 1e-4, f"got {got2:.6f}, published 0.853659")
    # Group long|choice|2: T_raw 2.53276, stored T 1.01016, n 15.
    # This is the group that discriminates geometric from arithmetic: its weight is
    # 15/115 = 0.1304, so a linear blend keeps most of the huge raw value.
    got3 = shrink_temperature(2.53276, 0.8800546821789332, 15)
    check("small-n group is pulled hard toward the global value",
          abs(got3 - 1.01016) < 1e-4, f"got {got3:.6f}, published 1.01016")
    arith3 = 0.1304347826 * 2.53276 + 0.8695652174 * 0.8800546821789332
    check("an arithmetic blend is decisively wrong on that group (blend is geometric)",
          abs(arith3 - 1.01016) > 1e-2, f"arithmetic would give {arith3:.6f}")

    check("n=0 falls back to the global value",
          abs(shrink_temperature(3.0, 0.88, 0) - 0.88) < 1e-9)
    check("huge n keeps the raw value",
          abs(shrink_temperature(2.5, 0.88, 10**9) - 2.5) < 1e-6)

    fit = TemperatureFit(global_=1.5, by_type={"choice": 2.0},
                         by_bucket={"2": 5.0069, "21+": 1.5144})
    check("bucket wins over type when both exist",
          abs(fit.get("choice", 2) - 5.0069) < 1e-9, f"{fit.get('choice', 2)}")
    check("type wins when the bucket is absent",
          abs(fit.get("choice", 7) - 2.0) < 1e-9, f"{fit.get('choice', 7)}")
    check("global is the final fallback",
          abs(fit.get("score", 7) - 1.5) < 1e-9, f"{fit.get('score', 7)}")
    check("T may be below 1 (underconfident models sharpen)",
          TemperatureFit(global_=0.8800546821789332).get() < 1.0)
    try:
        fit.get("nope", 2)
        check("unknown question type rejected", False)
    except ValueError:
        check("unknown question type rejected", True)

    rt = TemperatureFit.from_dict(fit.to_dict())
    check("to_dict/from_dict round-trips",
          rt.get("choice", 2) == fit.get("choice", 2)
          and rt.get("choice", 7) == fit.get("choice", 7)
          and rt.get("score", 7) == fit.get("score", 7))
    check("serialised dict names the shrinkage constant",
          fit.to_dict()["shrinkage_k"] == 100.0)
    check("T lookup is stable for a 2-option noul", abs(
        TemperatureFit(global_=1.0).get("noul", 2) - 1.0) < 1e-9)


def test_metrics_known_values() -> None:
    print("\n4. ECE and Brier on hand-checkable cases")
    # perfectly confident and perfectly correct -> ECE 0
    probs = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    labels = torch.tensor([0, 0])
    check("ECE = 0 for perfect confident+correct", abs(expected_calibration_error(probs, labels)) < 1e-6)
    # perfectly confident and perfectly wrong -> ECE 1
    probs_bad = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    labels_bad = torch.tensor([1, 1])
    check("ECE = 1 for confident+wrong", abs(expected_calibration_error(probs_bad, labels_bad) - 1.0) < 1e-6)
    # uniform 2-way -> Brier = (0.5-1)^2+(0.5-0)^2)/2 = 0.5/2 = 0.25
    uni = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    check("Brier = 0.25 for uniform 2-way", abs(brier_score(uni, labels) - 0.25) < 1e-6,
          f"{brier_score(uni, labels):.4f}")
    # perfect one-hot -> Brier 0
    onehot = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    check("Brier = 0 for perfect one-hot", abs(brier_score(onehot, torch.tensor([0, 1]))) < 1e-6)


def test_prompt_contract() -> None:
    print("\n5. Prompt contract")
    q = Question(key="route", instructions="Which team should take this ticket?",
                 options=("billing", "support", "sales"))
    req = DecisionRequest(state={"subject": "Charged twice"}, questions=[q])
    text = render_request(req)
    check("state is rendered from the dict", "subject: \"Charged twice\"" in text, )
    check("options are numbered from 1", "1. billing" in text and "3. sales" in text)
    check("reply instruction names the count", "(1-3)" in text)

    single = render_single_question("Some state text", q)
    check("single-question render has one QUESTION", single.count("QUESTION:") == 1)
    check("multi-question render has N QUESTIONs", render_request(
        DecisionRequest(state="s", questions=[q, q])).count("QUESTION:") == 2)

    h1, h2 = prompt_sha256(req), prompt_sha256(req)
    check("prompt hash is stable", h1 == h2, h1[:16])
    check("prompt hash is 64 hex chars", len(h1) == 64)

    # noul gets Yes/No by default
    nq = Question(key="urgent", instructions="Does this need urgent attention?", type="noul")
    check("noul defaults to Yes/No options", nq.options == ("Yes", "No"), str(nq.options))

    # permutation is a real permutation
    pq = permute(q, [2, 0, 1])
    check("permute reorders options", pq.options == ("sales", "billing", "support"), str(pq.options))
    check("permute preserves the option set", set(pq.options) == set(q.options))
    check("permute changes the prompt hash", prompt_sha256(
        DecisionRequest(state=req.state, questions=[pq])) != h1)

    # validation
    try:
        Question(key="bad", instructions="x", options=("only",))
        check("single-option question rejected", False)
    except ValueError:
        check("single-option question rejected", True)
    try:
        Question(key="bad", instructions="x", options=tuple(f"o{i}" for i in range(300)))
        check(">256 options rejected", False)
    except ValueError:
        check(">256 options rejected", True)


def test_inference_math_matches_reference() -> None:
    print("\n6. Head math matches the Jev-Omni reference implementation")
    torch.manual_seed(7)
    hidden, k = 1024, 7
    head = DecisionHead(hidden_size=hidden)
    with torch.no_grad():
        head.mu.normal_(0, 0.5)
        head.sd.uniform_(0.5, 2.0)
    feats = torch.randn(3, hidden)

    # reference: z = linear((f - mu) / sd); mask; softmax
    z_ref = torch.nn.functional.linear((feats.float() - head.mu) / head.sd, head.linear.weight, head.linear.bias)
    mask = torch.arange(256)[None, :] >= torch.tensor([2, 7, 4])[:, None]
    z_ref = z_ref.masked_fill(mask, -1e30)
    p_ref = torch.softmax(z_ref, dim=-1)

    logits = head(feats, torch.tensor([2, 7, 4]))
    p = probabilities(logits, 1.0)
    check("probability output matches reference bit-for-bit", torch.equal(p, p_ref))
    check("max abs difference is 0", float((p - p_ref).abs().max()) == 0.0)


def main() -> int:
    print("=" * 72)
    print("Jeb-Omni-Nano smoke test")
    print("=" * 72)
    test_head_shapes_and_masking()
    test_temperature_monotonicity()
    test_fit_temperature_recovers_known_value()
    test_metrics_known_values()
    test_prompt_contract()
    test_inference_math_matches_reference()
    test_answer_confidence_formulas()
    test_option_buckets_and_shrinkage()
    print("\n" + "=" * 72)
    if FAILED:
        print(f"FAILED ({len(FAILED)}): " + ", ".join(FAILED))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
