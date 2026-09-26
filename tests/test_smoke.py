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
    print("\n" + "=" * 72)
    if FAILED:
        print(f"FAILED ({len(FAILED)}): " + ", ".join(FAILED))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
