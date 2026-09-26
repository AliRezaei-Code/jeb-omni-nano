"""Backbone loading and the end-to-end decision path.

This module wires an LFM2.5 backbone to :class:`~jeb_nano.head.DecisionHead` and turns
a :class:`~jeb_nano.prompt.DecisionRequest` into calibrated per-option probabilities
in a **single forward pass**. No tokens are generated, nothing is parsed, and there is
no refusal path.

Backbone choice
---------------
======================================  =======  =====  =========  ==========================
model                                   params   hidden  ctx        note
======================================  =======  =====  =========  ==========================
LiquidAI/LFM2.5-350M                    350M     1024    128k      best cost/latency; default
LiquidAI/LFM2.5-2.6B                    2.69B    2048    128k      if you can afford 4x
LiquidAI/LFM2.5-Encoder-350M            350M     1024    8k        bidirectional; text only
======================================  =======  =====  =========  ==========================

The head costs ``hidden_size * 256 + 256`` parameters: 262,400 on the 350M (0.074% of
the backbone) and 524,544 on the 2.6B. That ratio is the entire point — the expensive
part is the pretrained backbone, and the decision interface is nearly free.

Reading the hidden state
------------------------
We register a forward hook on the backbone and take the **last** token position, exactly
as Jev-Omni does::

    decoder.register_forward_hook(lambda _m, _a, out: capture(
        "hidden", out.last_hidden_state[:, -1].float()))

The last position is the one the model would have predicted the answer at. Because we
read a representation instead of a token, **no decoding happens at all**.

Media
-----
Image, audio and video are attached through the LFM2.5-VL / processor path. Jev-Omni
converts to fixed inputs before the model sees anything, and we keep those conventions
because they are the ones the published MMAU / MVBench numbers were measured under:
audio decoded to 16 kHz mono and truncated to 30 s; video sampled to 16 frames. Note
this differs from the *text-only* LFM2.5 models, which have no media tower at all — for
multimodal you need the LFM2.5-VL checkpoints.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

from .head import DecisionHead, TemperatureFit, probabilities
from .prompt import DecisionRequest, Question, render_request

DEFAULT_BACKBONE = "LiquidAI/LFM2.5-350M"
VISION_BACKBONE = "LiquidAI/LFM2.5-VL-450M"
AUDIO_REVISION = "9e6c6ccf47cd318696e137d381a7ded8fe4df09f"  # example of a pinned rev


@dataclass
class DecisionAnswer:
    """One question's answer: a distribution, an argmax, and the numbers to act on.

    ``confidence`` uses the formula TypeSafe's reference adapter uses and that Kev
    reproduces, ``(p_max - 1/K) / (1 - 1/K)``. It is **not** a measured accuracy rate
    and it is not a calibrated correctness rate; the calibrated quantity is
    ``probabilities``. For an ``noul`` the answer is P(true) and confidence is P(true).
    """

    key: str
    type: str
    prediction: str
    prediction_index: int
    confidence: float
    probabilities: dict[str, float]
    #: For ``score`` questions only: the probability-weighted mean level index, which
    #: may fall between levels. This is the useful summary of an ordinal answer.
    expected_score: float | None = None


def _mean_confidence(probs: torch.Tensor) -> float:
    """TypeSafe's choice confidence: (p_max - 1/K) / (1 - 1/K), 1.0 for a single option."""
    k = int((probs > 0).sum().item())
    if k <= 1:
        return 1.0
    p_max = float(probs.max().item())
    return max(0.0, (p_max - 1.0 / k) / (1.0 - 1.0 / k))


def _score_confidence(probs: torch.Tensor) -> float:
    """TypeSafe's score confidence: max(0, 1 - E|level - mode| / D).

    ``D`` is the mean distance of a uniform distribution over *all* declared levels
    from its middle -- 2/3 for three levels -- so all mass on one level gives 1 and a
    uniform or wider spread gives 0.

    ``K`` is the number of declared levels, **not** the number with non-zero
    probability. A level the model assigned exactly 0.0 is still part of the legend and
    still counts toward ``D``; dropping it inflates the denominator's absence and
    understates confidence. On the published ``{0: 0.00, 1: 0.56, 2: 0.44}`` example
    this distinction is the difference between 0.34 and a wrong 0.12.
    """
    k = int(probs.numel())
    if k <= 1:
        return 1.0
    levels = [(i, float(p)) for i, p in enumerate(probs.tolist())]
    mode = max(levels, key=lambda t: t[1])[0]
    total = sum(p for _, p in levels)
    if total <= 0:
        return 0.0
    expected = sum(i * p for i, p in levels) / total
    # D: mean |level - middle| of a uniform distribution over all k levels.
    middle = (k - 1) / 2.0
    d = sum(abs(i - middle) for i in range(k)) / k
    if d <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(expected - mode) / d)


class JebNanoModel(nn.Module):
    """An LFM2.5 backbone plus a decision head. One forward pass, no generation.

    Parameters
    ----------
    backbone:
        HF model id or local path. Text-only models answer text questions; the
        LFM2.5-VL checkpoints additionally accept ``media`` for image/audio/video.
    max_options:
        Slot count in the head. 256 matches the rest of the ecosystem. Drop to 20 if
        your application never exceeds it — the head is the only part that changes size.
    temperatures:
        Per-type temperatures, or ``None`` to serve raw probabilities.
    revision:
        Pin a revision for reproducibility. Strongly recommended; several projects in
        this ecosystem got burned by a silent base-model swap.
    """

    def __init__(
        self,
        backbone: str = DEFAULT_BACKBONE,
        *,
        max_options: int = 256,
        temperatures: TemperatureFit | None = None,
        revision: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        device: str | Literal["mps", "cuda", "cpu"] = "cpu",
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.backbone_name = backbone
        self.device = device
        self.temperatures = temperatures or TemperatureFit()
        self.max_options = max_options

        config = AutoConfig.from_pretrained(backbone, revision=revision,
                                            trust_remote_code=trust_remote_code)
        text_config = getattr(config, "text_config", config)
        self.hidden_size = int(getattr(text_config, "hidden_size"))

        is_vision = "vl" in str(getattr(config, "model_type", "")).lower()
        if is_vision:
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(
                backbone, revision=revision, dtype=dtype, trust_remote_code=trust_remote_code)
            from transformers import AutoProcessor
            self.tokenizer = AutoProcessor.from_pretrained(
                backbone, revision=revision, trust_remote_code=trust_remote_code).tokenizer
            self.processor = self.tokenizer  # replaced below if available
            try:
                self.processor = AutoProcessor.from_pretrained(
                    backbone, revision=revision, trust_remote_code=trust_remote_code)
            except Exception:  # pragma: no cover - processor optional
                pass
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                backbone, revision=revision, dtype=dtype, trust_remote_code=trust_remote_code)
            self.tokenizer = AutoTokenizer.from_pretrained(
                backbone, revision=revision, trust_remote_code=trust_remote_code)
            self.processor = None

        self.model = self.model.to(device).eval()
        self.model.requires_grad_(False)

        self.head = DecisionHead(self.hidden_size, max_options=max_options).to(device).eval()

        self._capture: dict[str, torch.Tensor] = {}
        self._install_hook()

    # ---------------------------------------------------------------- internals
    def _find_text_backbone(self) -> nn.Module:
        """Locate the module whose output is the text hidden state.

        Jev-Omni probes four candidate paths because the multimodal wrapper nests the
        text stack differently from the text-only model. We try the same ladder.
        """
        for path in ("model.language_model", "language_model.model", "model.text_model",
                     "model", "transformer", "base_model"):
            node: Any = self.model
            for part in path.split("."):
                node = getattr(node, part, None)
                if node is None:
                    break
            if node is not None and hasattr(node, "layers"):
                return node
        raise RuntimeError(
            f"could not locate the text backbone of {self.backbone_name!r}; "
            "expected a submodule with a `.layers` attribute"
        )

    def _install_hook(self) -> None:
        target = self._find_text_backbone()

        def hook(_m, _args, out):
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            self._capture["hidden"] = hidden[:, -1].float()

        target.register_forward_hook(hook)

    # ------------------------------------------------------------------ forward
    def _encode(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, add_special_tokens=False,
                              return_tensors="pt").input_ids

    @torch.inference_mode()
    def _forward_last_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        self._capture.clear()
        kwargs: dict[str, Any] = {"use_cache": False}
        import inspect
        if "logits_to_keep" in inspect.signature(self.model.forward).parameters:
            kwargs["logits_to_keep"] = 1
        self.model(input_ids=input_ids.to(self.device), **kwargs)
        if "hidden" not in self._capture:
            raise RuntimeError(
                "the forward hook did not fire; the backbone layout changed or the "
                "model returned no hidden state"
            )
        return self._capture["hidden"]

    @torch.inference_mode()
    def decide(
        self,
        state: str | dict[str, Any],
        question: Question,
        *,
        media: str | None = None,
        modality: str = "text",
        apply_temperature: bool = True,
    ) -> DecisionAnswer:
        """Answer one typed question about a state. One forward pass, no generation.

        ``modality`` is ``text``, ``image``, ``audio`` or ``video``; pass a local path in
        ``media`` for the last three. Requires a vision-capable backbone
        (``LiquidAI/LFM2.5-VL-450M`` or larger) — the text-only LFM2.5 checkpoints
        cannot accept media.
        """
        request = DecisionRequest(state=state, questions=[question])
        prompt = render_request(request)
        input_ids = self._build_inputs(prompt, media, modality)
        hidden = self._forward_last_hidden(input_ids)
        n = torch.tensor([question.n_options], device=self.device)
        logits = self.head(hidden, n)
        # Look up T by option-count bucket first, then question type, then global.
        t = (self.temperatures.get(question.type, question.n_options)
             if apply_temperature else 1.0)
        probs = probabilities(logits, t)[0]
        return _to_answer(question, probs)

    @torch.inference_mode()
    def decide_many(
        self,
        state: str | dict[str, Any],
        questions: list[Question],
        *,
        media: str | None = None,
        modality: str = "text",
        apply_temperature: bool = True,
    ) -> list[DecisionAnswer]:
        """Answer several questions, each in its own forward pass over the state.

        This is the *isolated* path: no question can see another's text, because the
        tokens are not in the sequence. The cost is re-encoding the state per question.

        If your questions genuinely need isolation, that is the right trade. If they do
        not, batching them into one prompt is faster — but then a question *can* read
        its siblings, and you must decide whether that is acceptable. Several published
        implementations of this class ship a switch for exactly this reason.
        """
        return [
            self.decide(state, q, media=media, modality=modality,
                        apply_temperature=apply_temperature)
            for q in questions
        ]

    def _build_inputs(self, prompt: str, media: str | None, modality: str) -> torch.Tensor:
        if modality == "text":
            return self._encode(prompt)
        if media is None:
            raise ValueError(f"modality={modality!r} requires media=<path>")
        if self.processor is None:
            raise ValueError(
                f"{self.backbone_name} has no processor; use a vision checkpoint such "
                f"as {VISION_BACKBONE} for image/audio/video input"
            )
        content: list[dict[str, Any]] = []
        if modality == "image":
            from PIL import Image
            content.append({"type": "image", "image": Image.open(media).convert("RGB")})
        elif modality == "video":
            content.extend({"type": "image", "image": f} for f in sample_video_frames(media, 16))
        elif modality == "audio":
            content.append({"type": "audio", "audio": _to_wav_16k_mono(media, seconds=30)})
        else:
            raise ValueError("modality must be text, image, audio or video")
        content.append({"type": "text", "text": prompt})
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        )
        return inputs["input_ids"]

    def save_pretrained(self, out_dir: str) -> None:
        """Save the head and the temperature fit. The backbone is not duplicated."""
        from safetensors.torch import save_file

        os.makedirs(out_dir, exist_ok=True)
        save_file({k: v.contiguous() for k, v in self.head_state_dict().items()},
                  os.path.join(out_dir, "head.safetensors"))
        import json
        with open(os.path.join(out_dir, "decision_config.json"), "w") as f:
            json.dump(
                {
                    "backbone": self.backbone_name,
                    "backbone_class": type(self.model).__name__,
                    "hidden_size": self.hidden_size,
                    "output_classes": self.max_options,
                    "dtype": "float32",
                    "temperatures": self.temperatures.to_dict(),
                },
                f, indent=2,
            )


def _to_answer(question: Question, probs: torch.Tensor) -> DecisionAnswer:
    values = [float(p) for p in probs.tolist()]
    best = int(max(range(len(values)), key=lambda i: values[i]))
    expected = None
    if question.type == "score":
        expected = sum(i * p for i, p in enumerate(values))
        conf = _score_confidence(probs)
    elif question.type == "noul":
        conf = values[best]
    else:
        conf = _mean_confidence(probs)
    return DecisionAnswer(
        key=question.key,
        type=question.type,
        prediction=question.options[best],
        prediction_index=best,
        confidence=float(conf),
        probabilities={opt: values[i] for i, opt in enumerate(question.options)},
        expected_score=expected,
    )


def _to_wav_16k_mono(path: str, seconds: int = 30) -> str:
    """Decode any audio/video to 16 kHz mono WAV, truncated to ``seconds``.

    Matches Jev-Omni exactly: ``ffmpeg -v error -i IN -t 30 -ac 1 -ar 16000 OUT``.
    16 kHz mono is also what Whisper-style encoders expect, so this keeps the input
    identical to the convention the published audio numbers were measured under.
    """
    import subprocess
    import tempfile

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-t", str(seconds),
         "-ac", "1", "-ar", "16000", tmp.name],
        check=True,
    )
    return tmp.name


def sample_video_frames(path: str, count: int = 16) -> list:
    """Uniformly sample ``count`` frames, Jev-Omni's convention.

    Frame indices are ``round((total - 1) * (k + 0.5) / count)`` — the centre of each of
    ``count`` equal bins, so a clip's duration changes which frames land where. Each
    frame is then fed as a *separate image*, which is why video is expensive here:
    16 frames means 16 image encodes.
    """
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"could not read a frame count from {path!r}")
    wanted = sorted({int(round((total - 1) * (k + 0.5) / count)) for k in range(count)})
    frames, current = [], 0
    for index in wanted:
        while current < index and cap.grab():
            current += 1
        ok, frame = cap.read()
        current += 1
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise ValueError(f"could not decode any frame from {path!r}")
    return frames
