from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankerSelection:
    enabled: bool
    mode: str
    model_name: str | None
    device: str | None
    reason: str


class CrossEncoderReranker:
    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        max_length: int = 1024,
        batch_size: int = 16,
    ):
        from sentence_transformers import CrossEncoder
        from transformers import AutoConfig

        self.model_name = str(model_name)
        self.device = str(device)
        self.requested_max_length = int(max_length)
        self.batch_size = int(batch_size)

        model_config = AutoConfig.from_pretrained(self.model_name)
        model_max_positions = getattr(model_config, "max_position_embeddings", None)

        if model_max_positions:
            self.max_length = min(self.requested_max_length, int(model_max_positions))
        else:
            self.max_length = self.requested_max_length

        logger.info(
            "Loading CrossEncoder reranker model=%s device=%s requested_max_length=%s "
            "effective_max_length=%s model_max_positions=%s batch_size=%s",
            self.model_name,
            self.device,
            self.requested_max_length,
            self.max_length,
            model_max_positions,
            self.batch_size,
        )

        self.model = CrossEncoder(
            self.model_name,
            max_length=self.max_length,
            device=self.device,
        )

    def predict(self, pairs: list[list[str]]) -> list[float]:
        if not pairs:
            return []

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
        )

        return scores.tolist() if hasattr(scores, "tolist") else list(scores)


def select_reranker(
    *,
    use_reranker: bool,
    mode: str,
    cpu_model: str,
    gpu_model: str,
) -> RerankerSelection:
    normalized_mode = str(mode or "auto").strip().lower()

    if not use_reranker:
        return RerankerSelection(
            enabled=False,
            mode=normalized_mode,
            model_name=None,
            device=None,
            reason="RAG_USE_RERANKER is false",
        )

    if normalized_mode == "off":
        return RerankerSelection(
            enabled=False,
            mode=normalized_mode,
            model_name=None,
            device=None,
            reason="RAG_RERANKER_MODE=off",
        )

    if normalized_mode not in {"auto", "cpu_light", "gpu_heavy"}:
        raise ValueError(
            "Invalid RAG_RERANKER_MODE. Expected one of: off, cpu_light, gpu_heavy, auto. "
            f"Got: {mode!r}"
        )

    import torch

    cuda_available = bool(torch.cuda.is_available())

    if normalized_mode == "cpu_light":
        return RerankerSelection(
            enabled=True,
            mode=normalized_mode,
            model_name=cpu_model,
            device="cpu",
            reason="CPU-light reranker forced by config",
        )

    if normalized_mode == "gpu_heavy":
        if not cuda_available:
            raise RuntimeError(
                "RAG_RERANKER_MODE=gpu_heavy was requested, but CUDA is not available. "
                "Use RAG_RERANKER_MODE=auto or cpu_light on CPU-only machines."
            )

        return RerankerSelection(
            enabled=True,
            mode=normalized_mode,
            model_name=gpu_model,
            device="cuda",
            reason="GPU-heavy reranker forced by config",
        )

    if cuda_available:
        return RerankerSelection(
            enabled=True,
            mode=normalized_mode,
            model_name=gpu_model,
            device="cuda",
            reason="auto selected GPU-heavy reranker because CUDA is available",
        )

    return RerankerSelection(
        enabled=True,
        mode=normalized_mode,
        model_name=cpu_model,
        device="cpu",
        reason="auto selected CPU-light reranker because CUDA is not available",
    )