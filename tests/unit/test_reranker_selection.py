from __future__ import annotations

import pytest

from app.services.retrieval.reranker import select_reranker


def test_select_reranker_disabled_by_flag():
    selection = select_reranker(
        use_reranker=False,
        mode="auto",
        cpu_model="cpu-model",
        gpu_model="gpu-model",
    )

    assert selection.enabled is False
    assert selection.model_name is None
    assert selection.device is None


def test_select_reranker_disabled_by_mode_off():
    selection = select_reranker(
        use_reranker=True,
        mode="off",
        cpu_model="cpu-model",
        gpu_model="gpu-model",
    )

    assert selection.enabled is False
    assert selection.reason == "RAG_RERANKER_MODE=off"


def test_select_reranker_cpu_light_forces_cpu_model():
    selection = select_reranker(
        use_reranker=True,
        mode="cpu_light",
        cpu_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        gpu_model="BAAI/bge-reranker-v2-m3",
    )

    assert selection.enabled is True
    assert selection.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert selection.device == "cpu"


def test_select_reranker_rejects_invalid_mode():
    with pytest.raises(ValueError):
        select_reranker(
            use_reranker=True,
            mode="bad_mode",
            cpu_model="cpu-model",
            gpu_model="gpu-model",
        )