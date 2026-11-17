from __future__ import annotations

import importlib.util
import logging
import threading

from app.core.config import Settings


logger = logging.getLogger(__name__)

# 模型缓存：bge-reranker-v2-m3 是 cross-encoder，加载较重（约 2GB），
# 只加载一次，之后按 model 名复用；加载/推理失败后标记 failed，后续直接回退。
_LOCK = threading.Lock()
_LOADED_MODEL = None
_FAILED_MODEL = None


def semantic_rerank(settings: Settings, query: str, texts: list[str]) -> list[float] | None:
    """用 bge-reranker-v2-m3 对候选做语义重排，返回归一化相关分（0~1）。

    任一环节不可用（未启用 / 未安装 FlagEmbedding / 模型加载或推理失败）时返回 None，
    由调用方回退到本地词面重排，保证检索链路不中断。
    """
    if not settings.knowledge_rerank_enabled or not texts:
        return None
    model = _get_model(settings)
    if model is None:
        return None
    pairs = [[query, text if text.strip() else " "] for text in texts]
    try:
        scores = model.compute_score(pairs, normalize=True)
    except Exception as exc:  # noqa: BLE001 - 推理失败不能拖垮检索
        logger.warning("bge-reranker compute_score failed; fallback to lexical rerank: %s", exc)
        _mark_failed(settings.knowledge_reranker_model)
        return None
    if isinstance(scores, float):
        scores = [scores]
    return [float(score) for score in scores]


def semantic_reranker_name(settings: Settings) -> str:
    """轻量返回当前语义重排器状态（不触发模型加载），供 status 展示。"""
    if not settings.knowledge_rerank_enabled:
        return "disabled"
    if importlib.util.find_spec("FlagEmbedding") is None:
        return "lexical hybrid_score (FlagEmbedding not installed)"
    return f"{settings.knowledge_reranker_model} on {settings.knowledge_reranker_device}"


def _get_model(settings: Settings):
    global _LOADED_MODEL, _FAILED_MODEL
    name = settings.knowledge_reranker_model
    if _LOADED_MODEL is not None:
        return _LOADED_MODEL
    if _FAILED_MODEL == name:
        return None
    with _LOCK:
        if _LOADED_MODEL is not None:
            return _LOADED_MODEL
        if _FAILED_MODEL == name:
            return None
        try:
            from FlagEmbedding import FlagReranker

            _LOADED_MODEL = FlagReranker(
                name,
                use_fp16=False,
                device=settings.knowledge_reranker_device,
            )
            logger.info("loaded bge reranker %s on %s", name, settings.knowledge_reranker_device)
            return _LOADED_MODEL
        except Exception as exc:  # noqa: BLE001 - 缺依赖/缺模型都回退词面重排
            _FAILED_MODEL = name
            logger.warning("bge reranker %s unavailable (%s); fallback to lexical rerank", name, exc)
            return None


def _mark_failed(name: str) -> None:
    global _LOADED_MODEL, _FAILED_MODEL
    _LOADED_MODEL = None
    _FAILED_MODEL = name
