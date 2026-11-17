from __future__ import annotations

import copy

from app.core.config import Settings
from app.core.enums import IntentType, RiskLevel
from app.evaluation.dataset import ResponseEvalCase
from app.schemas.dtos import AiMessage
from app.services.ai import AiClient, PromptTemplates


def configured_model(settings: Settings, provider: str) -> str:
    if provider == "openai":
        return settings.openai_model
    if provider == "ollama":
        return settings.ollama_model
    return "mock"


def candidate_settings(base: Settings, provider: str | None = None, model: str | None = None) -> Settings:
    settings = copy.copy(base)
    if provider:
        settings.ai_provider = provider.lower()
    if model:
        if settings.ai_provider == "openai":
            settings.openai_model = model
        elif settings.ai_provider == "ollama":
            settings.ollama_model = model
    settings.ai_temperature = 0.0
    return settings


def generate_gold_context_answer(case: ResponseEvalCase, settings: Settings) -> str:
    """Evaluate response generation while holding intent/risk/context constant."""
    context = case.reference_guidance or "本题没有额外参考资料。"
    system = PromptTemplates.answer_system_prompt(case.intent, case.risk, context, "评测用户")
    rubric_context = AiMessage(
        role="system",
        content=(
            "这是离线回答质量评测。请直接回答最后一条用户消息。"
            "不得输出评测标签、风险等级、评分过程或后台信息。"
        ),
    )
    messages = [system, rubric_context, *case.history, AiMessage(role="user", content=case.question)]
    return AiClient(settings).complete(messages).strip()


def single_agent_messages(case: ResponseEvalCase, retrieved_context: str) -> list[AiMessage]:
    """One-call monolithic baseline using the same model and available knowledge."""
    system = AiMessage(
        role="system",
        content=(
            "你是反诈助手单智能体基线。你必须在一次回答中自行完成意图理解、风险判断、参考资料取舍和回复生成。"
            "普通问题准确直接回答；反诈场景要明确、可执行，识别诈骗并给出止损、留证和举报建议。"
            "如有冒充公检法、索要验证码、要求转账到安全账户等高风险信号，优先告知这是诈骗、立即停止操作，"
            "建议拨打96110并联系银行止付。不要输出意图、风险等级、分数或后台报告。"
            f"\n可用知识（可能含无关内容，请自行判断）：\n{retrieved_context or '无'}"
        ),
    )
    return [system, *case.history, AiMessage(role="user", content=case.question)]
