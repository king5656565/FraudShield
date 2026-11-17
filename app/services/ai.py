from __future__ import annotations

import json
from typing import Iterable

import httpx

from app.core.config import Settings
from app.core.enums import IntentType, RiskLevel
from app.schemas.dtos import AiMessage


class PromptTemplates:
    @staticmethod
    def intent_prompt(history: list[AiMessage], user_input: str) -> list[AiMessage]:
        return [
            AiMessage(role="system", content=(
                "你是一个反诈意图分类器，只做意图识别，不回答问题。"
                "只输出 CHAT、CONSULT、RISK 之一。"
                "CHAT 包含普通信息、日常对话、正常咨询；"
                "CONSULT 包含疑似诈骗信号（索要个人信息、诱导点击链接或加好友、异常返利返现、可疑投资理财、冒充客服理赔、贷款或征信异常等）；"
                "RISK 包含明确诈骗或高风险（冒充公检法、要求转账到安全账户、索要验证码、紧急催促威胁、涉嫌洗钱恐吓等）。"
            )),
            AiMessage(role="user", content=f"最近上下文：\n{format_history(history)}\n\n当前输入：\n{user_input}"),
        ]

    @staticmethod
    def fraud_assessment_prompt(history: list[AiMessage], user_input: str) -> list[AiMessage]:
        return [
            AiMessage(role="system", content=(
                "你负责分析疑似诈骗信息。只返回严格 JSON："
                '{"signal":"NORMAL|SUSPECTED|HIGH_RISK_FEATURE|HIGH_RISK","signalScore":0.0,'
                '"risk":"LOW|MEDIUM|HIGH","confidence":0.0,"summary":"short reason"}'
                "其中 signal 表示风险信号类型：NORMAL 正常信息、SUSPECTED 疑似诈骗、HIGH_RISK_FEATURE 高风险诈骗特征、HIGH_RISK 明确诈骗。"
            )),
            AiMessage(role="user", content=f"最近上下文：\n{format_history(history)}\n\n当前输入：\n{user_input}"),
        ]

    @staticmethod
    def answer_system_prompt(intent: IntentType, risk: RiskLevel, context: str, display_name: str, skill_context: str = "") -> AiMessage:
        if intent == IntentType.CHAT:
            content = (
                "你是反诈风险识别助手。"
                "普通信息、日常咨询请自然、准确、直接地回答。"
                "不要主动做诈骗判定，不要输出风险等级、诈骗标签或报告口吻。"
                f"用户显示名：{display_name}"
            )
            return AiMessage(role="system", content=content)
        crisis_rule = ""
        if risk == RiskLevel.HIGH:
            crisis_rule = (
                "\n高风险处理规则：先明确告知「这是诈骗，立即停止一切操作」，"
                "再给出拦截与处置建议；不向用户输出后台风险等级或报告分数。"
            )
        content = (
            "你是反诈风险识别与拦截助手。"
            "回答要明确、直接、可执行，帮助用户识别诈骗、停止转账、保留证据并举报。"
            "不要向用户输出风险等级、报告分数或后台标签。"
            "优先基于检索知识回答；知识不足时明确说明并给出通用反诈建议。"
            f"\n用户显示名：{display_name}\n检索知识：\n{context}\n\n可用 skill 指引：\n{skill_context or '无'}{crisis_rule}"
        )
        return AiMessage(role="system", content=content)


class AiClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def complete(self, messages: list[AiMessage]) -> str:
        provider = self.settings.ai_provider.lower()
        if provider == "ollama":
            return self._ollama(messages, stream=False)
        if provider == "openai":
            return self._openai(messages, stream=False)
        return self._mock(messages)

    async def stream(self, messages: list[AiMessage]):
        provider = self.settings.ai_provider.lower()
        if provider == "ollama":
            async for token in self._ollama_stream(messages):
                yield token
            return
        if provider == "openai":
            async for token in self._openai_stream(messages):
                yield token
            return
        text = self._mock(messages)
        for chunk in split_text(text, 12):
            yield chunk

    def _ollama(self, messages: list[AiMessage], stream: bool) -> str:
        payload = {
            "model": self.settings.ollama_model,
            "messages": [m.model_dump() for m in messages],
            "stream": stream,
            "options": {"temperature": self.settings.ai_temperature, "num_predict": self.settings.ai_max_tokens},
        }
        response = httpx.post(f"{self.settings.ollama_base_url}/api/chat", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["message"]["content"]

    async def _ollama_stream(self, messages: list[AiMessage]):
        payload = {
            "model": self.settings.ollama_model,
            "messages": [m.model_dump() for m in messages],
            "stream": True,
            "options": {"temperature": self.settings.ai_temperature, "num_predict": self.settings.ai_max_tokens},
        }
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", f"{self.settings.ollama_base_url}/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token

    def _openai(self, messages: list[AiMessage], stream: bool) -> str:
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        payload = {
            "model": self.settings.openai_model,
            "messages": [m.model_dump() for m in messages],
            "temperature": self.settings.ai_temperature,
            "max_tokens": self.settings.ai_max_tokens,
            "stream": stream,
        }
        response = httpx.post(f"{self.settings.openai_base_url}/chat/completions", headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def _openai_stream(self, messages: list[AiMessage]):
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        payload = {
            "model": self.settings.openai_model,
            "messages": [m.model_dump() for m in messages],
            "temperature": self.settings.ai_temperature,
            "max_tokens": self.settings.ai_max_tokens,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", f"{self.settings.openai_base_url}/chat/completions", headers=headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line.removeprefix("data: ").strip()
                    if raw == "[DONE]":
                        break
                    data = json.loads(raw)
                    token = data["choices"][0].get("delta", {}).get("content", "")
                    if token:
                        yield token

    def _mock(self, messages: list[AiMessage]) -> str:
        last = next((m.content for m in reversed(messages) if m.role == "user"), "")
        system = " ".join(m.content for m in messages if m.role == "system")
        if "严格 JSON" in system:
            if has_high_risk_signal(last):
                return '{"signal":"HIGH_RISK","signalScore":4.0,"risk":"HIGH","confidence":0.95,"summary":"检测到明确诈骗或高风险转账信号"}'
            if has_consult_signal(last):
                return '{"signal":"SUSPECTED","signalScore":2.5,"risk":"MEDIUM","confidence":0.72,"summary":"检测到疑似诈骗信号"}'
            return '{"signal":"NORMAL","signalScore":0.0,"risk":"LOW","confidence":0.66,"summary":"未检测到明显诈骗信号"}'
        if "意图分类器" in system:
            if has_high_risk_signal(last):
                return "RISK"
            if has_consult_signal(last):
                return "CONSULT"
            return "CHAT"
        if "fraud_high_risk_interception" in system and has_high_risk_signal(last):
            return "这是典型的电信网络诈骗。请立即停止一切操作：不要再转账、不要透露验证码、不要按对方要求下载 APP 或开启屏幕共享。马上拨打 96110 反诈专线说明情况，并通过「国家反诈中心」APP 举报该号码；如已转账，请立即联系银行申请冻结并报警。"
        if "当前由 ResponseAgent 以 support mode" in system:
            return "这条信息存在诈骗风险，请提高警惕：不要点击任何链接、不要转账或提供验证码，也不要透露银行卡号和身份证信息。建议保留聊天记录作为证据，并拨打 96110 或通过「国家反诈中心」APP 核实举报。"
        if "当前由 ResponseAgent 以 normal_chat mode" in system:
            return "我在。这个问题可以直接回答，请把具体情况发给我。"
        if "ContextAgent" in system and "SUFFICIENT" in system:
            return "SUFFICIENT"
        if "ContextAgent" in system:
            return last[:40] or "反诈知识库"
        return "我在。请把疑似诈骗的信息（短信、聊天记录或来电内容）发给我，我会帮你识别风险并给出处置建议。"


def format_history(history: list[AiMessage]) -> str:
    if not history:
        return "无"
    return "\n".join(f"{m.role}: {m.content}" for m in history[-20:])


HIGH_RISK_WORDS = ["诈骗", "洗钱", "安全账户", "冻结账户", "资金冻结", "涉嫌犯罪", "通缉令", "逮捕令", "索要验证码", "立即转账", "马上转账", "转入安全", "保证金", "解冻费", "不要挂电话", "冒充公检法", "冒充公安", "冒充法院", "杀猪盘", "刷单返利"]
CONSULT_WORDS = ["返利", "返现", "高收益", "投资理财", "稳赚", "兼职刷单", "刷单", "客服理赔", "退款", "中奖", "激活费", "加微信", "加QQ", "点击链接", "下载APP", "贷款", "征信", "提额", "免费领取"]


def has_high_risk_signal(text: str) -> bool:
    normalized = text.lower()
    return any(word in normalized for word in HIGH_RISK_WORDS)


def has_consult_signal(text: str) -> bool:
    normalized = text.lower()
    return any(word in normalized for word in CONSULT_WORDS)


def split_text(text: str, size: int) -> Iterable[str]:
    for index in range(0, len(text), size):
        yield text[index:index + size]
