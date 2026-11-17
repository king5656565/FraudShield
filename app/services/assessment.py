from __future__ import annotations

import json
from dataclasses import dataclass

from app.core.enums import RiskSignal, RiskLevel
from app.schemas.dtos import AiMessage
from app.services.ai import AiClient, PromptTemplates, has_consult_signal, has_high_risk_signal


@dataclass
class RiskAssessment:
    signal: RiskSignal
    signal_score: float
    risk: RiskLevel
    confidence: float
    summary: str


class RiskAssessmentService:
    def __init__(self, ai: AiClient):
        self.ai = ai

    def assess(self, text: str, history: list[AiMessage] | None = None) -> RiskAssessment:
        if has_high_risk_signal(text):
            return RiskAssessment(RiskSignal.HIGH_RISK, 4.0, RiskLevel.HIGH, 0.95, "检测到明确诈骗信号")
        try:
            raw = self.ai.complete(PromptTemplates.fraud_assessment_prompt(history or [], text))
            start = raw.find("{")
            end = raw.rfind("}")
            data = json.loads(raw[start:end + 1] if start >= 0 and end > start else raw)
            signal = RiskSignal(data.get("signal", "NORMAL").upper())
            score = float(data.get("signalScore", score_for_signal(signal)))
            risk = RiskLevel(data.get("risk", risk_from_score(score).value).upper())
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.75))))
            score_risk = risk_from_score(score)
            if risk_order(score_risk) > risk_order(risk):
                risk = score_risk
            if signal == RiskSignal.HIGH_RISK:
                risk = RiskLevel.HIGH
            return RiskAssessment(signal, score, risk, confidence, data.get("summary", "模型评估结果"))
        except Exception:
            return heuristic(text)


def heuristic(text: str) -> RiskAssessment:
    if has_high_risk_signal(text):
        return RiskAssessment(RiskSignal.HIGH_RISK, 4.0, RiskLevel.HIGH, 0.95, "检测到明确诈骗信号")
    if has_consult_signal(text):
        return RiskAssessment(RiskSignal.SUSPECTED, 2.4, RiskLevel.MEDIUM, 0.74, "检测到疑似诈骗信号")
    return RiskAssessment(RiskSignal.NORMAL, 0.0, RiskLevel.LOW, 0.66, "未检测到明显诈骗信号")


def score_for_signal(signal: RiskSignal) -> float:
    return {
        RiskSignal.HIGH_RISK: 4.0,
        RiskSignal.HIGH_RISK_FEATURE: 3.0,
        RiskSignal.SUSPECTED: 2.0,
        RiskSignal.NORMAL: 0.0,
    }[signal]


def risk_from_score(score: float) -> RiskLevel:
    if score >= 4:
        return RiskLevel.HIGH
    if score >= 3:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def risk_order(risk: RiskLevel) -> int:
    return {RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2, RiskLevel.HIGH: 3}[risk]
