from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from app.core.enums import IntentType, RiskLevel
from app.models.entities import RiskReport, UserAccount


class SkillLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class SkillValidationIssue:
    level: str
    message: str


@dataclass(frozen=True)
class FraudShieldSkill:
    name: str
    description: str
    body: str
    path: Path
    metadata: dict[str, str] = field(default_factory=dict)

    def prompt_context(self) -> str:
        return f"应用 skill: {self.name}\n{self.body.strip()}"

    def validation_issues(self) -> list[SkillValidationIssue]:
        issues: list[SkillValidationIssue] = []
        if self.path.parent.name != self.name:
            issues.append(SkillValidationIssue("WARN", f"目录名 {self.path.parent.name} 与 skill name {self.name} 不一致"))
        if "## Workflow" not in self.body:
            issues.append(SkillValidationIssue("WARN", "建议包含 ## Workflow 小节，便于人工审阅和模型稳定加载"))
        if len(self.description) < 20:
            issues.append(SkillValidationIssue("WARN", "description 太短，可能无法准确表达触发场景"))
        if self.name == "fraud_case_handoff_summary" and "```text" not in self.body:
            issues.append(SkillValidationIssue("ERROR", "fraud_case_handoff_summary 必须包含 text 模板"))
        return issues


class FraudShieldSkillRegistry:
    def __init__(self, root: Path | None = None):
        self.root = root or Path(__file__).resolve().parents[2] / "skills"

    def list_skills(self) -> list[FraudShieldSkill]:
        if not self.root.exists():
            return []
        skills = []
        for skill_file in sorted(self.root.glob("*/SKILL.md")):
            skills.append(self._load_skill_file(skill_file))
        return skills

    def status_items(self) -> list[dict]:
        if not self.root.exists():
            return []
        items = []
        for skill_file in sorted(self.root.glob("*/SKILL.md")):
            try:
                skill = self._load_skill_file(skill_file)
                issues = skill.validation_issues()
            except SkillLoadError as exc:
                items.append(
                    {
                        "name": skill_file.parent.name,
                        "status": "FAILED",
                        "description": str(exc),
                        "path": str(skill_file.relative_to(self.root.parent)),
                        "issues": [{"level": "ERROR", "message": str(exc)}],
                    }
                )
                continue
            has_error = any(issue.level == "ERROR" for issue in issues)
            items.append(
                {
                    "name": skill.name,
                    "status": "FAILED" if has_error else "READY" if not issues else "WARN",
                    "description": skill.description,
                    "path": str(skill.path.relative_to(self.root.parent)),
                    "issues": [{"level": issue.level, "message": issue.message} for issue in issues],
                    "metadata": skill.metadata,
                }
            )
        return items

    def get_required(self, name: str) -> FraudShieldSkill:
        for skill in self.list_skills():
            if skill.name == name:
                return skill
        raise SkillLoadError(f"required standard skill not found: {name}")

    def template_for(self, name: str) -> str:
        skill = self.get_required(name)
        match = re.search(r"```text\s*\n(?P<template>.*?)\n```", skill.body, re.DOTALL)
        if match is None:
            raise SkillLoadError(f"standard skill {name} does not define a text template")
        return match.group("template").strip()

    def _load_skill_file(self, path: Path) -> FraudShieldSkill:
        text = path.read_text(encoding="utf-8")
        metadata, body = _split_frontmatter(text, path)
        name = metadata.get("name") or path.parent.name
        description = metadata.get("description", "")
        if not name.strip():
            raise SkillLoadError(f"{path} is missing frontmatter name")
        if not description.strip():
            raise SkillLoadError(f"{path} is missing frontmatter description")
        if not body.strip():
            raise SkillLoadError(f"{path} is missing skill body")
        return FraudShieldSkill(name=name.strip(), description=description.strip(), body=body.strip(), path=path, metadata=metadata)


class FraudShieldSkillLibrary:
    @staticmethod
    def registry() -> FraudShieldSkillRegistry:
        return FraudShieldSkillRegistry()

    @staticmethod
    def list_skills() -> list[FraudShieldSkill]:
        return FraudShieldSkillLibrary.registry().list_skills()

    @staticmethod
    def status_items() -> list[dict]:
        return FraudShieldSkillLibrary.registry().status_items()

    @staticmethod
    def response_skill_context(intent: IntentType, risk: RiskLevel, text: str) -> str:
        names = FraudShieldSkillLibrary.response_skill_names(intent, risk, text)
        registry = FraudShieldSkillLibrary.registry()
        return "\n\n".join(registry.get_required(name).prompt_context() for name in names)

    @staticmethod
    def response_skill_names(intent: IntentType, risk: RiskLevel, text: str) -> list[str]:
        if intent == IntentType.CHAT:
            return []

        if risk == RiskLevel.HIGH:
            return ["fraud_response_baseline", "fraud_high_risk_interception"]

        lowered = text.lower()
        names = ["fraud_response_baseline", "fraud_report_guidance"]
        if _contains_any(lowered, ["公检法", "公安", "法院", "检察院", "客服", "理赔", "领导", "冒充", "验证码", "安全账户"]):
            names.append("impersonation_fraud_disposal")
        if _contains_any(lowered, ["投资", "理财", "杀猪盘", "网恋", "高收益", "稳赚", "股票", "虚拟币", "恋爱"]):
            names.append("investment_romance_disposal")
        if _contains_any(lowered, ["贷款", "征信", "刷单", "返利", "返现", "兼职", "提额", "保证金", "解冻费"]):
            names.append("loan_brushing_disposal")
        return _dedupe(names)

    @staticmethod
    def fraud_high_risk_interception_prompt() -> str:
        return FraudShieldSkillLibrary.registry().get_required("fraud_high_risk_interception").prompt_context()

    @staticmethod
    def fraud_case_handoff_summary(report: RiskReport, user: UserAccount | None) -> str:
        template = FraudShieldSkillLibrary.registry().template_for("fraud_case_handoff_summary")
        user_label = _user_label(user, report.user_id)
        urgency = "立即跟进" if report.risk_level == RiskLevel.HIGH.value else "尽快跟进"
        next_steps = [
            f"{urgency}，确认用户是否已转账、是否泄露验证码，以及是否仍在与诈骗分子联系。",
            "联系用户本人，引导其立即停止转账并申请银行止付。",
            "指导用户保留聊天记录与转账凭证，并拨打 96110 或前往公安机关报案。",
            "将后续安排、接手人和下一次回访时间写入个案备注。",
        ]
        return _render_template(
            template,
            {
                "report_id": str(report.id),
                "user": user_label,
                "risk_level": report.risk_level,
                "signal": report.signal,
                "confidence": f"{report.confidence:.2f}",
                "summary": report.summary,
                "next_steps": "\n".join(f"- {step}" for step in next_steps),
                "content_excerpt": _truncate(report.content, 700),
            },
        )


def _split_frontmatter(text: str, path: Path) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        raise SkillLoadError(f"{path} is missing YAML frontmatter")
    end = text.find("\n---", 4)
    if end == -1:
        raise SkillLoadError(f"{path} has unterminated YAML frontmatter")
    metadata = {}
    for line in text[4:end].splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise SkillLoadError(f"{path} has invalid frontmatter line: {line}")
        key, value = stripped.split(":", 1)
        metadata[key.strip()] = value.strip().strip("\"'")
    return metadata, text[end + len("\n---") :].strip()


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _render_template(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    return rendered


def _user_label(user: UserAccount | None, user_id: int) -> str:
    if user is None:
        return f"userId={user_id}"
    if user.display_name:
        return f"{user.display_name} ({user.username})"
    return user.username


def _truncate(text: str, limit: int) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit - 3]}..."
