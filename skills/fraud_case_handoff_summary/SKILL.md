---
name: fraud_case_handoff_summary
description: Use when creating a staff-facing handoff summary for anti-fraud staff or administrators after a high-risk fraud report is generated.
---

# Fraud Case Handoff Summary

## Workflow

- Write for anti-fraud staff or administrators, not for the user.
- Preserve the user's original meaning while avoiding unnecessary sensitive detail.
- Include the report identity, user identity, risk level, signal label, confidence, model summary, follow-up actions, and a bounded excerpt of the user's expression.
- Make the first follow-up action about whether money was transferred, whether verification codes were shared, and whether the user is still in contact with the scammer.
- Keep the handoff factual and actionable; do not add unsupported conclusions.

## Output Template

```text
应用 skill: fraud_case_handoff_summary
报告ID：{{report_id}}
用户：{{user}}
风险等级：{{risk_level}}
风险信号：{{signal}}
置信度：{{confidence}}
模型摘要：{{summary}}
建议跟进：
{{next_steps}}
用户原始表达：
{{content_excerpt}}
```
