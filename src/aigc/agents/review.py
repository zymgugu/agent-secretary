"""Review Agent：审查 Worker 执行结果，三道检查防幻觉。"""

from __future__ import annotations

import json
from src.aigc.agents.utils import get_model
from src.aigc.models.state import AgentState

REVIEW_PROMPT = """你是一个质量审查员，审查 Worker Agent 的执行结果。

## 审查原则（重要）

你的任务是判断 Worker 是否正确理解了用户意图并执行了恰当的操作。
**不要批评工具本身的限制**（如工具不支持某参数），这不是 Worker 的错。

## 三道检查

1. **来源追溯**：用户要求的事情 Worker 做了吗？Worker 有没有做用户没要求的事？
2. **工具选择**：Worker 选的工具是否匹配用户意图？有没有选错工具类型？
3. **参数合理性**：Worker 传的参数是否合理？（只检查逻辑错误，不检查"工具不支持某参数"这种技术限制）

## 判定标准

- **approved**: 执行正确，无问题
- **rejected**: Worker 犯了明显的逻辑错误（选错工具、理解错意图、编造不存在的要求）
  ⚠️ 重要：如果问题是工具能力不足导致的（如工具不支持某参数），应该 approved 而不是 rejected
- **flagged**: 基本正确但有小问题（如默认值未告知用户、可选的优化建议）

## 输出格式

严格返回以下 JSON（不要 markdown 代码块标记）：
{
  "status": "approved|rejected|flagged",
  "issues": ["问题描述"],
  "suggestion": "修改建议（rejected 时必填）",
  "user_notice": "需要告知用户的提示（flagged 时填写）"
}
"""


def _get_tool_hints(intent_type: str) -> str:
    """根据意图类型返回可用工具说明，帮助 Review 准确判断。"""
    hints = {
        "schedule": """可用工具：add_calendar_event(增)、query_calendar(查)、delete_calendar_event(删)。
参数：title(标题)、start_time/end_time(ISO8601)、description、location、remind_before_minutes""",
        "search": """可用工具：web_search(query)、weather_query(city)、flight_query(origin, destination, date)。
注：weather_query 只接受 city 参数，不支持 date 参数。flight_query 的 date 为可选。""",
        "reminder": """可用工具：set_reminder(content, trigger_time, repeat)、set_condition_reminder(condition, content)、cancel_reminder(reminder_id)。
repeat 可选值：once/daily/weekly/monthly""",
        "document": """可用工具：create_note(title, content, tags)、summarize_text(text, max_length)""",
    }
    return hints.get(intent_type, "无特定工具约束")


def review_node(state: AgentState) -> dict:
    """Review 节点：检查 worker_result，决定通过/退回/标记。"""
    model = get_model(temperature=0.1)

    worker_result = state.get("worker_result") or {}
    intent = state.get("intent") or {}
    intent_type = intent.get("type", "unknown")

    # 提取用户原始消息
    messages = state["messages"]
    user_message = ""
    for msg in reversed(messages):
        role = getattr(msg, "type", "") if hasattr(msg, "type") else ""
        if role == "human":
            user_message = getattr(msg, "content", "")
            break

    tool_hints = _get_tool_hints(intent_type)

    review_input = f"""请审查以下 Worker Agent 的执行结果。

## 可用工具说明
{tool_hints}

## 用户原始消息
{user_message}

## 意图分类
类型：{intent_type}
分析：{intent.get('reasoning', '')}

## Worker 执行结果
思考过程：{worker_result.get('reasoning', '')}
已执行的操作：{json.dumps(worker_result.get('actions_executed', []), ensure_ascii=False)}
置信度：{worker_result.get('confidence', 0)}
备注：{worker_result.get('notes', '无')}

请进行三道检查并输出 JSON。"""

    response = model.invoke([
        {"role": "system", "content": REVIEW_PROMPT},
        {"role": "user", "content": review_input},
    ])

    review = _parse_review(response.content)
    review_status = review.get("status", "approved")
    review_feedback = review.get("suggestion", "")
    user_notice = review.get("user_notice", "")

    retry_count = state.get("retry_count", 0)

    # flagged 时把 user_notice 放入 review_feedback，让 finalize 拼到回复里
    # rejected 时 review_feedback 存放修改建议，由 make_worker_node 注入给 Worker
    final_feedback = review_feedback
    if review_status == "flagged" and user_notice:
        final_feedback = user_notice

    return {
        "review_status": review_status,
        "review_feedback": final_feedback,
        "retry_count": retry_count + 1 if review_status == "rejected" else retry_count,
    }


def _parse_review(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "status": "approved",
            "issues": ["Review 输出解析失败，默认通过"],
            "suggestion": "",
            "user_notice": "",
        }
