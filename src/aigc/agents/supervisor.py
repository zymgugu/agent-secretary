"""Supervisor Agent：意图分类 + 任务拆解 + 路由决策。

核心升级：
- 支持将复杂请求拆解为多个子任务（sub_intents）
- 注入用户画像，让 LLM 基于用户上下文做判断
- 向后兼容：单一意图退化为单元素列表
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from src.aigc.agents.utils import get_model
from src.aigc.models.state import AgentState
from src.aigc.user_profile import load_profile


def _build_supervisor_prompt() -> str:
    today = datetime.now(timezone.utc).strftime("%Y年%m月%d日 (%A)")
    profile = load_profile()

    return (
        "你是一个智能秘书系统的调度中心。你的任务是：\n"
        "1. 分析用户消息的意图\n"
        "2. 判断是单一任务还是复合任务（涉及多个领域）\n"
        "3. 如果是复合任务，拆解为多个子任务\n"
        "4. 为每个子任务指定类型、参数和详细说明\n"
        "\n"
        f"当前日期：{today}\n"
        "\n"
        + profile +
        "\n"
        "## 任务拆解原则\n"
        "\n"
        "**单一任务**：用户只涉及一个领域（只安排日程、只查信息、只写文档），不需要拆解。\n"
        "**复合任务**：用户请求涉及多个领域，需要多个 Agent 协作。例如：\n"
        '- "下周三去北京做工作报告" → schedule(出差日程) + search(查航班) + document(报告模板)\n'
        '- "帮我安排下周的周会并做会议纪要模板" → schedule(创建周会) + document(会议纪要模板)\n'
        '- "运动会动员演讲" → schedule(创建演讲日程) + document(生成演讲稿)\n'
        '- "查天气然后决定明天穿什么" → search(查天气) + chat(穿搭建议)\n'
        "\n"
        "拆解时注意：\n"
        "- 每个子任务必须有独立的、明确的目标\n"
        "- 子任务之间可以有依赖关系（前一个结果影响后一个），按逻辑顺序排列\n"
        "- 给每个子任务写清晰的 reasoning，让对应的 Worker 知道要做什么\n"
        "- 不要让两个子任务做同一件事\n"
        "\n"
        "## 补充信息 / 追问回复（重要）\n"
        "\n"
        "当用户消息是**对上一轮追问的回答**时（如补充时长、地点、偏好），不要创建新的独立任务。应该：\n"
        "1. 将用户补充的信息合并到原有的任务参数中\n"
        "2. 如果信息足够 → 直接执行（confidence 高，needs_clarification=false）\n"
        "3. 如果还有关键缺失 → 继续追问，但不是重头开始\n"
        "\n"
        "示例：\n"
        '- 上一轮秘书问了"出差几天？要订机票吗？"，用户回复"三天，返航上午，不用模板"\n'
        "  → schedule（创建/更新三天出差日程，含返航偏好）+ search（查返航航班，上午出发）\n"
        "  → 不要 needs_clarification，信息已经够执行了\n"
        "\n"
        "## 意图类型\n"
        "- schedule: 日程管理（安排会议、创建日历事件、查询日程、出差规划）\n"
        "- search: 信息检索（搜索资料、查天气、查航班、查新闻）\n"
        "- reminder: 提醒设置（定时提醒、条件提醒）\n"
        "- document: 文档处理（写讲稿、做纪要、写备忘、准备模板、总结归纳）\n"
        "- chat: 闲聊或通用问答\n"
        "\n"
        "## 输出格式\n"
        "\n"
        "严格返回以下 JSON（不要 markdown 代码块标记）：\n"
        '{"is_complex": false, "sub_intents": [{"type": "...", "params": {...}, "confidence": 0.0, "needs_clarification": false, "reasoning": "..."}], "overall_reasoning": "..."}\n'
        "\n"
        "字段说明：\n"
        "- is_complex: 是否需要多个 Agent 协作\n"
        "- sub_intents: 子任务列表（单一任务只有 1 个元素）\n"
        "- sub_intents[].type: schedule|search|reminder|document|chat\n"
        "- sub_intents[].params: 提取的参数（schedule提取title/time/participants，search提取query/type，等等）\n"
        "- sub_intents[].confidence: 置信度 0-1\n"
        "- sub_intents[].needs_clarification: 该子任务是否需要追问（任何子任务需要追问则整体追问）\n"
        "- sub_intents[].reasoning: 给 Worker 的详细说明，说清楚要做什么、注意什么\n"
        "- overall_reasoning: 整体拆解思路\n"
        "\n"
        "## 置信度\n"
        "- 1.0: 意图非常明确，参数完整（包括用户补充信息后参数已齐备的直接执行）\n"
        "- 0.7-0.9: 意图明确但部分参数缺失\n"
        "- 0.5-0.7: 意图模糊\n"
        "- <0.5: 无法判断，该子任务 needs_clarification=true\n"
    )


def supervisor_node(state: AgentState) -> dict:
    """Supervisor 节点：分析用户消息，拆解为子任务列表。

    上下文策略：
    - 上一轮有 pending_intents（追问未完成）→ 合并用户新信息，继续原计划
    - 上一轮秘书在追问（含 "？"） → 用户大概率在回答，必须带上下文
    - 用户消息本身完整（如"帮我安排..."） → 新话题，不带上下文省 token
    """
    model = get_model(temperature=0.1)

    messages = state["messages"]
    pending_intents = state.get("pending_intents") or []

    # 找到最后一条用户消息和它前面的 AI 消息
    user_message = ""
    prev_ai_content = ""
    found_user = False
    for msg in reversed(messages):
        role = getattr(msg, "type", "") if hasattr(msg, "type") else ""
        content = getattr(msg, "content", "") or ""
        if role == "human" and not found_user:
            user_message = content
            found_user = True
        elif role == "ai" and found_user and not str(content).startswith("[Review"):
            prev_ai_content = content
            break

    # 判断是否需要带上下文
    need_context = bool(pending_intents) or _has_question(prev_ai_content)

    if need_context:
        context_parts = []
        for msg in messages[-8:]:
            role = getattr(msg, "type", "") if hasattr(msg, "type") else ""
            content = getattr(msg, "content", "") or ""
            if not content:
                continue
            if role == "human":
                context_parts.append(f"用户：{content}")
            elif role == "ai" and not str(content).startswith("[Review"):
                short = content[:400] + "..." if len(content) > 400 else content
                context_parts.append(f"秘书：{short}")
        context_text = "\n".join(context_parts) if context_parts else "无历史对话"

        # 注入上一轮暂存的待确认子任务，让 LLM 合并而非重建
        pending_block = ""
        if pending_intents:
            pending_desc = []
            for i, p in enumerate(pending_intents):
                pending_desc.append(
                    f"  {i+1}. type={p.get('type')}, reasoning={p.get('reasoning', '')}, "
                    f"params={p.get('params', {})}, confidence={p.get('confidence')}"
                )
            pending_block = (
                "\n## ⚠️ 上一轮已规划但等待确认的子任务（必须合并，不要重建）\n"
                + "\n".join(pending_desc) +
                "\n\n用户当前消息是对这些子任务的补充回答。请：\n"
                "1. 将用户补充的信息合并到对应子任务的 params 中\n"
                "2. 信息足够的子任务 → needs_clarification=false，confidence 提高\n"
                "3. 仍然缺失关键信息的 → 保留 needs_clarification=true\n"
                "4. 不要创建全新的子任务列表，在原有基础上更新\n"
            )

        analysis_prompt = (
            "请分析用户最新消息的意图。\n\n"
            f"## 对话上下文\n{context_text}\n"
            f"{pending_block}\n"
            f"## 需要分析的用户消息\n{user_message}\n\n"
            "这是一条对上一轮追问的回复，请结合上下文判断意图，合并已有计划。"
        )
    else:
        analysis_prompt = f"请分析以下用户消息的意图：\n\n{user_message}"

    response = model.invoke([
        {"role": "system", "content": _build_supervisor_prompt()},
        {"role": "user", "content": analysis_prompt},
    ])

    result = _parse_supervisor_output(response.content)

    sub_intents = result.get("sub_intents", [])
    overall_reasoning = result.get("overall_reasoning", "")

    # 检查是否有子任务需要追问
    needs_clarification = any(
        sub.get("needs_clarification", False) for sub in sub_intents
    )

    final_response = None
    if needs_clarification:
        final_response = _generate_clarification(user_message, sub_intents, overall_reasoning)

    # 设置第一个子任务为当前 intent（向后兼容）
    first_intent = sub_intents[0] if sub_intents else {
        "type": "chat", "params": {}, "confidence": 0.3,
        "needs_clarification": False, "reasoning": "无法解析意图"
    }

    return {
        "intent": first_intent,
        "sub_intents": sub_intents,
        "current_sub_index": 0,
        "sub_results": [],
        "final_response": final_response,
        # 追问时暂存子任务，下一轮自动合并；执行时清除
        "pending_intents": sub_intents if needs_clarification else None,
    }


def _parse_supervisor_output(text: str) -> dict:
    """解析 Supervisor 的 JSON 输出。"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        result = json.loads(text)
        # 规范化：确保 sub_intents 存在
        if "sub_intents" not in result:
            # 兼容旧格式（单意图）
            result["sub_intents"] = [{
                "type": result.get("type", "chat"),
                "params": result.get("params", {}),
                "confidence": result.get("confidence", 0.5),
                "needs_clarification": result.get("needs_clarification", False),
                "reasoning": result.get("reasoning", ""),
            }]
        result.setdefault("is_complex", len(result["sub_intents"]) > 1)
        result.setdefault("overall_reasoning", "")
        return result
    except json.JSONDecodeError:
        return {
            "is_complex": False,
            "sub_intents": [{
                "type": "chat",
                "params": {},
                "confidence": 0.3,
                "needs_clarification": False,
                "reasoning": "意图解析失败，回退为闲聊模式",
            }],
            "overall_reasoning": "JSON 解析失败",
        }


def _has_question(text: str) -> bool:
    """判断文本是否是追问（询问用户补充信息）。

    用 LLM 判断而非简单问号匹配，避免「有什么需要调整的吗？」
    这类客气话被误判为需要带上下文的追问。
    """
    if not text:
        return False
    # 快速排除：完全没有问号的不可能是追问
    if "？" not in text and "?" not in text:
        return False
    # 追问特征：涉及用户信息补充、选择、确认
    question_markers = [
        "需要我", "要不要", "要不要", "是否", "哪个", "哪种",
        "几点", "多久", "哪天", "哪里", "多少", "怎么",
        "你希望", "你想", "你倾向", "你偏好", "你习惯",
        "可以吗", "好吗", "行吗",
    ]
    return any(marker in text for marker in question_markers)


def _generate_clarification(user_message: str, sub_intents: list, overall_reasoning: str) -> str:
    """生成追问：当子任务中有不确定性时，一次性问清楚。"""
    model = get_model(temperature=0.5)
    unclear = [sub for sub in sub_intents if sub.get("needs_clarification")]
    unclear_desc = "\n".join(
        f"- {sub['type']}: {sub.get('reasoning', '')}" for sub in unclear
    )
    response = model.invoke([
        {
            "role": "system",
            "content": "你是智能秘书。用户的消息中有不够明确的地方，你需要友好地追问。一次性问清楚所有缺失信息，不要反复追问。直接输出追问内容，不要加前缀。",
        },
        {
            "role": "user",
            "content": f"用户消息：{user_message}\n\n整体分析：{overall_reasoning}\n\n不明确的子任务：\n{unclear_desc}\n\n请生成追问：",
        },
    ])
    return response.content
