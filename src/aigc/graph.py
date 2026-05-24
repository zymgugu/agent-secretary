"""主 StateGraph：Supervisor → 子任务循环 → 聚合 → 输出。

图结构（支持多意图协作）：

    START → supervisor ──→ prepare_subtask ──→ worker ──→ review ──→ advance
                 │                                    ↑         ↑          │
                 │                          (retry)   └─────────┘          │
                 │                                                         │
                 └──→ finalize ←── (done) ─────────────────────────────────┘
                         ↑
              chat ──────┘ (chat 跳过 review)

单一意图退化为一次循环，行为与旧版一致。
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.aigc.models.state import AgentState
from src.aigc.config import settings

from src.aigc.agents.supervisor import supervisor_node
from src.aigc.agents.schedule import schedule_node
from src.aigc.agents.search import search_node
from src.aigc.agents.reminder import reminder_node
from src.aigc.agents.document import document_node
from src.aigc.agents.chat import chat_node
from src.aigc.agents.review import review_node


# ── 路由函数 ─────────────────────────────────────────────


def route_supervisor(state: AgentState) -> str:
    """Supervisor 之后的第一个路由。

    - 需要追问 → 直接 finalize（supervisor 已设置 final_response）
    - 有子任务 → prepare_subtask 开始循环
    - 其他 → finalize（兜底）
    """
    if state.get("final_response"):
        return "finalize"
    sub_intents = state.get("sub_intents") or []
    if sub_intents:
        return "prepare_subtask"
    return "finalize"


def route_by_intent(state: AgentState) -> str:
    """prepare_subtask 之后：按当前子任务类型路由到对应 Worker。"""
    intent = state.get("intent") or {}
    intent_type = intent.get("type", "chat")
    mapping = {
        "schedule": "schedule",
        "search": "search",
        "reminder": "reminder",
        "document": "document",
        "chat": "chat",
    }
    return mapping.get(intent_type, "chat")


def route_after_review(state: AgentState) -> str:
    """Review 之后：退回重做 或 进入 advance。"""
    review_status = state.get("review_status", "approved")
    retry_count = state.get("retry_count", 0)

    if review_status == "rejected" and retry_count < settings.max_retry_count:
        intent = state.get("intent") or {}
        return intent.get("type", "chat")
    return "advance"


def route_advance(state: AgentState) -> str:
    """advance 之后：还有子任务 → prepare_subtask，全部完成 → finalize。"""
    sub_intents = state.get("sub_intents") or []
    current_idx = state.get("current_sub_index", 0)

    if current_idx < len(sub_intents):
        return "prepare_subtask"
    return "finalize"


# ── 节点函数 ─────────────────────────────────────────────


def prepare_subtask_node(state: AgentState) -> dict:
    """子任务准备节点：从 sub_intents 中取出当前任务，设置为 intent。

    首次由 supervisor 触发（current_sub_index=0），
    后续由 advance 触发（current_sub_index 已递增）。
    """
    sub_intents = state.get("sub_intents") or []
    current_idx = state.get("current_sub_index", 0)

    if current_idx >= len(sub_intents):
        return {}

    current_intent = sub_intents[current_idx]

    return {
        "intent": current_intent,
        "worker_result": None,
        "review_status": "pending",
        "review_feedback": None,
        "retry_count": 0,
    }


def advance_node(state: AgentState) -> dict:
    """推进节点：保存当前子任务结果，递增索引。

    在 Review 通过后调用（chat 跳过 Review 直接到这里）。
    """
    sub_intents = state.get("sub_intents") or []
    current_idx = state.get("current_sub_index", 0)
    sub_results = list(state.get("sub_results") or [])

    # 保存当前子任务结果（去重：不超过当前索引）
    current_result = {
        "intent": state.get("intent"),
        "result": state.get("worker_result"),
        "review_status": state.get("review_status", "approved"),
    }
    if len(sub_results) <= current_idx:
        sub_results.append(current_result)
    else:
        sub_results[current_idx] = current_result

    next_idx = current_idx + 1

    # 如果还有下一个子任务，预先设置 intent
    if next_idx < len(sub_intents):
        return {
            "sub_results": sub_results,
            "current_sub_index": next_idx,
        }

    return {
        "sub_results": sub_results,
        "current_sub_index": next_idx,  # >= len，route_advance 会走到 finalize
    }


def finalize_node(state: AgentState) -> dict:
    """最终节点：组装最终回复。

    单意图：直接输出 worker_result.reasoning（向后兼容）。
    多意图：用 LLM 将多个子任务结果融合为一条连贯回复。
    """
    if state.get("final_response"):
        return {}

    sub_results = state.get("sub_results") or []
    review_feedback = state.get("review_feedback", "")
    review_status = state.get("review_status", "")

    # 单意图或无子结果：直接取 worker_result（向后兼容）
    if len(sub_results) <= 1:
        worker_result = state.get("worker_result") or {}
        worker_text = worker_result.get("reasoning", "")
        # 如果 worker_result 为空，尝试从 sub_results 取第一个
        if not worker_text and sub_results:
            worker_text = (sub_results[0].get("result") or {}).get("reasoning", "")
        if review_status == "flagged" and review_feedback:
            worker_text = f"{worker_text}\n\n💡 {review_feedback}"
        if worker_text:
            return {"final_response": worker_text}
        # fallback
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if getattr(msg, "type", "") == "ai":
                content = getattr(msg, "content", "") or ""
                if content and not str(content).startswith("[Review"):
                    return {"final_response": content}
        return {"final_response": "抱歉，处理过程出现了问题，请重试。"}

    # 多意图：用 LLM 融合多个子结果
    return _aggregate_multi(sub_results, review_status, review_feedback, state)


def _aggregate_multi(
    sub_results: list,
    review_status: str,
    review_feedback: str,
    state: AgentState,
) -> dict:
    """用 LLM 将多个子任务结果融合为一条自然连贯的回复。"""
    from src.aigc.agents.utils import get_model

    parts_desc = []
    for i, sub in enumerate(sub_results):
        intent = sub.get("intent") or {}
        result = sub.get("result") or {}
        parts_desc.append(
            f"子任务{i+1} [{intent.get('type', '')}]：{result.get('reasoning', '无输出')}"
        )

    model = get_model(temperature=0.3)
    response = model.invoke([
        {
            "role": "system",
            "content": (
                "你是智能秘书，需要将多个子任务的执行结果融合为一条连贯、自然的回复。\n"
                "规则：\n"
                '- 直接对用户说话，不要用「子任务1、子任务2」这种内部分解术语\n'
                "- 按逻辑顺序组织内容（先日程，再文档，最后其他）\n"
                "- 去掉重复信息，每个事实只说一次\n"
                "- 语气自然友好，不要像机器人在列清单\n"
                "- 如果某个子任务只是辅助性的（如查询），不需要单独提及，融入主体即可"
            ),
        },
        {
            "role": "user",
            "content": f"用户说的是：{_get_user_message(state)}\n\n各子任务执行结果：\n\n" + "\n".join(parts_desc) + "\n\n请融合为一条回复：",
        },
    ])

    final_text = response.content
    if review_status == "flagged" and review_feedback:
        final_text = f"{final_text}\n\n💡 {review_feedback}"

    return {"final_response": final_text}


def _get_user_message(state: AgentState) -> str:
    """从 state 中提取用户原始消息。"""
    for msg in reversed(state.get("messages", [])):
        if getattr(msg, "type", "") == "human":
            return getattr(msg, "content", "") or ""
    return ""


# ── 图构建 ───────────────────────────────────────────────


def build_graph() -> StateGraph:
    """构建并编译 StateGraph。"""
    workflow = StateGraph(AgentState)

    # 注册节点
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("prepare_subtask", prepare_subtask_node)
    workflow.add_node("schedule", schedule_node)
    workflow.add_node("search", search_node)
    workflow.add_node("reminder", reminder_node)
    workflow.add_node("document", document_node)
    workflow.add_node("chat", chat_node)
    workflow.add_node("review", review_node)
    workflow.add_node("advance", advance_node)
    workflow.add_node("finalize", finalize_node)

    # START → supervisor
    workflow.add_edge(START, "supervisor")

    # supervisor → prepare_subtask 或 finalize
    workflow.add_conditional_edges("supervisor", route_supervisor, {
        "prepare_subtask": "prepare_subtask",
        "finalize": "finalize",
    })

    # prepare_subtask → worker（按当前 intent.type 路由）
    workflow.add_conditional_edges("prepare_subtask", route_by_intent, {
        "schedule": "schedule",
        "search": "search",
        "reminder": "reminder",
        "document": "document",
        "chat": "chat",
    })

    # Worker → review（chat 直接到 advance）
    for name in ["schedule", "search", "reminder", "document"]:
        workflow.add_edge(name, "review")
    workflow.add_edge("chat", "advance")

    # review → worker（重试）或 advance（继续）
    workflow.add_conditional_edges("review", route_after_review, {
        "schedule": "schedule",
        "search": "search",
        "reminder": "reminder",
        "document": "document",
        "chat": "chat",
        "advance": "advance",
    })

    # advance → prepare_subtask（还有子任务）或 finalize（全部完成）
    workflow.add_conditional_edges("advance", route_advance, {
        "prepare_subtask": "prepare_subtask",
        "finalize": "finalize",
    })

    # finalize → END
    workflow.add_edge("finalize", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
