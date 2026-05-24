"""Agent 共享工具：LLM 工厂、Worker 节点工厂、WorkerResult 提取。"""

from __future__ import annotations

import json
import re
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from src.aigc.config import settings
from src.aigc.models.state import AgentState, WorkerResult


def get_model(
    temperature: float = 0.3,
    reasoning_effort: str | None = None,
) -> ChatOpenAI:
    """获取 DeepSeek LLM 实例。

    Args:
        temperature: 温度参数，Router/Review 用低值保证一致性，闲聊可调高
        reasoning_effort: 推理强度 (low/medium/high)，None 表示不开启 thinking
    """
    kwargs = dict(
        model=settings.deepseek_model,
        base_url=settings.deepseek_base_url,
        api_key=settings.deepseek_api_key,
        temperature=temperature,
        max_retries=3,
        extra_body={"thinking": {"type": "disabled"}},
    )
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    return ChatOpenAI(**kwargs)


def extract_worker_result(messages: list, skip_count: int = 0) -> WorkerResult:
    """从消息历史中提取 WorkerResult。

    扫描 AIMessage 中的 tool_calls 和对应的 ToolMessage 结果，
    将最后一条 AI 回复作为 reasoning。

    Args:
        messages: 完整消息列表
        skip_count: 跳过的消息数量（用于多 Agent 场景，只提取当前 Agent 新增的消息）
    """
    actions_executed = []
    last_ai_content = ""

    relevant = messages[skip_count:] if skip_count > 0 else messages

    for msg in relevant:
        msg_dict = msg if isinstance(msg, dict) else _msg_to_dict(msg)
        role = msg_dict.get("role") or msg_dict.get("type", "")

        if role == "ai" or msg.__class__.__name__ == "AIMessage":
            content = msg_dict.get("content", "") or ""
            if content:
                last_ai_content = content
            tool_calls = msg_dict.get("tool_calls", [])
            for tc in tool_calls:
                actions_executed.append({
                    "tool": tc.get("name", ""),
                    "args": tc.get("args", {}),
                })

        elif role == "tool" or msg.__class__.__name__ == "ToolMessage":
            tool_name = msg_dict.get("name", "")
            tool_result = msg_dict.get("content", "")
            # 将结果关联到对应的 action
            for action in actions_executed:
                if action["tool"] == tool_name and "result" not in action:
                    action["result"] = tool_result
                    break

    confidence = _parse_confidence(last_ai_content)
    return {
        "reasoning": last_ai_content,
        "actions_executed": actions_executed,
        "confidence": confidence,
        "notes": None,
    }


def _parse_confidence(text: str) -> float:
    """从 Worker 输出中解析置信度。

    支持格式：置信度：0.9 / 置信度: 0.9 / confidence: 0.9
    未找到时基于是否有实际行动做启发式判断。
    """
    match = re.search(r"(?:置信度|confidence)[：:]\s*([\d.]+)", text, re.IGNORECASE)
    if match:
        try:
            value = float(match.group(1))
            return max(0.0, min(1.0, value))
        except ValueError:
            pass
    return 0.5


def _msg_to_dict(msg) -> dict:
    """将 LangChain 消息对象转为 dict。"""
    if hasattr(msg, "model_dump"):
        return msg.model_dump()
    if hasattr(msg, "dict"):
        return msg.dict()
    return {"role": getattr(msg, "type", "unknown"), "content": getattr(msg, "content", "")}


def make_worker_node(agent_graph):
    """创建 Worker 节点函数。

    将 create_react_agent 编译图包装为 StateGraph 节点，
    运行后自动从消息中提取 WorkerResult。

    支持 review 反馈：当 state.review_feedback 非空时，
    将其作为系统指令注入，引导 Worker 修正上一轮的问题。

    Args:
        agent_graph: create_react_agent 返回的编译图
    Returns:
        节点函数 (state: AgentState) -> dict
    """
    def node(state: AgentState) -> dict:
        messages = list(state["messages"])
        review_feedback = state.get("review_feedback")
        if review_feedback:
            feedback_msg = HumanMessage(
                content=(
                    f"[系统审查反馈] 你的上一次执行存在以下问题：\n"
                    f"{review_feedback}\n\n"
                    "这是系统的自动审查意见，不是用户的消息。请根据以上反馈修正你的执行结果，"
                    "然后继续处理用户的原始请求。"
                )
            )
            messages.append(feedback_msg)

        prev_count = len(messages)
        result = agent_graph.invoke({"messages": messages})
        worker_result = extract_worker_result(result["messages"], skip_count=prev_count)
        return {
            "messages": result["messages"],
            "worker_result": worker_result,
        }
    return node
