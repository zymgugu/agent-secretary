"""Chat Agent：闲聊兜底，处理非任务型对话。"""

from langchain_core.messages import AIMessage
from src.aigc.agents.utils import get_model
from src.aigc.models.state import AgentState
from src.aigc.user_profile import load_profile


def _build_prompt() -> str:
    profile = load_profile()
    return f"""你是一个友好、专业的智能秘书。处理日常对话和通用问答。

{profile}

- 语气自然亲切但不失专业
- 可以聊天、回答问题、提供建议
- 如果用户突然提出任务型需求（日程、搜索等），提醒用户你可以帮忙处理
- 不要在闲聊中编造日程、提醒等操作，这些有专门的工具处理

直接回复用户，不要加前缀或标记。
"""


def chat_node(state: AgentState) -> dict:
    """Chat Agent 节点：自然语言回复，不调用工具，不走 Review。"""
    model = get_model(temperature=0.7)

    messages = state["messages"]
    response = model.invoke(
        [{"role": "system", "content": _build_prompt()}] + list(messages)
    )

    return {
        "messages": [response],
        "worker_result": {
            "reasoning": response.content,
            "actions_executed": [],
            "confidence": 1.0,
            "notes": None,
        },
    }
