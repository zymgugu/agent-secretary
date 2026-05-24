"""Reminder Agent：提醒设置执行者。"""

from langgraph.prebuilt import create_react_agent
from src.aigc.agents.utils import get_model, make_worker_node
from src.aigc.tools.reminder_tools import set_reminder, set_condition_reminder, cancel_reminder
from src.aigc.user_profile import load_profile


def _build_prompt() -> str:
    profile = load_profile()
    return f"""你是一个智能提醒助手。你可以直接调用工具设置提醒。

{profile}
## 工作原则

1. **主动执行**：用户让你设置提醒，直接调用工具完成。
2. **时间解析**：从自然语言中提取提醒时间（"明天早上9点"、"每周五下午3点"），结合用户画像中的工作时间判断合理性。
3. **重复模式判断**：
   - 每天/每日 → repeat="daily"
   - 每周/每周末 → repeat="weekly"
   - 每月/月底 → repeat="monthly"
   - 一次性/明天 → repeat="once"
4. **执行后告知**：告诉用户提醒已设置，何时触发。
5. **禁止编造**：用户没说的提醒内容不要自己加。

## 工具

- `set_reminder`: 设置定时提醒
- `set_condition_reminder`: 设置条件提醒
- `cancel_reminder`: 取消提醒

执行完毕后用自然语言告知用户结果，并在末尾标注置信度（格式：`置信度：0.9`）。
"""


def create_reminder_agent():
    model = get_model()
    return create_react_agent(
        model=model,
        tools=[set_reminder, set_condition_reminder, cancel_reminder],
        prompt=_build_prompt(),
        name="reminder_agent",
    )


_reminder_agent = None
_reminder_node_fn = None


def reminder_node(state):
    global _reminder_agent, _reminder_node_fn
    if _reminder_agent is None:
        _reminder_agent = create_reminder_agent()
        _reminder_node_fn = make_worker_node(_reminder_agent)
    return _reminder_node_fn(state)
