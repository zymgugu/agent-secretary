"""Search Agent：信息检索执行者。"""

from langgraph.prebuilt import create_react_agent
from src.aigc.agents.utils import get_model, make_worker_node
from src.aigc.tools.search_tools import web_search, weather_query, flight_query
from src.aigc.user_profile import load_profile


def _build_prompt() -> str:
    profile = load_profile()
    return f"""你是一个智能信息检索助手。你可以直接调用搜索工具获取信息。

{profile}
## 工作原则

1. **主动搜索**：理解意图后直接调用工具搜索，不要反问"需要我帮你搜索吗"。
2. **自然总结**：搜索结果用自然语言总结，不要只丢链接。
3. **多维度搜索**：如果用户问题涉及多个方面（如"北京和上海的天气"），分别查询。
4. **利用用户画像**：搜航班、出行等信息时，优先使用用户画像中的偏好（出发地、高铁/飞机偏好等）。
5. **明确告知来源**：告知用户信息来自哪个渠道。

## 工具

- `web_search`: 网页搜索
- `weather_query`: 天气查询
- `flight_query`: 航班查询

执行完毕后用自然语言汇报结果，并在末尾标注置信度（格式：`置信度：0.9`）。
"""


def create_search_agent():
    model = get_model()
    return create_react_agent(
        model=model,
        tools=[web_search, weather_query, flight_query],
        prompt=_build_prompt(),
        name="search_agent",
    )


_search_agent = None
_search_node_fn = None


def search_node(state):
    global _search_agent, _search_node_fn
    if _search_agent is None:
        _search_agent = create_search_agent()
        _search_node_fn = make_worker_node(_search_agent)
    return _search_node_fn(state)
