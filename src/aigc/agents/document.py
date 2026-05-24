"""Document Agent：文档笔记处理者。"""

from langgraph.prebuilt import create_react_agent
from src.aigc.agents.utils import get_model, make_worker_node
from src.aigc.tools.document_tools import create_note, summarize_text
from src.aigc.user_profile import load_profile


def _build_prompt() -> str:
    profile = load_profile()
    return f"""你是一个智能文档处理助手。你可以创建笔记、写讲稿、做会议纪要、准备模板。

{profile}
## 工作原则

1. **主动创作**：用户需要讲稿就直接写讲稿，需要模板就准备模板，不要反问"需要我帮你写吗"。
2. **内容为王**：产出有实质内容的文档，而不是一两句敷衍。讲稿要有开头正文结尾，模板要有结构框架。
3. **自动命名**：如果用户没有指定标题，根据内容自动生成简洁的标题。
4. **结构化整理**：将散乱的信息整理成有条理的文档。
5. **风格匹配**：根据文档类型调整风格（讲稿口语化，报告正式化，备忘简洁化）。

## 工具

- `create_note`: 创建笔记/备忘/讲稿/模板（参数：title, content, tags）
- `summarize_text`: 对文本进行摘要（参数：text, max_length）

执行完毕后用自然语言告知用户结果，并简要展示创建的内容，末尾标注置信度（格式：`置信度：0.9`）。
"""


def create_document_agent():
    model = get_model()
    return create_react_agent(
        model=model,
        tools=[create_note, summarize_text],
        prompt=_build_prompt(),
        name="document_agent",
    )


_document_agent = None
_document_node_fn = None


def document_node(state):
    global _document_agent, _document_node_fn
    if _document_agent is None:
        _document_agent = create_document_agent()
        _document_node_fn = make_worker_node(_document_agent)
    return _document_node_fn(state)
