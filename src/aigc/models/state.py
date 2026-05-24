from __future__ import annotations

from typing import Annotated, TypedDict, Optional
from langgraph.graph.message import add_messages


class WorkerResult(TypedDict, total=False):
    """Worker Agent 的输出结构：思考链 + 已执行的 actions + 置信度 + 备注"""
    reasoning: str
    actions_executed: list[dict]
    confidence: float
    notes: Optional[str]


class Intent(TypedDict):
    """Supervisor 意图分类结果（单个子任务）"""
    type: str  # schedule | search | reminder | document | chat
    params: dict
    confidence: float
    needs_clarification: bool
    reasoning: str


class SubResult(TypedDict, total=False):
    """单个子任务的执行结果"""
    intent: dict       # Intent
    result: dict       # WorkerResult
    review_status: str


class AgentState(TypedDict):
    """全局 State，贯穿整个 StateGraph

    支持多意图：Supervisor 可将复杂任务拆解为 sub_intents，
    逐个执行后结果汇入 sub_results，最终聚合为一条回复。
    """
    messages: Annotated[list, add_messages]
    # Supervisor 输出
    intent: Optional[dict]           # 当前正在处理的子任务 Intent
    sub_intents: Optional[list[dict]]  # 所有待处理的子任务列表
    # 子任务进度
    current_sub_index: int           # 当前子任务索引（0-based）
    sub_results: Optional[list[dict]]  # 已完成子任务的结果 SubResult[]
    # Worker 输出
    worker_result: Optional[dict]    # WorkerResult
    # Review
    review_status: str               # pending | approved | rejected | flagged
    review_feedback: Optional[str]
    retry_count: int
    # 追问暂存：当 Supervisor 生成追问时，将子任务暂存于此，下一轮自动合并
    pending_intents: Optional[list[dict]]
    # 最终输出
    final_response: Optional[str]
