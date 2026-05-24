"""CLI 入口：交互式 Multi-Agent 秘书系统。

支持多轮对话：上下文延续、日程冲突检测、跨轮记忆。

运行方式：
    PYTHONIOENCODING=utf-8 python -m src.main
"""

import uuid
from langchain_core.messages import HumanMessage
from src.aigc.graph import build_graph


def main():
    graph = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("=" * 50)
    print("  Multi-Agent 智能秘书系统 v2")
    print(f"  会话 ID: {thread_id[:8]}...")
    print("  多轮对话 | 冲突检测 | 多Agent协作 | 用户画像")
    print("=" * 50)
    print("  输入 'quit' 退出 | 'clear' 开始新会话")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if user_input.lower() == "clear":
            # 新会话：换 thread_id，checkpoint 从头开始
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            print(f"[新会话 {thread_id[:8]}... 已开始]")
            continue

        # 只需传入新消息，LangGraph checkpoint 自动管理历史
        # sub_intents / pending_intents 由 supervisor 管理，不在此重置
        input_state = {
            "messages": [HumanMessage(content=user_input)],
            "intent": None,
            "sub_results": [],
            "current_sub_index": 0,
            "worker_result": None,
            "review_status": "pending",
            "review_feedback": None,
            "retry_count": 0,
            "final_response": None,
        }

        print("处理中...")
        result = graph.invoke(input_state, config)

        # 输出回复
        final = result.get("final_response", "")
        if final:
            print(f"\n秘书：{final}")
        else:
            for msg in reversed(result.get("messages", [])):
                if getattr(msg, "type", "") == "ai":
                    content = getattr(msg, "content", "") or ""
                    if content and not str(content).startswith("[Review"):
                        print(f"\n秘书：{content}")
                        break

        # 调试信息
        sub_intents = result.get("sub_intents") or []
        sub_results = result.get("sub_results") or []
        if len(sub_intents) > 1:
            types = " → ".join(s.get("type", "?") for s in sub_intents)
            done = len(sub_results)
            print(f"  [复合任务: {types} | 完成:{done}/{len(sub_intents)} | review:{result.get('review_status')}]")
        elif sub_intents:
            s = sub_intents[0]
            print(f"  [{s.get('type')} | 置信度:{s.get('confidence')} | review:{result.get('review_status')}]")


if __name__ == "__main__":
    main()
