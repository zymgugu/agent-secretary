"""文档工具 — 基于 JSON 文件存储。

存储位置：data/notes.json
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

TZ = timezone(timedelta(hours=8))
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "notes.json")


def _load() -> dict:
    if not os.path.exists(DATA_FILE):
        return {"notes": []}
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict) -> None:
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def create_note(title: str, content: str, tags: Optional[list[str]] = None) -> dict:
    """创建笔记、备忘、讲稿或模板文档。

    Args:
        title: 文档标题
        content: 文档内容
        tags: 标签列表，可选
    """
    note_id = f"note-{uuid.uuid4().hex[:8]}"
    note = {
        "note_id": note_id,
        "title": title,
        "content": content,
        "tags": tags or [],
        "created_at": datetime.now(TZ).isoformat(),
    }
    data = _load()
    data["notes"].append(note)
    _save(data)
    return {
        "status": "success",
        "note_id": note_id,
        "title": title,
        "message": f"笔记《{title}》已创建",
    }


def summarize_text(text: str, max_length: int = 200) -> dict:
    """对长文本进行摘要提取。

    Args:
        text: 待摘要的原始文本
        max_length: 摘要最大长度，默认200字
    """
    if len(text) <= max_length:
        return {
            "status": "success",
            "summary": text,
            "original_length": len(text),
            "message": "原文较短，无需摘要",
        }

    from src.aigc.agents.utils import get_model

    model = get_model(temperature=0.1)
    response = model.invoke([
        {
            "role": "system",
            "content": (
                f"你是一个文本摘要工具。请将用户提供的文本总结为不超过{max_length}字的摘要。"
                "保留关键事实、数字和人名。只输出摘要文本，不要加任何前缀或说明。"
            ),
        },
        {"role": "user", "content": text},
    ])

    return {
        "status": "success",
        "summary": response.content.strip(),
        "original_length": len(text),
        "message": "摘要生成完成",
    }
