"""提醒工具 — 基于 JSON 文件存储。

存储位置：data/reminders.json
"""

import json
import os
import uuid
from datetime import datetime, timezone, timedelta

TZ = timezone(timedelta(hours=8))
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "reminders.json")


def _load() -> dict:
    if not os.path.exists(DATA_FILE):
        return {"reminders": []}
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict) -> None:
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def set_reminder(content: str, trigger_time: str, repeat: str = "once") -> dict:
    """设置定时提醒。

    Args:
        content: 提醒内容
        trigger_time: 触发时间 (ISO8601 格式)
        repeat: 重复模式 (once/daily/weekly/monthly)
    """
    reminder_id = f"rem-{uuid.uuid4().hex[:8]}"
    reminder = {
        "reminder_id": reminder_id,
        "content": content,
        "trigger_time": trigger_time,
        "repeat": repeat,
        "created_at": datetime.now(TZ).isoformat(),
    }
    data = _load()
    data["reminders"].append(reminder)
    _save(data)
    return {
        "status": "success",
        "reminder_id": reminder_id,
        "content": content,
        "trigger_time": trigger_time,
        "repeat": repeat,
        "message": f"提醒已设置：{content}（{trigger_time}，{repeat}）",
    }


def set_condition_reminder(condition: str, content: str) -> dict:
    """设置条件提醒（满足条件时触发）。

    Args:
        condition: 触发条件描述
        content: 提醒内容
    """
    reminder_id = f"rem-{uuid.uuid4().hex[:8]}"
    reminder = {
        "reminder_id": reminder_id,
        "content": content,
        "condition": condition,
        "type": "condition",
        "created_at": datetime.now(TZ).isoformat(),
    }
    data = _load()
    data["reminders"].append(reminder)
    _save(data)
    return {
        "status": "success",
        "reminder_id": reminder_id,
        "condition": condition,
        "content": content,
        "message": f"条件提醒已设置：当 {condition} 时提醒「{content}」",
    }


def cancel_reminder(reminder_id: str) -> dict:
    """取消指定提醒。

    Args:
        reminder_id: 提醒ID
    """
    data = _load()
    for i, rem in enumerate(data["reminders"]):
        if rem["reminder_id"] == reminder_id:
            content = rem["content"]
            data["reminders"].pop(i)
            _save(data)
            return {"status": "success", "reminder_id": reminder_id, "message": f"提醒「{content}」已取消"}
    return {"status": "error", "reminder_id": reminder_id, "message": f"提醒 {reminder_id} 不存在"}
