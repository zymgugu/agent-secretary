"""日程工具 — 基于 JSON 文件存储，接口对齐 vivo 日历 API。

存储位置：data/schedule.json
队友可直接读取该文件做前端展示。

特性：
- JSON 文件持久化，重启不丢失
- 自动清理过期日程（end_time 已过的）
- 冲突检测：查询已有日程后自行判断时间重叠
"""

import json
import os
import uuid
from datetime import datetime, timezone, timedelta

# 北京时间
TZ = timezone(timedelta(hours=8))

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "schedule.json")


def _load() -> dict:
    """从 JSON 文件加载所有日程，并自动清理过期事件。"""
    if not os.path.exists(DATA_FILE):
        return {"events": []}

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 自动清理：移除 end_time 已在过去的事件
    now = datetime.now(TZ)
    active = []
    cleaned = 0
    for evt in data.get("events", []):
        try:
            end_str = evt.get("end_time", "")
            end_time = datetime.fromisoformat(end_str)
            if end_time < now:
                cleaned += 1
                continue
        except (ValueError, TypeError):
            pass
        active.append(evt)

    if cleaned > 0:
        _save({"events": active})

    return {"events": active}


def _save(data: dict) -> None:
    """保存日程到 JSON 文件。"""
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def add_calendar_event(
    title: str,
    start_time: str,
    end_time: str,
    description: str = "",
    location: str = "",
    remind_before_minutes: int = 15,
) -> dict:
    """创建日历事件。"""
    event_id = f"evt-{uuid.uuid4().hex[:8]}"
    event = {
        "event_id": event_id,
        "title": title,
        "start_time": start_time,
        "end_time": end_time,
        "description": description,
        "location": location,
        "remind_before_minutes": remind_before_minutes,
        "created_at": datetime.now(TZ).isoformat(),
        "status": "active",
    }

    data = _load()
    data["events"].append(event)
    _save(data)

    return {
        "status": "success",
        "event_id": event_id,
        "title": title,
        "start_time": start_time,
        "end_time": end_time,
        "message": f"日程「{title}」已创建（{start_time} ~ {end_time}）",
    }


def query_calendar(start_date: str, end_date: str) -> dict:
    """查询指定日期范围内的日历事件。

    Args:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
    """
    data = _load()
    matched = []
    for evt in data["events"]:
        evt_start = evt["start_time"][:10]
        if start_date <= evt_start <= end_date:
            matched.append(evt)

    if not matched:
        return {
            "status": "success",
            "events": [],
            "message": f"{start_date} 至 {end_date} 暂无日程",
        }
    return {
        "status": "success",
        "events": matched,
        "message": f"找到 {len(matched)} 个日程",
    }


def delete_calendar_event(event_id: str) -> dict:
    """删除日历事件。"""
    data = _load()
    for i, evt in enumerate(data["events"]):
        if evt["event_id"] == event_id:
            title = evt["title"]
            data["events"].pop(i)
            _save(data)
            return {"status": "success", "event_id": event_id, "message": f"日程「{title}」已删除"}
    return {"status": "error", "event_id": event_id, "message": f"事件 {event_id} 不存在"}
