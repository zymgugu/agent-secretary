"""搜索工具 — 当前为 mock 实现。"""


def web_search(query: str) -> dict:
    """搜索网页。

    Args:
        query: 搜索关键词
    """
    return {
        "status": "success",
        "results": [
            {
                "title": f"关于「{query}」的搜索结果（mock）",
                "url": "https://example.com/mock-result",
                "snippet": f"这是关于 {query} 的模拟搜索结果摘要。实际接入后将返回真实数据。",
            }
        ],
        "message": f"已完成搜索：{query}（mock）",
    }


def weather_query(city: str) -> dict:
    """查询天气。

    Args:
        city: 城市名称
    """
    return {
        "status": "success",
        "city": city,
        "weather": "晴",
        "temperature": "25°C",
        "humidity": "60%",
        "message": f"{city} 天气：晴，25°C（mock）",
    }


def flight_query(origin: str, destination: str, date: str = "") -> dict:
    """查询航班信息。

    Args:
        origin: 出发城市
        destination: 到达城市
        date: 日期 (YYYY-MM-DD)，可选
    """
    return {
        "status": "success",
        "flights": [
            {
                "flight_number": "MU5101",
                "origin": origin,
                "destination": destination,
                "departure_time": f"{date or '2026-05-22'}T08:00:00",
                "arrival_time": f"{date or '2026-05-22'}T10:30:00",
                "price": "¥1280",
            }
        ],
        "message": f"已查询 {origin} → {destination} 航班（mock）",
    }
