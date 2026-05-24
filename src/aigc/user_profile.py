"""用户画像加载器。

读取 data/user_profile.md，格式化为 prompt 可用的上下文文本。
"""

import os


PROFILE_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "user_profile.md")


def load_profile() -> str:
    """加载用户画像，返回格式化的上下文字符串。

    如果文件不存在，返回空字符串。
    """
    if not os.path.exists(PROFILE_FILE):
        return ""

    with open(PROFILE_FILE, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return ""

    return f"""## 用户画像（请在所有决策中参考以下信息）

{content}

---
"""
