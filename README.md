# agent-secretary

基于 LangGraph 的智能秘书，7 个 Agent 协作。支持多意图拆解、用户画像、智能日程规划、多轮对话。

## 运行

```bash
#依赖安装
pip install -r requirements.txt

# 设 DeepSeek API Key
set DEEPSEEK_API_KEY=sk-xxxx          # Windows
export DEEPSEEK_API_KEY=sk-xxxx       # Mac/Linux

#运行
python -m src.main
```

## 示例

```
你：下周五下午3点运动会动员演讲，在操场，帮我写一份演讲稿
秘书：已安排：2026-05-29 15:00-15:45 操场 | 演讲稿已生成，标题「运动会动员演讲稿」
      [复合任务: schedule → document | 完成:2/2]

你：下周三去北京做工作报告
秘书：行程已规划：6月17日全天出差，已查深圳→北京航班 MU5101，报告模板已备好
      [复合任务: schedule → search → document | 完成:3/3]

你：明天下午有什么安排
秘书：明天下午3点与张三开会讨论Q2规划（15:00-17:00）
```

日程/提醒/笔记存在 `data/` 目录的 JSON 文件里。

## 工作原理

```
用户 → Supervisor(意图拆解) → [子任务循环: Worker → Review] → 聚合回复
```

- Supervisor 把复杂请求拆成多个子任务（出差 = 日程 + 航班 + 文档）
- 每个子任务走 Worker(执行) → Review(审查)，单一任务退化为一次循环
- 多个子任务结果用 LLM 融合成一条自然回复
- 追问回复自动带上下文，新话题不带上下文

## 用户画像

编辑 `data/user_profile.md` 定制秘书行为：位置、工作时间、会议偏好、常用联系人、出行偏好等。所有 Agent 在做决策时都会参考。

## Agent 清单

| Agent | 作用 |
|--------|------|
| Supervisor | 意图分类 + 任务拆解，复杂请求自动调度多 Agent |
| Schedule | 创建/查询/删除日程，根据事件类型和画像智能判断 |
| Search | 搜网页、查天气、查航班（目前 mock） |
| Reminder | 定时提醒、条件提醒 |
| Document | 写讲稿、做纪要、写备忘、准备模板 |
| Chat | 闲聊 |
| Review | 防幻觉，最多循环一次 |

## 目录

```
src/
├── main.py            # 命令行入口
├── graph.py           # StateGraph 构建
├── user_profile.py    # 用户画像加载
├── config.py          # API Key 配置
├── models/state.py    # 状态定义
├── agents/            # 7 个 Agent
├── tools/             # 工具集（目前 mock）
data/
├── user_profile.md    # 用户画像（可编辑）
├── schedule.json      # 日程数据
├── reminders.json
└── notes.json
```

## 补充

```
Windows 下中文乱码的话，`set PYTHONIOENCODING=utf-8`。
```