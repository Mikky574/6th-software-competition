# 项目结构设计说明（工程化规范）

本仓库按照比赛工程化标准，划分为三大核心模块，用于实现 **解耦开发 + 可扩展AI系统 + 可维护策略框架**。

---

# 🧱 一、整体目录结构

```
6th-software-competition/
│
├── competition/                 # 📌 赛题文件夹（原始规则与协议）
│   ├── communication_part1.md
│   ├── communication_part2.md
│   ├── rules.md
│   └── task_description.md
│
├── design/                      # 🧠 设计文件夹（系统架构与方案）
│   ├── design_engine.md
│   ├── system_architecture.md
│   ├── strategy_design.md
│   └── decision_model.md
│
├── src/                         # 💻 代码文件夹（核心实现）
│   ├── core/
│   ├── world/
│   ├── agent/
│   ├── strategy/
│   ├── decision/
│   └── utils/
│
└── README.md
```

---

# 📌 二、三大模块职责说明

## 1️⃣ competition（赛题层）

职责：**只存放规则与协议，不写任何代码逻辑**

内容包括：
- 通信协议（inquire / action）
- 游戏规则
- 任务定义
- 评分机制

特点：
- ❌ 不允许写代码实现
- ❌ 不允许策略逻辑
- ✔ 仅作为“规则源”

---

## 2️⃣ design（设计层）

职责：**系统架构设计与算法方案定义**

内容包括：
- 系统分层架构
- Agent设计
- World Model设计
- Strategy设计
- Decision机制设计

特点：
- ✔ 用于指导实现
- ✔ 可以包含伪代码
- ❌ 不直接运行

---

## 3️⃣ src（代码层）

职责：**实际运行的AI系统实现**

模块划分：

### core/
- state管理
- protocol解析

### world/
- 图结构
- 路径规划（A* / Dijkstra）

### agent/
- 决策Agent
- rule/greedy/rl agent

### strategy/
- resource策略
- task策略
- contest策略

### decision/
- score函数
- action选择器

### utils/
- log
- time
- math工具

---

# 🔁 三、数据流（核心运行链路）

```
competition (规则)
        ↓
protocol parser
        ↓
state update
        ↓
world model build
        ↓
strategy evaluation
        ↓
decision engine
        ↓
action builder
        ↓
send to server
```

---

# 🧠 四、设计原则（非常重要）

## ✔ 1. 完全解耦
- 赛题 ≠ 设计 ≠ 代码

## ✔ 2. 单向依赖
```
competition → design → src
```

## ✔ 3. 可替换性
- strategy 可替换
- decision 可替换
- world model 可替换

## ✔ 4. 可扩展性
支持：
- rule-based
- heuristic
- ML model
- RL policy

---

# 🚀 五、工程目标

本结构用于支持：

- ✔ 比赛快速开发baseline
- ✔ 多策略对比实验
- ✔ 可扩展AI agent系统
- ✔ 后期强化学习接入

---

# 📌 六、总结

该结构的核心目标是：

> 将“赛题理解、系统设计、代码实现”彻底解耦，形成可持续演进的AI竞赛工程体系。
