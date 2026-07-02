# 工程级设计说明（6th Software Competition）

## 1. 项目定位
本项目为一个“多智能体策略对抗系统”，核心运行模式为：

- 服务端：逐帧推演（Frame-based Game Engine）
- 客户端：逐帧决策（Policy Agent System）
- 目标：最大化最终得分（Score Optimization Problem）

---

## 2. 系统整体架构

### 2.1 总体分层

```
Client (AI Agent)
    │
    ├── Protocol Layer（通信协议）
    ├── State Layer（状态维护）
    ├── World Layer（地图/图结构）
    ├── Decision Layer（策略决策）
    ├── Strategy Layer（业务策略）
    └── Action Layer（动作生成）
    │
Server (Game Engine)
```

---

## 3. 核心模块设计

### 3.1 Protocol Layer（协议层）
职责：解析 inquire / 序列化 action / 校验合法性

### 3.2 State Layer
维护全局状态、历史帧、对象状态

### 3.3 World Layer
图结构 + A* / Dijkstra + 成本模型

### 3.4 Decision Layer
action = argmax(strategy_score)

### 3.5 Strategy Layer
resource / task / contest / defense / squad

### 3.6 Action Layer
MOVE / CLAIM_TASK / CLAIM_RESOURCE / SET_GUARD / BREAK_GUARD / WINDOW_CARD

---

## 4. 数据流
inquire → parser → state → world → decision → strategy → action → send

---

## 5. 核心难点
多目标优化 + 窗口博弈 + 动态天气 + 路径阻塞

---

## 6. 推荐架构
Baseline / Heuristic / Competition-level / RL

---

## 7. 工程目标
稳定600帧运行 + 可扩展策略 + 无崩溃 + 可恢复断线
