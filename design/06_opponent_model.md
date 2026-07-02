# Opponent Model Design

## 1. Overview

The opponent model predicts enemy behavior under partial observability.

---

## 2. Observable Signals

We can observe:
- last known position
- visible actions
- task interactions
- resource usage hints

---

## 3. Inference Objectives

We aim to predict:

- next node position
- target task
- interception intention
- resource strategy

---

## 4. Prediction Model

### 4.1 Heuristic Model (Baseline)

Opponent score function:

Score(node) =
- proximity to tasks
- shortest path efficiency
- reward density

---

### 4.2 Path Inference

Assume opponent uses shortest path (Dijkstra-like behavior)

Predict next position along shortest route.

---

## 5. Threat Estimation

Each node is assigned threat level:

Threat(node) =
- probability(opponent passes)
- task competition overlap
- interception potential

---

## 6. Interception Prediction

We compute:

- possible collision nodes
- shared path segments
- contest windows

---

## 7. Limitations

- partial observability
- stochastic opponent behavior
- delayed information updates

---

## 8. Design Principle

- lightweight heuristic first
- upgrade to ML model if needed
- always frame-consistent
