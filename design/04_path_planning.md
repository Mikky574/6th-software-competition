# Path Planning Design

## 1. Overview

Path planning is responsible for computing optimal routes on the map graph (S01-S15).

---

## 2. Graph Model

The environment is modeled as a weighted directed graph:

Nodes:
- S01 ~ S15

Edges:
- ROAD
- WATER
- MOUNTAIN
- BRANCH

Each edge has dynamic weight:

W = base_distance + terrain_cost + weather_penalty + congestion_penalty

---

## 3. Core Algorithms

### 3.1 Dijkstra (default)
Used for deterministic shortest path.

### 3.2 A* (optional)
Used when heuristic estimation is available.

Heuristic:
- Euclidean distance between nodes

---

## 4. Dynamic Weighting System

Weights updated per frame:

- Rain → WATER cost increases
- Mountain → slower traversal
- Enemy presence → congestion penalty

---

## 5. Multi-Criteria Optimization

Path cost function:

Cost(path) =
- travel_time
- resource consumption
- risk exposure

---

## 6. Route Selection Strategy

Select path minimizing:

Score = α * time + β * risk + γ * reward_opportunity

Default:
- α = 0.6
- β = 0.3
- γ = 0.1

---

## 7. Replanning Strategy

Recompute path when:
- opponent detected nearby
- task updated
- weather changed
- congestion threshold exceeded

---

## 8. Design Principle

- Graph must be updated every frame
- Path is never static
- Always prefer adaptive planning
