# State Model Design

## 1. Overview

The State Model defines the internal representation of all observable and inferred game information at each frame.

It transforms raw protocol data into a structured, queryable world state.

---

## 2. Core State Object

### StructuredState

Contains:
- Frame index
- Team state
- Map state
- Task state
- Resource state
- Opponent state (partial)

---

## 3. Map State Representation

### Graph Model

The map is represented as a weighted directed graph:

Nodes:
- S01 ~ S15 (stations)

Edges:
- ROAD
- WATER
- MOUNTAIN
- BRANCH

Each edge contains:
- distance
- traversal cost
- terrain type
- dynamic weight (weather / congestion)

---

## 4. Team State

Tracks controlled assets:

- convoy_position (current node)
- good_fruits
- bad_fruits
- freshness_score
- action_points
- units (small squads)

---

## 5. Task State

Each task instance includes:

- task_id
- task_type (T01, T02...)
- location node
- reward value
- time cost (frames)
- status (available / claimed / completed)

---

## 6. Resource State

Resources include:

- ice storage (freshness control)
- horse units (speed boost)
- ship access (water traversal)
- permits (access control)
- intelligence tokens

---

## 7. Opponent State (Partial Observability)

Due to hidden information:

Observed:
- last known position
- visible actions
- task interactions

Inferred:
- trajectory prediction
- target estimation

---

## 8. Frame State

Each frame contains:

- frame_id (1~600)
- public_state snapshot
- event log

---

## 9. Design Principle

- State must be immutable per frame
- All updates produce a new state object
- No direct mutation of previous frames

---

## 10. Purpose

This model serves as the ONLY input source for:
- Planning Layer
- Strategy Layer
- Action Generation Layer
