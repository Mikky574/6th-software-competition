# Task Strategy Design

## 1. Overview

Task strategy determines how to select, prioritize, and execute royal tasks (皇榜任务).

---

## 2. Task Model

Each task instance contains:

- task_id
- location
- reward
- time_cost (frames)
- competition level
- expiration risk

---

## 3. Task Scoring Function

Each task is evaluated by:

Score(task) =

Reward
- λ1 * travel_time
- λ2 * completion_risk
- λ3 * competition_pressure
+ λ4 * strategic_value

Default weights:
- λ1 = 0.5
- λ2 = 0.2
- λ3 = 0.2
- λ4 = 0.1

---

## 4. Task Selection Policy

### Greedy Strategy
Select task with highest score / cost ratio.

---

## 5. Task Conflict Handling

When multiple agents compete:

- prioritize earlier claim timestamp
- break ties using proximity
- fallback to WAIT or switch task

---

## 6. Task Lifecycle

1. Available
2. Claimed
3. In Progress
4. Completed / Failed

---

## 7. Dynamic Re-evaluation

Tasks are re-evaluated every frame:

- if new tasks appear
- if path cost changes
- if opponent interference increases

---

## 8. Design Principle

- Task selection is continuous, not one-shot
- Always adapt to updated frame state
