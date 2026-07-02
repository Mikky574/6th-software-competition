# Scoring Model Design

## 1. Overview

The scoring model evaluates all candidate actions and decisions.

It is the core of the utility-based decision system.

---

## 2. Utility Function

Each action is assigned a score:

U(action) =
+ reward_gain
- time_cost
- resource_cost
- risk_penalty
+ strategic_value

---

## 3. Reward Components

- task completion reward
- delivery bonus
- early arrival bonus

---

## 4. Cost Components

- travel time
- resource consumption
- opportunity cost

---

## 5. Risk Model

Risk includes:

- opponent interception probability
- congestion risk
- resource depletion risk

---

## 6. Strategic Value

Long-term benefits:

- controlling key nodes
- securing future tasks
- blocking opponent routes

---

## 7. Action Ranking

All candidate actions are sorted by U(action).

Top-K selection is used when needed.

---

## 8. Normalization

All components are normalized to [0,1].

---

## 9. Design Principle

- utility-driven decision making
- frame-consistent scoring
- extensible weighting system
