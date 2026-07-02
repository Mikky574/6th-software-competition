# Action System Design

## 1. Overview

The Action System defines how abstract decisions are converted into executable protocol commands.

It is the ONLY layer that interacts with the communication protocol.

---

## 2. Action Space Definition

Supported atomic actions:

### 2.1 Movement Actions
- MOVE(node_id)
- PATH_MOVE(route_id)

### 2.2 Task Actions
- CLAIM_TASK(task_id)
- ABANDON_TASK(task_id)

### 2.3 Resource Actions
- USE_RESOURCE(type, amount)
- EXCHANGE_RESOURCE(from, to)

### 2.4 Unit Actions
- DEPLOY_UNIT(unit_id, target)
- RECALL_UNIT(unit_id)

### 2.5 Idle Action
- WAIT

---

## 3. Action Constraints

Each action must satisfy:

- frame legality
- resource availability
- adjacency constraints (map graph)
- cooldown restrictions
- protocol schema compliance

---

## 4. Action Validation Pipeline

```
Candidate Action
   ↓
Constraint Checker
   ↓
Resource Validator
   ↓
Map Legality Validator
   ↓
Protocol Formatter
   ↓
Final Action Output
```

---

## 5. Action Scoring Interface

Each action is assigned a utility score:

U(action) = reward_gain - time_cost - risk_penalty

---

## 6. Conflict Resolution

If multiple actions conflict:

Priority order:
1. Task completion
2. Survival / resource preservation
3. Movement optimization
4. Exploration

---

## 7. Design Principle

- All actions are stateless objects
- Execution is deterministic
- Invalid actions default to WAIT
