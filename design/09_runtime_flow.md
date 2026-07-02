# Runtime Flow Design

## 1. Overview

This file defines the complete execution loop of the AI agent per frame.

---

## 2. Main Loop (Per Frame)

```
for frame in range(1, 601):
    state = parse_protocol_input()
    world = update_world_model(state)
    candidates = generate_actions(world)
    scored = score_actions(candidates)
    action = select_best(scored)
    output_action(action)
```

---

## 3. Execution Stages

### 3.1 Input Stage
- receive protocol JSON
- decode state

### 3.2 State Update
- update graph
- update resources
- update tasks

### 3.3 Planning Stage
- compute paths
- evaluate tasks

### 3.4 Strategy Stage
- opponent prediction
- risk adjustment

### 3.5 Action Stage
- validate action
- serialize output

---

## 4. Failure Handling

If any stage fails:

- fallback to last valid action
- else WAIT

---

## 5. Performance Considerations

- must complete within frame time limit
- avoid recomputation of full graph when possible
- cache shortest paths when stable

---

## 6. Design Principle

- strict pipeline execution
- no cross-layer coupling
- deterministic per frame output
