# Protocol Adapter Design

## 1. Overview

The Protocol Adapter is the ONLY module that interacts with the external communication protocol defined in docs/communication.

It translates internal actions into protocol-compliant messages.

---

## 2. Input / Output

### Input:
- StructuredAction (from Action System)
- FrameState

### Output:
- Protocol JSON message

---

## 3. Responsibilities

- serialize actions
- validate schema compliance
- ensure field correctness
- handle fallback (WAIT)

---

## 4. Mapping Rules

| Internal Action | Protocol Command |
|----------------|-----------------|
| MOVE           | move_command    |
| CLAIM_TASK     | claim_task      |
| USE_RESOURCE   | use_resource    |
| DEPLOY_UNIT    | deploy_unit     |
| WAIT           | wait            |

---

## 5. Validation Pipeline

1. check required fields
2. check enum constraints
3. check frame legality
4. check resource validity
5. fallback to WAIT if invalid

---

## 6. Error Handling

If protocol rejection occurs:

- log error
- revert to last valid action
- default WAIT if unrecoverable

---

## 7. Design Principle

- strict separation from strategy logic
- stateless conversion layer
- fully deterministic serialization
