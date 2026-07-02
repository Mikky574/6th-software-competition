# Architecture Design

## 1. System Overview

This system is a layered decision-making AI agent designed for a frame-based adversarial logistics simulation.

It replaces the legacy `src/core` logic with a modular architecture defined in the `design/` layer.

---

## 2. Core Modules

### 2.1 State Parser Layer
Responsible for converting raw protocol messages into structured internal state.

Inputs:
- Public map state
- Team state
- Task list
- Resource status

Outputs:
- StructuredState object

---

### 2.2 World Model Layer
Builds persistent simulation of environment.

Responsibilities:
- Graph construction (S01-S15)
- Edge weighting (terrain + weather + congestion)
- Task graph modeling

Data Structures:
- Graph<Node, Edge>
- TaskPool
- ResourceMap

---

### 2.3 Planning Layer (Core Intelligence)

This is the decision engine.

Functions:
- Shortest path computation (Dijkstra / A*)
- Multi-objective optimization (score vs time vs risk)
- Task scheduling

Output:
- Candidate Action Set

---

### 2.4 Strategy Layer
High-level decision policy:

- Opponent prediction model (trajectory inference)
- Interception planning
- Window competition strategy
- Endgame acceleration logic

---

### 2.5 Action Translator Layer
Converts abstract decisions into protocol commands:

Supported actions:
- MOVE
- CLAIM_TASK
- USE_RESOURCE
- DEPLOY_UNIT
- WAIT

Ensures compliance with communication protocol.

---

## 3. Data Flow

```
Protocol Input
   ↓
State Parser
   ↓
World Model Update
   ↓
Planning Engine
   ↓
Strategy Filter
   ↓
Action Translator
   ↓
Protocol Output
```

---

## 4. Replacement of legacy core/

- `src/core` is considered legacy logic
- All decision logic must be migrated to:
  - Planning Layer
  - Strategy Layer

No business logic should remain in core.

---

## 5. Design Principles

- Modular decomposition
- Stateless input parsing
- Deterministic decision pipeline
- Graph-centric reasoning
- Frame-consistent execution
