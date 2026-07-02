# Design Overview

## 1. System Type
This project is a multi-agent, frame-based competitive control system derived from a graph-structured logistics combat simulation.

## 2. Core Paradigm
- Graph-based environment (S01–S15 nodes)
- Frame-based decision loop (600 frames per match)
- Partial observability via public state
- Adversarial multi-agent interaction

## 3. System Architecture Layers

### 3.1 State Layer
Responsible for parsing and maintaining world state:
- Position of convoy
- Resource inventory (good/bad fruits, freshness)
- Task list (royal tasks / reward tasks)
- Map connectivity and dynamic events

### 3.2 World Model Layer
Builds structured internal representation:
- Graph model of map
- Cost-weighted edges (weather, terrain, congestion)
- Task nodes with rewards and deadlines

### 3.3 Planning Layer
Core decision engine:
- Path planning (Dijkstra / A*)
- Task selection optimization
- Resource allocation strategy
- Risk evaluation under adversarial interference

### 3.4 Strategy Layer
High-level intelligence:
- Opponent trajectory prediction
- Interception and blocking decisions
- Window competition strategy
- Endgame acceleration logic

### 3.5 Action Layer
Converts decisions into protocol-compliant actions:
- MOVE
- CLAIM_TASK
- USE_RESOURCE
- DEPLOY_UNIT
- WAIT

## 4. Execution Loop (Per Frame)
1. Parse public state
2. Update world model
3. Generate candidate actions
4. Score actions via utility function
5. Select best action
6. Output action via communication protocol

## 5. Key Design Principles
- Deterministic execution under uncertainty
- Modular separation of planning and execution
- Graph-centric optimization
- Frame-consistent decision making
