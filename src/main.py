import sys
import os

# ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.agent.baseline_agent import BaselineAgent
from src.protocol.adapter import ProtocolAdapter
from src.engine.world_model import WorldModel


def main():
    world = WorldModel()
    agent = BaselineAgent(world)
    adapter = ProtocolAdapter()

    while True:
        # 1. read input state (from platform)
        raw_state = input()

        # 2. parse into world state
        state = adapter.parse_state(raw_state)
        world.update(state)

        # 3. decision making
        action = agent.act(world)

        # 4. output protocol action
        output = adapter.to_protocol(action)
        print(output)


if __name__ == "__main__":
    main()
