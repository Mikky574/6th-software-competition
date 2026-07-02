# Baseline Agent

class BaselineAgent:
    def __init__(self, world):
        self.world = world

    def act(self, world):
        """
        Main decision function
        """
        state = world.get_state()

        # simple heuristic baseline
        if state is None:
            return {"action": "WAIT"}

        # 1. try complete task if available
        if state.tasks:
            best_task = max(state.tasks, key=lambda t: t.reward)
            return {
                "action": "CLAIM_TASK",
                "task_id": best_task.task_id
            }

        # 2. move toward next node (greedy)
        if state.next_target:
            return {
                "action": "MOVE",
                "target": state.next_target
            }

        # 3. fallback
        return {"action": "WAIT"}
