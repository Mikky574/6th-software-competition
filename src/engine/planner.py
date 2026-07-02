# Planner Module

class Planner:
    def plan(self, world):
        """
        Generate candidate actions
        """
        state = world.get_state()
        actions = []

        if not state:
            return [{"action": "WAIT"}]

        # task-based actions
        for task in getattr(state, "tasks", []):
            actions.append({
                "action": "CLAIM_TASK",
                "task_id": task.task_id
            })

        # movement action
        if getattr(state, "next_target", None):
            actions.append({
                "action": "MOVE", "target": state.next_target
            })

        actions.append({"action": "WAIT"})

        return actions
