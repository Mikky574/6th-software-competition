# Scoring Module

class Scorer:
    def score(self, action, state):
        """
        Basic utility scoring
        """
        score = 0

        if action["action"] == "CLAIM_TASK":
            task = self._find_task(state, action.get("task_id"))
            if task:
                score += task.reward

        if action["action"] == "MOVE":
            score -= 1  # travel cost penalty

        if action["action"] == "WAIT":
            score -= 0.5

        return score

    def _find_task(self, state, task_id):
        for t in state.tasks:
            if t.task_id == task_id:
                return t
        return None
