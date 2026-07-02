class State:
    """Global game state (single source of truth)"""

    def __init__(self):
        self.round = 0
        self.players = {}
        self.nodes = {}
        self.tasks = {}
        self.events = {}
        self.history = []

    def update(self, inquire: dict):
        self.round += 1
        self.players = inquire.get("players", {})
        self.nodes = inquire.get("nodes", {})
        self.tasks = inquire.get("tasks", {})
        self.events = inquire.get("events", {})

        self.history.append({
            "round": self.round,
            "snapshot": inquire
        })
