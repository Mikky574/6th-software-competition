# Protocol Adapter

import json

class ProtocolAdapter:
    def parse_state(self, raw):
        """
        Convert raw input into structured state
        """
        try:
            return json.loads(raw)
        except Exception:
            return None

    def to_protocol(self, action):
        """
        Convert internal action to protocol message
        """
        return json.dumps(action)
