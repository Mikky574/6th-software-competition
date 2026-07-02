from abc import ABC, abstractmethod
from typing import Any, Dict, List

class IParser(ABC):
    @abstractmethod
    def parse_inquire(self, raw: Dict[str, Any]) -> Any:
        pass

class IState(ABC):
    @abstractmethod
    def update(self, inquire: Dict[str, Any]) -> None:
        pass

class IWorldModel(ABC):
    @abstractmethod
    def build_graph(self, state: Any) -> None:
        pass

    @abstractmethod
    def shortest_path(self, start: Any, end: Any) -> List[Any]:
        pass

    @abstractmethod
    def cost(self, a: Any, b: Any, state: Any) -> float:
        pass

class IStrategy(ABC):
    @abstractmethod
    def evaluate(self, state: Any, world: Any) -> List[Dict[str, Any]]:
        pass

class IDecisionEngine(ABC):
    @abstractmethod
    def decide(self, state: Any, world: Any) -> Dict[str, Any]:
        pass

class IActionBuilder(ABC):
    @abstractmethod
    def build(self, action: Dict[str, Any]) -> Dict[str, Any]:
        pass