"""
Simple Memory System
Stores and retrieves user memories
"""
from datetime import datetime
from typing import Dict, List


class MemorySystem:
    """In-memory storage for user memories"""

    def __init__(self) -> None:
        self.memories: List[Dict] = []

    def add_memory(self, content: str, context: Dict) -> str:
        """Add a new memory"""
        timestamp = datetime.now().timestamp()
        memory_id = f"mem_{len(self.memories)}_{timestamp}"
        memory = {
            "id": memory_id,
            "content": content,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "importance": self._calculate_importance(content, context),
        }
        self.memories.append(memory)
        return memory_id

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant memories"""
        # Simple retrieval - return most recent
        return sorted(self.memories, key=lambda m: m["timestamp"], reverse=True)[:k]

    def _calculate_importance(self, content: str, context: Dict) -> float:
        """Calculate memory importance"""
        base_importance = 0.5

        # Boost importance for certain contexts
        if context.get("user_initiated"):
            base_importance += 0.2
        if context.get("task_completed"):
            base_importance += 0.15

        return min(base_importance, 1.0)


# Initialize global memory system
memory_system = MemorySystem()
