"""
Task Scheduler using simple heuristics
Simplified version for production
"""
from typing import Dict, List


class TaskScheduler:
    """Simple priority-based task scheduler"""

    def __init__(self) -> None:
        self.scheduled_tasks: List[Dict] = []

    def schedule(self, tasks: List[Dict]) -> List[Dict]:
        """
        Schedule tasks based on priority and deadline

        Args:
            tasks: List of task dicts with priority, deadline, effort

        Returns:
            List of scheduled tasks with timing
        """
        # Sort by priority (high to low) and deadline (soon to late)
        sorted_tasks = sorted(
            tasks, key=lambda t: (-t.get("priority", 0.5), t.get("deadline", 100))
        )

        scheduled: List[Dict] = []
        current_time = 0

        for task in sorted_tasks:
            effort = task.get("effort", 1)
            scheduled.append(
                {
                    "task_id": task.get("id", f"task_{len(scheduled)}"),
                    "name": task.get("name", "Unnamed"),
                    "start_time": current_time,
                    "end_time": current_time + effort,
                    "priority": task.get("priority", 0.5),
                }
            )
            current_time += effort

        return scheduled

    def get_completion_rate(self, tasks: List[Dict]) -> float:
        """Estimate completion rate"""
        if not tasks:
            return 0.0

        high_priority = sum(1 for t in tasks if t.get("priority", 0) > 0.7)
        return min(0.85 + (high_priority * 0.05), 0.95)


# Initialize global scheduler
task_scheduler = TaskScheduler()
