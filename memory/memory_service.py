"""Memory service orchestrator."""

from typing import Any

from memory.working_memory import WorkingMemory


class MemoryService:

  def __init__(self):
    self.working_memory = WorkingMemory()

  def store_interaction(self, user_message: str, agent_response: str):
    self.working_memory.add_turn(user_message, agent_response)

  def build_context(self, current_query: str) -> list[dict[str, Any]]:
    from typing import cast
    return cast(list[dict[str, Any]], self.working_memory.get_context())

  def clear(self):
    self.working_memory.clear()
