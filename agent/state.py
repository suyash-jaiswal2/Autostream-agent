from typing import TypedDict, Optional
from langgraph.graph.message import add_messages
from typing import Annotated

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # full conversation history
    intent: Optional[str]                      # greeting | inquiry | high_intent
    retrieved_context: Optional[str]           # RAG context from knowledge base
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_captured: bool