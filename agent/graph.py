from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import (
    classify_intent,
    retrieve_context,
    generate_response,
    collect_lead_info,
    parse_lead_fields,
    execute_lead_capture,
)


def route_after_intent(state: AgentState) -> str:
    intent = state.get("intent", "greeting")
    if intent == "high_intent":
        return "parse_lead_fields"
    elif intent == "inquiry":
        return "retrieve_context"
    else:
        return "generate_response"


def route_after_parse(state: AgentState) -> str:
    name = state.get("lead_name")
    email = state.get("lead_email")
    platform = state.get("lead_platform")

    if name and email and platform:
        return "execute_lead_capture"
    else:
        return "collect_lead_info"


def build_graph():
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("generate_response", generate_response)
    graph.add_node("parse_lead_fields", parse_lead_fields)
    graph.add_node("collect_lead_info", collect_lead_info)
    graph.add_node("execute_lead_capture", execute_lead_capture)

    # Entry point
    graph.set_entry_point("classify_intent")

    # Routing after intent classification
    graph.add_conditional_edges("classify_intent", route_after_intent, {
        "retrieve_context": "retrieve_context",
        "generate_response": "generate_response",
        "parse_lead_fields": "parse_lead_fields",
    })

    # After RAG → respond
    graph.add_edge("retrieve_context", "generate_response")
    graph.add_edge("generate_response", END)

    # Lead flow routing
    graph.add_conditional_edges("parse_lead_fields", route_after_parse, {
        "collect_lead_info": "collect_lead_info",
        "execute_lead_capture": "execute_lead_capture",
    })

    graph.add_edge("collect_lead_info", END)
    graph.add_edge("execute_lead_capture", END)

    return graph.compile()


agent_graph = build_graph()