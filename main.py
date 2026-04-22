from langchain_core.messages import HumanMessage
from agent.graph import agent_graph
from agent.state import AgentState

def run_agent():
    print("🤖 AutoStream Assistant — Type 'quit' to exit\n")

    state: AgentState = {
        "messages": [],
        "intent": None,
        "retrieved_context": None,
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False,
    }

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye! 👋")
            break

        if not user_input:
            continue

        # Add the new user message to state
        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]

        # Run through the graph
        state = agent_graph.invoke(state)

        # Print the last AI message
        last_ai = next(
            (m for m in reversed(state["messages"]) if hasattr(m, "type") and m.type == "ai"),
            None
        )
        if last_ai:
            print(f"\nAgent: {last_ai.content}\n")

        if state.get("lead_captured"):
            print("✅ Lead successfully captured. Ending session.")
            break


if __name__ == "__main__":
    run_agent()