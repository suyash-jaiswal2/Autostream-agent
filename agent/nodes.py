
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from agent.state import AgentState
from agent.tools import mock_lead_capture
from rag.retriever import retrieve_context as rag_retrieve

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)


# ── Node 1: Classify Intent ──────────────────────────────────────────────────
def classify_intent(state: AgentState) -> AgentState:
    last_message = state["messages"][-1].content

    prompt = f"""Classify the intent of this user message into exactly one of:
- greeting       → casual hello, how are you, etc.
- inquiry        → asking about product, pricing, features, policies
- high_intent    → ready to sign up, wants to try, buy, or start a plan

User message: "{last_message}"

Reply with only one word: greeting, inquiry, or high_intent"""

    response = llm.invoke([HumanMessage(content=prompt)])
    intent = response.content.strip().lower()

    if intent not in ["greeting", "inquiry", "high_intent"]:
        intent = "inquiry"  # safe fallback

    return {**state, "intent": intent}


# ── Node 2: Retrieve RAG Context ─────────────────────────────────────────────
def retrieve_context(state: AgentState) -> AgentState:
    last_message = state["messages"][-1].content
    context = rag_retrieve(last_message)
    return {**state, "retrieved_context": context}


# ── Node 3: Generate Response ─────────────────────────────────────────────────
def generate_response(state: AgentState) -> AgentState:
    context = state.get("retrieved_context", "")
    intent = state.get("intent", "greeting")

    system_prompt = f"""You are a sales assistant ONLY for AutoStream, an AI video editing SaaS platform.

STRICT RULES:
- ONLY talk about AutoStream's product, pricing, and policies
- NEVER ask about cars, personal life, or anything unrelated
- If asked something unrelated, politely redirect back to AutoStream
- Keep responses short and focused

{"Use this knowledge base to answer accurately:" + chr(10) + context if context else ""}

If the user seems interested in signing up, guide them toward sharing their name, email, and creator platform (YouTube, Instagram, etc.)."""

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)

    return {**state, "messages": state["messages"] + [AIMessage(content=response.content)]}


# ── Node 4: Collect Lead Info ─────────────────────────────────────────────────
def collect_lead_info(state: AgentState) -> AgentState:
    name = state.get("lead_name")
    email = state.get("lead_email")
    platform = state.get("lead_platform")

    # Figure out what's still missing
    if not name:
        question = "I'd love to get you started with AutoStream! Could you share your full name?"
    elif not email:
        question = f"Thanks {name}! What's your email address so we can set up your account?"
    elif not platform:
        question = f"Almost done! Which content platform do you mainly use — YouTube, Instagram, TikTok, or another?"
    else:
        # All collected — this node shouldn't be called, but handle gracefully
        question = "Let me finalize your registration now!"

    return {**state, "messages": state["messages"] + [AIMessage(content=question)]}


# ── Node 5: Parse Lead Fields from Latest User Message ───────────────────────
def parse_lead_fields(state: AgentState) -> AgentState:
    last_message = state["messages"][-1].content
    name = state.get("lead_name")
    email = state.get("lead_email")
    platform = state.get("lead_platform")

    prompt = f"""Extract any of the following from this message if present:
- Full name (a person's name)
- Email address (contains @)
- Creator platform (YouTube, Instagram, TikTok, Twitter, Facebook, etc.)

Message: "{last_message}"

Reply ONLY in this format (use null if not found):
name: <value or null>
email: <value or null>
platform: <value or null>"""

    response = llm.invoke([HumanMessage(content=prompt)])
    lines = response.content.strip().split("\n")

    parsed = {}
    for line in lines:
        if ":" in line:
            key, val = line.split(":", 1)
            parsed[key.strip()] = val.strip()

    def clean(val):
        return None if val in (None, "null", "None", "") else val

    return {
        **state,
        "lead_name": name or clean(parsed.get("name")),
        "lead_email": email or clean(parsed.get("email")),
        "lead_platform": platform or clean(parsed.get("platform")),
    }


# ── Node 6: Execute Lead Capture Tool ────────────────────────────────────────
def execute_lead_capture(state: AgentState) -> AgentState:
    result = mock_lead_capture(
        name=state["lead_name"],
        email=state["lead_email"],
        platform=state["lead_platform"]
    )

    confirmation = (
        f"🎉 You're all set, {state['lead_name']}! "
        f"We've captured your details and our team will reach out to {state['lead_email']} shortly. "
        f"Welcome to AutoStream — can't wait to see your {state['lead_platform']} content shine! 🚀"
    )

    return {
        **state,
        "lead_captured": True,
        "messages": state["messages"] + [AIMessage(content=confirmation)]
    }