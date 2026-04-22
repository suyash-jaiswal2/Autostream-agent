# AutoStream AI Agent 🤖

An AI-powered conversational agent for AutoStream, a fictional SaaS video editing platform. Built as part of the ServiceHive Inflx ML Intern Assignment.

---

## What It Does

- Greets users and answers product/pricing questions using RAG
- Detects high-intent users ready to sign up
- Collects lead details (name, email, platform) conversationally
- Captures the lead using a mock API tool only after all details are collected

---

## Project Structure
autostream-agent/
├── knowledge_base/
│   └── autostream_kb.md       # Product knowledge base
├── agent/
│   ├── state.py               # Shared LangGraph state
│   ├── nodes.py               # All graph node functions
│   ├── graph.py               # LangGraph graph definition
│   └── tools.py               # Mock lead capture tool
├── rag/
│   └── retriever.py           # FAISS vector store + retrieval
├── main.py                    # Entry point / chat loop
├── requirements.txt
└── .env                       # API keys (not committed)

---

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/autostream-agent.git
cd autostream-agent
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at [console.groq.com](https://console.groq.com)

### 5. Run the agent
```bash
python main.py
```

---

## Example Conversation

You: Hi there!
Agent: Hello! Welcome to AutoStream...
You: What are your pricing plans?
Agent: We offer two plans — Basic at $29/month (10 videos, 720p)
and Pro at $79/month (unlimited videos, 4K, AI captions)...
You: The Pro plan sounds great, I want to sign up for my YouTube channel
Agent: Great! Could you share your full name?
You: John Doe
Agent: Thanks John! What's your email address?
You: john@gmail.com
Agent: Which platform do you mainly create content on?
You: YouTube
Agent: 🎉 You're all set, John Doe! We've captured your details...

✅ Lead captured: John Doe, john@gmail.com, YouTube

---

## Architecture Explanation

This project uses **LangGraph** to implement a stateful, multi-node conversational agent. LangGraph was chosen over AutoGen because it provides explicit, deterministic control flow — critical for a lead capture workflow where the mock tool must not be triggered prematurely. Each conversation turn passes through a typed graph with clearly defined nodes and conditional routing, making the logic transparent and easy to debug.

State is managed using a typed `AgentState` TypedDict that is passed between all nodes. It holds the full message history (accumulated via LangGraph's built-in `add_messages` reducer), the classified intent, RAG-retrieved context, and individual lead fields (name, email, platform). A `lead_captured` boolean acts as a terminal condition. Since the entire state is passed through the graph on every turn, memory is naturally retained across 5–6 conversation turns without any external memory buffer.

The RAG pipeline uses a local Markdown knowledge base, split into chunks using `MarkdownTextSplitter`, embedded with `sentence-transformers/all-MiniLM-L6-v2`, and stored in a FAISS vector index. On each inquiry, the top 3 relevant chunks are retrieved and injected into the system prompt.

---

## WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp using the **WhatsApp Business API (Meta)**:

1. **Set up a webhook** using Flask or FastAPI that listens for POST requests from Meta's servers.
2. **Verify the webhook** by handling the GET verification request Meta sends on setup.
3. **Parse incoming messages** — extract the sender's phone number and message text from the JSON payload.
4. **Maintain session state** per user by storing each user's `AgentState` in a database like Redis, keyed by their phone number.
5. **Invoke the agent graph** with the user's message and their retrieved state.
6. **Send the response back** using Meta's `/messages` API endpoint with the agent's reply text.

This approach allows the same LangGraph agent to handle multi-turn conversations across multiple WhatsApp users simultaneously, with each user maintaining their own independent session state.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10 |
| Agent Framework | LangGraph |
| LLM | Llama 3.3 70B (via Groq) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| RAG | LangChain + local Markdown KB |

---

## Evaluation Criteria Addressed

- ✅ Agent reasoning & intent detection
- ✅ Correct use of RAG (FAISS + local KB)
- ✅ Clean state management (LangGraph TypedDict)
- ✅ Proper tool calling logic (no premature trigger)
- ✅ Code clarity & structure
- ✅ Real-world deployability (WhatsApp webhook design documented)