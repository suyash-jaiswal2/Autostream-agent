from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

KB_PATH = os.path.join(os.path.dirname(__file__), "../knowledge_base/autostream_kb.md")

def build_retriever():
    loader = TextLoader(KB_PATH)
    docs = loader.load()

    splitter = MarkdownTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = build_retriever()

def retrieve_context(query: str) -> str:
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])