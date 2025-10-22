# app.py
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Page Configuration ---
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# --- Load Embeddings and Database (Cached for performance) ---
@st.cache_resource
def load_resources():
    """Load FAISS and embeddings for retrieval."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("new_book_db", embeddings, allow_dangerous_deserialization=True)
    # Fetch more chunks for deeper and richer answers
    return db.as_retriever(search_kwargs={"k": 10})

retriever = load_resources()

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")

    model_provider = st.selectbox(
        "Choose a Model Provider:",
        ("Google Gemini", "Hugging Face")
    )

    if model_provider == "Google Gemini":
        model_choice = st.selectbox(
            "Select Gemini Model:",
            (
                "models/gemini-2.5-pro",
                "models/gemini-2.5-flash",
                "models/gemini-2.5-flash-lite",
                "models/gemini-pro-latest",
                "models/gemini-flash-latest"
            )
        )
    else:
        model_choice = "google/flan-t5-xxl"  # larger and more detailed model

    api_key_input = st.text_input(
        "Enter Your API Key:",
        type="password",
        placeholder="Paste your key here"
    )

    # Optional: User can control response length
    response_length = st.slider(
        "Response Length (in words)",
        min_value=500,
        max_value=4000,
        value=2000,
        step=500,
        help="Higher values = longer and more detailed answers."
    )

    load_dotenv()
    if not api_key_input:
        if model_provider == "Google Gemini":
            api_key_input = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        else:
            api_key_input = os.getenv("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")

# --- Main App Interface ---
st.title("📚 RAG Chatbot (Deep Knowledge Mode)")
st.write("Ask any question about your documents — get a **detailed, structured, and well-explained answer.**")

# --- Initialize the Model ---
llm = None
if api_key_input:
    if model_provider == "Google Gemini":
        llm = ChatGoogleGenerativeAI(
            model=model_choice,
            google_api_key=api_key_input,
            temperature=0.8,
            max_output_tokens=response_length
        )
    else:
        llm = HuggingFaceEndpoint(
            repo_id=model_choice,
            task="text2text-generation",
            temperature=0.8,
            max_new_tokens=response_length,
            huggingfacehub_api_token=api_key_input
        )

# --- Chat Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not llm:
            answer = "⚠️ Please enter your API key in the sidebar first."
            st.warning(answer)
        else:
            with st.spinner("🧠 Generating a detailed and structured answer..."):
                try:
                    # --- Enhanced Prompt for Depth and Reasoning ---
                    prompt_template = ChatPromptTemplate.from_template("""
You are an expert assistant specialized in providing **thorough, step-by-step, and well-reasoned answers**.

Use the context below carefully to construct your response.
Include:
1. Step-by-step reasoning
2. Clear explanations
3. Examples if helpful
4. Final summary or key takeaways

<context>
{context}
</context>

Question: {input}

Answer in detail:
""")
                    document_chain = create_stuff_documents_chain(llm, prompt_template)
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)

                    response = retrieval_chain.invoke({"input": prompt})
                    answer = response.get("answer", "🤔 I couldn't find an answer.")
                    st.markdown(answer)

                    # --- Sources ---
                    with st.expander("📖 Sources Used"):
                        for i, doc in enumerate(response.get("context", []), 1):
                            st.write(f"**Source {i}:**\n{doc.page_content}")

                except Exception as e:
                    answer = f"❌ An error occurred: {e}"
                    st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
