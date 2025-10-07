# app.py
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
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
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("new_book_db", embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 5})

retriever = load_resources()

# --- Sidebar for Configuration ---
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
                "models/gemini-2.5-flash",
                "models/gemini-2.5-pro",
                "models/gemini-2.5-flash-lite",
                "models/gemini-pro-latest",
                "models/gemini-flash-latest"
            )
        )
    else:
        model_choice = "google/flan-t5-large"

    api_key_input = st.text_input(
        "Enter Your API Key:",
        type="password",
        placeholder="Paste your key here"
    )

    load_dotenv()
    if not api_key_input:
        if model_provider == "Google Gemini":
            api_key_input = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        else:
            api_key_input = os.getenv("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")

# --- Main App Logic ---
st.title("📚 RAG Chatbot")
st.write("Ask a question about your documents.")

llm = None
if api_key_input:
    if model_provider == "Google Gemini":
        llm = ChatGoogleGenerativeAI(model=model_choice, google_api_key=api_key_input)
    else:
        llm = HuggingFaceEndpoint(
            repo_id=model_choice,
            task="text2text-generation",
            temperature=0.2,
            max_new_tokens=1024,
            huggingfacehub_api_token=api_key_input
        )

# Main chat interface
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
            answer = "⚠️ Please enter your API key in the sidebar to begin."
            st.warning(answer)
        else:
            with st.spinner("Thinking..."):
                try:
                    # Prompt templates
                    if model_provider == "Google Gemini":
                        prompt_template = ChatPromptTemplate.from_template("""
Answer based only on this context:
<context>{context}</context>
Question: {input}""")
                    else:
                        prompt_template = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context:
Context: {context}
Question: {input}""")

                    document_chain = create_stuff_documents_chain(llm, prompt_template)
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)

                    response = retrieval_chain.invoke({"input": prompt})
                    answer = response.get("answer", "🤔 I couldn't find an answer.")
                    st.markdown(answer)

                    with st.expander("📖 Show Sources"):
                        for i, doc in enumerate(response.get("context", []), 1):
                            st.write(f"**Source {i}:**\n{doc.page_content}")

                except Exception as e:
                    answer = f"❌ An error occurred: {e}"
                    st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
