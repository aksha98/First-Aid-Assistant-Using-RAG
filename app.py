import streamlit as st
from rag_utils import query_rag


st.set_page_config(
    page_title="First Aid Chatbot",
    layout="wide",
    
)


with st.sidebar:
    st.markdown("##  First Aid Assistant")
    st.markdown("A simple First Aid assistant that finds answers from given data using Retrieval-Augmented Generation (RAG).")
    st.markdown("---")
    st.markdown("**Usage tips:**")
    st.markdown("- Ask specific questions (e.g. _How to treat a burn?_)")
    st.markdown("- Expand 'Chunks Used' to see relevant chunks extracted.")
    st.markdown("- Built with local RAG and Ollama. ")

    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []

st.markdown("<h1 style='text-align: center;'> First Aid Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Assistant for Basic First Aid guidance.</p>", unsafe_allow_html=True)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Type your First Aid question here...")

if user_input:
    with st.spinner("Thinking..."):
        answer, sources, chunks_used = query_rag(user_input)
        st.session_state.chat_history.append({
            "user": user_input,
            "answer": answer,
            "sources": sources,
            "chunks": chunks_used
        })
for message in st.session_state.chat_history:
    with st.chat_message("user", avatar="🧑‍⚕️"):
        st.markdown(f"**You:** {message['user']}")

    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(f"**First Aid Bot:** {message['answer']}")

        with st.expander("📂 Sources"):
            if message["sources"]:
                st.write(message["sources"])
            else:
                st.write("No specific source ID available.")

        with st.expander("Chunks Used"):
            if message["chunks"]:
                st.write(f"**Number of relevant chunks:** {len(message['chunks'])}")
                for i, chunk in enumerate(message["chunks"], 1):
                    st.markdown(f"<div style='background-color:#f6f6f6; padding:10px; border-radius:5px;'><strong>Chunk {i}:</strong><br>{chunk}</div>", unsafe_allow_html=True)
            else:
                st.write("No relevant chunks found.")
