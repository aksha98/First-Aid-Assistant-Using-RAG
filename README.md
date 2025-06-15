# First-Aid-Assistant-Using-RAG

A smart assistant that helps users find accurate and quick responses to common first aid questions by combining AI and document search.




## What is RAG?

**RAG** stands for **Retrieval-Augmented Generation**. It's a powerful technique that combines two steps:

1. **Retrieval:** The system first searches a local database of text (built from your own PDF documents) to find the most relevant chunks of information.  
2. **Generation:** Then it passes those chunks to a language model (LLM) to generate a helpful, accurate, and grounded response based only on the retrieved information.

This prevents hallucinations and ensures that answers are based only on the knowledge you provide, making it perfect for safety-critical domains like first aid.



## About This Project

This chatbot is a basic AI-powered assistant that uses RAG to answer questions about first aid. You provide it with PDFs containing first aid information, and it becomes a smart helper that can respond to questions like:

- "How to treat a burn?"  
- "What to do in case of a nosebleed?"  
- "What are the steps of CPR?"  

It’s built using:

- **Streamlit** for a friendly web interface. 
- **LangChain** for document handling and RAG pipeline.  
- **ChromaDB** for local vector storage. 
- **Ollama** to run a local language model (like Mistral).  



## Setup Instructions

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/rag_chatbot.git
    ```

2. **Change to the project directory:**
    ```bash
    cd rag_chatbot
    ```

3. **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    Make sure you have **Ollama** installed and the following models available:

- `mistral` – for answering questions  
- `nomic-embed-text` – for embeddings


4. **Use the sample PDFs:**  
   The `data/` folder already contains two sample first aid PDF files used for testing and demonstration.  
   You can also add your own PDFs to this folder to customize the assistant.

5. **Embed the documents:**
    ```bash
    python a.py
    ```
    This script loads your PDFs, splits them into smaller chunks, embeds them using Ollama, and stores them in a vector database (Chroma).

6. **Launch the chatbot:**
    ```bash
    streamlit run app.py
    ```
    **Note**: You can modify `chunk_size` and `chunk_overlap` in the `split_documents()` function inside `rag_utils.py` for better accuracy or testing.


## Project Structure
```
├── app.py              # Streamlit app frontend
├── a.py                # PDF loader + chunker + vector store updater
├── rag_utils.py        # All RAG logic (retrieval + generation)
├── embedding.py        # Defines embedding function using Ollama
├── requirements.txt    # Python package dependencies
├── data/               # Includes two sample PDF files for testing
├── chroma/             # Local vector store folder (auto-created, not tracked)
```

## Streamlit Output
<img width="948" alt="image" src="https://github.com/user-attachments/assets/651413d4-1afb-43ba-a0c4-a228b38ba7ec" />








