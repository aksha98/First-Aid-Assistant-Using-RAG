import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from embedding import get_embedding_function

from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

CHROMA_PATH = "chroma"
DATA_PATH = "data"
PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""
You are a helpful and precise first aid assistant. Use only the information from the context below to answer the question.Format the answer in a clear,understandable way.

If the answer is found in the context, explain it clearly in full sentences.
If not, say: "I don't know."

Context:
{context}

---

Question: {question}
""")
def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    return loader.load()

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80
    )
    return splitter.split_documents(documents)

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

    return chunks
def add_to_chroma(chunks):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_ids = set(db.get(include=[])["ids"])
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        db.add_documents(new_chunks, ids=[chunk.metadata["id"] for chunk in new_chunks])
        return len(new_chunks)

    return 0


def query_rag(query_text: str):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    results = db.similarity_search_with_score(query_text, k=5)

    context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt = PROMPT_TEMPLATE.format(context=context, question=query_text)

    model = OllamaLLM(model="mistral")
    response = model.invoke(prompt)

    sources = [doc.metadata.get("id", "unknown") for doc, _ in results]
    chunks_used = [doc.page_content for doc, _ in results]

    return response, sources, chunks_used
