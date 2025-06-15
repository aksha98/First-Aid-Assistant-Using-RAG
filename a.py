from rag_utils import load_documents, split_documents, add_to_chroma
from langchain_chroma import Chroma
from embedding import get_embedding_function

docs = load_documents()
print(f"Loaded {len(docs)} documents.")

#Check document content
for i, doc in enumerate(docs):
    print(f"\nDocument {i+1} content preview:\n{doc.page_content[:300]}")
chunks = split_documents(docs)
print(f"\nSplit into {len(chunks)} chunks.")
num_added = add_to_chroma(chunks)
print(f"\nAdded {num_added} new chunks to Chroma.")
db = Chroma(persist_directory="chroma", embedding_function=get_embedding_function())
ids = db.get(include=[])["ids"]
print(f"\nTotal chunks stored in Chroma: {len(ids)}")
