from rag_utils import load_documents, split_documents, add_to_chroma
from langchain_chroma import Chroma
from embedding import get_embedding_function

# Step 1: Load PDF documents
docs = load_documents()
print(f"Loaded {len(docs)} documents.")

# Step 2: Check document content
for i, doc in enumerate(docs):
    print(f"\nDocument {i+1} content preview:\n{doc.page_content[:300]}")

# Step 3: Split into chunks
chunks = split_documents(docs)
print(f"\nSplit into {len(chunks)} chunks.")

# Step 4: Add chunks to Chroma
num_added = add_to_chroma(chunks)
print(f"\nAdded {num_added} new chunks to Chroma.")

# Step 5: Print current Chroma vector store IDs
db = Chroma(persist_directory="chroma", embedding_function=get_embedding_function())
ids = db.get(include=[])["ids"]
print(f"\nTotal chunks stored in Chroma: {len(ids)}")
