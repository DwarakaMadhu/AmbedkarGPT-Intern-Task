## main.py
"""
AmbedkarGPT Intern Task (Fixed & Improved Version)
RAG pipeline using LangChain + Chroma + HF Embeddings + Ollama (llama3.2)
"""

import os
import argparse
import chromadb

# Disable noisy telemetry warnings
chromadb.config.settings = chromadb.config.Settings(anonymized_telemetry=False)

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA


# ------------------------------
# CONFIG
# ------------------------------
CHROMA_DIR = "chroma_db"
TEXT_FILE = "speech.txt"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30
OLLAMA_MODEL = "llama3.2"   # <<=== your model
# ------------------------------


def build_vectorstore(text_path=TEXT_FILE, persist_directory=CHROMA_DIR):
    print("[INFO] Loading text...")
    loader = TextLoader(text_path, encoding="utf-8")
    docs = loader.load()

    print("[INFO] Splitting text...")
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        keep_separator=False
    )
    docs_split = splitter.split_documents(docs)

    print(f"[INFO] Created {len(docs_split)} chunks")

    print("[INFO] Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print("[INFO] Building Chroma vectorstore...")
    vectordb = Chroma.from_documents(
        documents=docs_split,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    vectordb.persist()
    print("[INFO] Vectorstore saved to", persist_directory)
    return vectordb


def load_vectorstore(persist_directory=CHROMA_DIR):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vectordb


def build_qa_chain(vectordb, k=3):
    print(f"[INFO] Using Ollama model: {OLLAMA_MODEL}")

    llm = Ollama(model=OLLAMA_MODEL)

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    return qa_chain


def interactive_loop(qa_chain):
    print("AmbedkarGPT — ask a question about the speech (type 'exit' to quit)")
    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() in ["exit", "quit"]:
            break
        if not q:
            continue

        try:
            answer = qa_chain.run(q)
            print("\nAnswer:\n", answer)
        except Exception as e:
            print("Error generating answer:", e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Rebuild vectorstore from speech.txt")
    args = parser.parse_args()

    if args.rebuild or not os.path.exists(CHROMA_DIR):
        print("[INFO] Building vectorstore from text...")
        vectordb = build_vectorstore()
    else:
        print("[INFO] Loading existing vectorstore...")
        vectordb = load_vectorstore()

    qa_chain = build_qa_chain(vectordb)
    interactive_loop(qa_chain)


if __name__ == "__main__":
    main()
