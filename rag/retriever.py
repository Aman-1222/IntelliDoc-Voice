from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

VECTORSTORE_DIR = "vectorstore"


def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    return retriever


if __name__ == "__main__":
    retriever = get_retriever()

    while True:
        query = input("\nAsk a question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        docs = retriever.invoke(query)

        print("\n--- Retrieved Chunks ---\n")
        for i, doc in enumerate(docs, 1):
            print(f"Chunk {i}:\n{doc.page_content}\n")