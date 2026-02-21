import os
from dotenv import load_dotenv
from groq import Groq
from datetime import datetime

from rag.retriever import get_retriever

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# RAG Tool
def answer_from_document(query: str) -> str:
    retriever = get_retriever()
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are answering using information from a PDF document.

    Rules:
    - Use ONLY the provided context.
    - Do NOT add assumptions.
    - Do NOT expand beyond the context.
    - If the answer is not clearly found, say:
    "I could not find this information in the document."
    - Keep the answer concise and factual.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content


# Calculator Tool
def calculator(expression: str) -> str:
    try:
        result = eval(expression)
        return str(result)
    except Exception:
        return "Invalid mathematical expression."


# Time Tool
def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")