# rag_engine.py

import json
import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ==================================================
# 1. CONVERSATIONAL Q&A PIPELINE (EXISTING)
# ==================================================

SYSTEM_PROMPT = """
You are ConsultIQ, a helpful enterprise document analysis assistant.

Answer the user's question ONLY using the information provided in the
retrieved context.

Rules:
1. Do not use outside knowledge.
2. Do not make up or assume information.
3. If the answer cannot be found in the retrieved context, say:
   "I cannot find this information in the provided document."
4. When possible, mention the relevant page number from the source.
5. Give a concise and factual answer.

Retrieved Context:
{context}

Question:
{question}
"""


def format_docs(docs):
    """
    Convert retrieved LangChain documents into text
    that can be passed to the LLM.
    """

    if not docs:
        return "No relevant information was found in the document."

    formatted_docs = []

    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 0) + 1

        formatted_docs.append(
            f"[Source: {source} | Page: {page}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(formatted_docs)


def build_rag_chain(vectorstore):
    """
    Create the retriever and RAG chain.
    """

    # Retrieve only chunks that meet the similarity threshold
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.35
        }
    )

    # Create the prompt
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

    # Local Ollama LLM
    llm = ChatOllama(
        model="llama3.2",
        temperature=0.0
    )

    # Build RAG pipeline
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


# ==================================================
# 2. GENERALIZED FINANCIAL EXTRACTION PIPELINE (NEW)
# ==================================================

GENERALIZED_FINANCIAL_PROMPT = """You are a Financial Analyst AI.
Analyze the provided financial context and extract all available chronological reporting periods (years, quarters, or fiscal periods) along with their corresponding metrics: Revenue and Net Profit (and Operating Profit if available).

Rules:
1. Output ONLY a valid JSON list of objects sorted chronologically from oldest to newest.
2. If any metric is not available or cannot be determined, use null.
3. Values must be clean numeric floats (convert millions/billions to standard numeric format, or keep raw float units consistently).
4. Do not include markdown code block formatting (no ```), conversational text, or explanations.

Target JSON Schema:
[
  {{"Period": "FY2022", "Revenue": 1200.5, "Net Profit": 150.2, "Operating Profit": 200.0}},
  {{"Period": "FY2023", "Revenue": 1450.0, "Net Profit": 180.0, "Operating Profit": 240.5}}
]

Context:
{context}
"""

def extract_generalized_financials(vectorstore):
    """
    Dynamically extracts all financial periods present in the document.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(
        "statement of profit and loss income statement revenue from operations net profit operating income financial highlights"
    )
    context_text = format_docs(docs)

    prompt = ChatPromptTemplate.from_template(GENERALIZED_FINANCIAL_PROMPT)
    llm = ChatOllama(model="llama3.2", temperature=0.0)
    chain = prompt | llm | StrOutputParser()

    raw_response = chain.invoke({"context": context_text})

    try:
        cleaned = re.sub(r"```(?:json)?", "", raw_response).strip()
        data = json.loads(cleaned)
        
        if isinstance(data, list) and len(data) > 0:
            return data, docs
    except Exception:
        pass

    return [], docs