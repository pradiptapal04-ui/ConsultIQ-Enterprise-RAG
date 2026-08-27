import os
import tempfile
import pandas as pd
import streamlit as st

from document_loader import load_document
from text_chunker import split_documents
from vector_store import create_vector_store
from rag_engine import build_rag_chain, extract_generalized_financials
from bianalysis import analyze_generalized_financials


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="📊 ConsultIQ",
    page_icon="📊",
    layout="centered"
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("📊 ConsultIQ")

st.subheader(
    "GenAI-Powered Business Intelligence & Document Analysis Assistant"
)

st.write(
    "Upload a business document, ask questions about it, "
    "and analyze key financial metrics automatically."
)


# --------------------------------------------------
# DOCUMENT UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload an annual report",
    type=["pdf", "txt"]
)


if uploaded_file is not None:

    # --------------------------------------------------
    # SAVE UPLOADED FILE TEMPORARILY
    # --------------------------------------------------
    file_ext = uploaded_file.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f".{file_ext}"
    ) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # --------------------------------------------------
    # PROCESS DOCUMENT & BUILD LOCAL INDEX
    # --------------------------------------------------
    if (
        "current_file" not in st.session_state
        or st.session_state["current_file"] != uploaded_file.name
    ):
        with st.spinner(
            "Processing document, creating chunks and building vector index..."
        ):
            # Load document
            docs = load_document(tmp_path)

            # Add source metadata
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name

            # Extracted full text preview
            full_text = "\n\n".join([doc.page_content for doc in docs])
            st.session_state["extracted_text"] = full_text

            # Text Chunking
            chunks = split_documents(docs)
            st.session_state["chunks"] = chunks

            # Create local FAISS vector store
            vectorstore = create_vector_store(chunks)
            st.session_state["vectorstore"] = vectorstore

            # Reset session state for new document
            st.session_state["current_file"] = uploaded_file.name
            st.session_state["messages"] = []
            st.session_state["financial_data"] = None
            st.session_state["fin_sources"] = []

        st.success("✅ Document processed successfully!")
    else:
        st.success("✅ Document uploaded successfully!")

    # --------------------------------------------------
    # DELETE TEMPORARY FILE
    # --------------------------------------------------
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    # --------------------------------------------------
    # DOCUMENT STATISTICS
    # --------------------------------------------------
    st.markdown("### 📄 Document Information")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Characters Extracted",
            f"{len(st.session_state.get('extracted_text', '')):,}"
        )
    with col2:
        st.metric(
            "Text Chunks",
            f"{len(st.session_state.get('chunks', [])):,}"
        )

    # --------------------------------------------------
    # EXTRACTED TEXT PREVIEW
    # --------------------------------------------------
    st.markdown("### Extracted Text")
    st.caption("Preview of the document content")

    st.text_area(
        label="",
        value=st.session_state.get("extracted_text", ""),
        height=220,
        disabled=True
    )

    st.divider()

    # ==================================================
    # RAG QUESTION & ANSWER SECTION
    # ==================================================
    st.markdown("## 🤖 Ask ConsultIQ")
    st.write("Ask questions about the uploaded document.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_query = st.chat_input("Ask a question about the uploaded document...")

    if user_query:
        st.session_state.messages.append(
            {"role": "user", "content": user_query}
        )

        with st.chat_message("user"):
            st.markdown(user_query)

        if "vectorstore" in st.session_state:
            try:
                chain, retriever = build_rag_chain(
                    st.session_state["vectorstore"]
                )

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing document..."):
                        response = chain.invoke(user_query)
                    st.markdown(response)

                    # Retrieved context expander
                    with st.expander("🔍 View Retrieved Context / Sources"):
                        sources = retriever.invoke(user_query)
                        if sources:
                            for i, doc in enumerate(sources):
                                page_num = doc.metadata.get("page", 0) + 1
                                st.markdown(
                                    f"**Chunk {i + 1} (Page {page_num})**"
                                )
                                st.caption(doc.page_content)
                                st.divider()
                        else:
                            st.info(
                                "No sufficiently relevant information was retrieved from the document."
                            )

                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

            except Exception as e:
                st.error(
                    f"An error occurred while generating the response: {e}"
                )

    st.divider()

    # ==================================================
    # DYNAMIC BUSINESS INTELLIGENCE DASHBOARD
    # ==================================================
    st.markdown("## 📈 Business Intelligence Dashboard")
    st.write(
        "Extract chronological financial metrics automatically from the document to compute KPIs and trends."
    )

    if st.button("📊 Auto-Extract & Analyze Financial Performance", type="primary"):
        with st.spinner("Retrieving financial records and parsing structured data via Llama 3.2..."):
            extracted_data, source_chunks = extract_generalized_financials(
                st.session_state["vectorstore"]
            )
            st.session_state["financial_data"] = extracted_data
            st.session_state["fin_sources"] = source_chunks

    if st.session_state.get("financial_data"):
        results = analyze_generalized_financials(st.session_state["financial_data"])

        if results is not None:
            df = results["df"]

            # --------------------------------------------------
            # DYNAMIC KPIs
            # --------------------------------------------------
            st.markdown("### 📌 Key Performance Indicators")

            if results["total_periods"] > 1:
                k1, k2, k3 = st.columns(3)
                with k1:
                    st.metric(
                        label=f"Revenue Growth ({results['prev_period']} → {results['latest_period']})",
                        value=f"{results['latest_rev_growth']:.2f}%"
                    )
                with k2:
                    st.metric(
                        label=f"Profit Growth ({results['prev_period']} → {results['latest_period']})",
                        value=f"{results['latest_profit_growth']:.2f}%"
                    )
                with k3:
                    latest_margin = df.iloc[-1].get("Net Margin (%)", 0.0)
                    st.metric(
                        label=f"Latest Net Margin ({results['latest_period']})",
                        value=f"{latest_margin:.2f}%"
                    )
            else:
                st.info(
                    f"Only one financial period detected ({results['latest_period']}). Multiple reporting periods are needed to compute growth rates."
                )

            # --------------------------------------------------
            # FINANCIAL DATA TABLE & EXPORT
            # --------------------------------------------------
            st.markdown("### 📋 Financial Data (Pandas)")
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Financial Summary (CSV)",
                data=csv,
                file_name=f"ConsultIQ_{st.session_state['current_file']}_Analysis.csv",
                mime="text/csv",
            )

            # --------------------------------------------------
            # DYNAMIC TREND CHARTS
            # --------------------------------------------------
            if "Period" in df.columns:
                st.markdown("### 📊 Performance Trends")
                chart_df = df.set_index("Period")

                c1, c2 = st.columns(2)
                if "Revenue" in chart_df.columns:
                    with c1:
                        st.markdown("**Revenue Trend**")
                        st.line_chart(chart_df[["Revenue"]])
                if "Net Profit" in chart_df.columns:
                    with c2:
                        st.markdown("**Net Profit Trend**")
                        st.line_chart(chart_df[["Net Profit"]])

            # --------------------------------------------------
            # AUTOMATED BUSINESS INTERPRETATION
            # --------------------------------------------------
            if results["total_periods"] > 1:
                st.markdown("### 💡 Business Interpretation")
                r_growth = results["latest_rev_growth"]
                p_growth = results["latest_profit_growth"]

                if r_growth > 0 and p_growth > r_growth:
                    st.success(
                        f"📌 **Operating Leverage:** Net profit is expanding faster ({p_growth:.2f}%) than top-line revenue ({r_growth:.2f}%) into {results['latest_period']}, indicating disciplined cost control and margin expansion."
                    )
                elif r_growth > 0 and p_growth < r_growth:
                    st.info(
                        f"📌 **Margin Pressure:** Revenue grew by {r_growth:.2f}%, but net profit growth lagged at {p_growth:.2f}%, suggesting rising operating costs or overhead."
                    )
                elif r_growth < 0 and p_growth > 0:
                    st.success(
                        f"📌 **Cost Restructuring:** Revenue contracted by {abs(r_growth):.2f}%, yet bottom-line profit grew by {p_growth:.2f}%, reflecting successful cost-cutting or higher operational efficiency."
                    )
                else:
                    st.warning(
                        f"📌 **Downturn Alert:** Both revenue ({r_growth:.2f}%) and net profit ({p_growth:.2f}%) declined, signaling market headwinds."
                    )

            # --------------------------------------------------
            # AUDIT SOURCES EXPANDER
            # --------------------------------------------------
            with st.expander("🔍 Auditable Extraction Sources"):
                for i, chunk in enumerate(st.session_state.get("fin_sources", [])):
                    page = chunk.metadata.get("page", 0) + 1
                    st.markdown(f"**Source Chunk {i+1} (Page {page}):**")
                    st.caption(chunk.page_content)
                    st.divider()

        else:
            st.warning(
                "Could not parse numerical financial tables from this document."
            )