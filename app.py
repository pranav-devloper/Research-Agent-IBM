"""
GraniteScholar - Autonomous AI Research Agent in Python
Powered by IBM Cloud Lite Services & IBM Granite LLMs
"""

import os
import json
import pandas as pd
import streamlit as st

from ibm_granite_engine import IBMGraniteEngine
from literature_fetcher import LiteratureFetcher
from research_generator import ResearchGenerator
from matrix_extractor import MatrixExtractor
from citation_manager import CitationManager

# Page Configuration
st.set_page_config(
    page_title="GraniteScholar - IBM Granite AI Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Aesthetics (CSS Injection)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    code, pre, [class*="stCode"] {
        font-family: 'JetBrains+Mono', monospace !important;
    }

    /* Main Container Glassmorphism */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(15, 23, 42, 1) 0%, rgba(30, 41, 59, 1) 90%);
        color: #F8FAFC;
    }

    /* Custom Header Banner */
    .main-header {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(147, 51, 234, 0.15));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #60A5FA, #A78BFA, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .main-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-top: 6px;
    }

    /* Badge Tags */
    .badge-ibm {
        background: rgba(59, 130, 246, 0.2);
        color: #60A5FA;
        border: 1px solid rgba(96, 165, 250, 0.3);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-sim {
        background: rgba(245, 158, 11, 0.2);
        color: #FBBF24;
        border: 1px solid rgba(251, 191, 36, 0.3);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Paper Cards */
    .paper-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .paper-card:hover {
        border-color: rgba(96, 165, 250, 0.4);
        transform: translateY(-2px);
    }
    .paper-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #F1F5F9;
        margin-bottom: 8px;
    }
    .paper-meta {
        font-size: 0.85rem;
        color: #94A3B8;
        margin-bottom: 12px;
    }

    /* IBM Granite Response Block */
    .granite-box {
        background: rgba(15, 23, 42, 0.8);
        border-left: 4px solid #3B82F6;
        border-radius: 8px;
        padding: 20px;
        margin-top: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "workspace_papers" not in st.session_state:
    st.session_state.workspace_papers = []
if "ibm_api_key" not in st.session_state:
    st.session_state.ibm_api_key = ""
if "ibm_project_id" not in st.session_state:
    st.session_state.ibm_project_id = ""
if "ibm_region" not in st.session_state:
    st.session_state.ibm_region = "us-south"
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "ibm/granite-3-8b-instruct"
if "search_results" not in st.session_state:
    st.session_state.search_results = []

# Instantiate Services
granite_engine = IBMGraniteEngine(
    api_key=st.session_state.ibm_api_key,
    project_id=st.session_state.ibm_project_id,
    region=st.session_state.ibm_region
)
granite_engine.model_id = st.session_state.selected_model

fetcher = LiteratureFetcher()
generator = ResearchGenerator(granite_engine)

# ================= SIDEBAR CONFIGURATION =================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("GraniteScholar Core")
    st.markdown("IBM Cloud Lite & Granite LLM Suite")
    
    st.divider()

    # IBM Granite Model Selector
    st.subheader("🤖 IBM Granite Model")
    model_options = list(IBMGraniteEngine.AVAILABLE_MODELS.keys())
    model_names = [IBMGraniteEngine.AVAILABLE_MODELS[m]["name"] for m in model_options]
    
    selected_idx = model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
    chosen_model_name = st.selectbox(
        "Active LLM",
        options=model_names,
        index=selected_idx,
        help="Select target IBM Granite model architecture"
    )
    st.session_state.selected_model = model_options[model_names.index(chosen_model_name)]
    
    model_info = IBMGraniteEngine.AVAILABLE_MODELS[st.session_state.selected_model]
    st.caption(f"ℹ️ {model_info['description']}")

    st.divider()

    # IBM Cloud Lite Credentials Drawer
    with st.expander("☁️ IBM Cloud Lite Credentials"):
        st.caption("Configure live IBM watsonx.ai REST endpoints:")
        api_key_input = st.text_input("IBM Cloud API Key", value=st.session_state.ibm_api_key, type="password")
        proj_id_input = st.text_input("watsonx Project ID", value=st.session_state.ibm_project_id)
        region_input = st.selectbox("Cloud Region", options=["us-south", "eu-de", "jp-tok"], index=0)
        
        if st.button("Save IBM Credentials", use_container_width=True):
            st.session_state.ibm_api_key = api_key_input
            st.session_state.ibm_project_id = proj_id_input
            st.session_state.ibm_region = region_input
            st.success("Credentials saved!")
            st.rerun()

    # Status Indicator
    if granite_engine.is_live_configured():
        st.markdown('<span class="badge-ibm">● Live IBM watsonx.ai Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-sim">⚡ IBM Granite Engine (Simulation Mode)</span>', unsafe_allow_html=True)

    st.divider()

    # Research Workspace Basket
    st.subheader(f"📚 Research Basket ({len(st.session_state.workspace_papers)})")
    if st.session_state.workspace_papers:
        for idx, paper in enumerate(st.session_state.workspace_papers):
            cols = st.columns([0.85, 0.15])
            cols[0].caption(f"• **{paper['title'][:32]}...** ({paper['year']})")
            if cols[1].button("❌", key=f"del_{idx}"):
                st.session_state.workspace_papers.pop(idx)
                st.rerun()

        if st.button("Clear Basket", use_container_width=True):
            st.session_state.workspace_papers = []
            st.rerun()
    else:
        st.caption("No papers added yet. Search and click 'Add to Basket'.")

# ================= MAIN PAGE HEADER =================
st.markdown("""
<div class="main-header">
    <div class="main-title">GraniteScholar: AI Research Agent</div>
    <div class="main-subtitle">Autonomous Academic Literature Discovery, Synthesis & Hypothesis Formulation powered by <b>IBM Cloud Lite & IBM Granite LLMs</b></div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_search, tab_summary, tab_studio, tab_matrix, tab_citations, tab_arch = st.tabs([
    "🔍 Literature Search",
    "⚡ Paper Summarizer & QA",
    "🔬 Synthesis & Hypotheses",
    "📊 Comparative Matrix",
    "📚 Citation Manager",
    "☁️ IBM Architecture"
])

# ================= TAB 1: LITERATURE SEARCH =================
with tab_search:
    st.subheader("Autonomous Academic Literature Search")
    st.caption("Search across millions of preprints and peer-reviewed studies via live arXiv API, PubMed API, or curated domains.")

    c1, c2, c3 = st.columns([0.5, 0.25, 0.25])
    search_query = c1.text_input("Search Prompt / Research Question", value="IBM Granite transformer genomic RAG", placeholder="e.g. quantum error mitigation, RAG in medicine")
    search_source = c2.selectbox("Source Database", options=["All", "arXiv", "PubMed"])
    max_results = c3.slider("Max Results", min_value=3, max_value=12, value=6)

    if st.button("🔍 Execute Search Query", type="primary", use_container_width=True):
        with st.spinner("Querying academic APIs and retrieving metadata..."):
            results = fetcher.search(query=search_query, source=search_source, max_results=max_results)
            st.session_state.search_results = results
            st.success(f"Retrieved {len(results)} scientific papers.")

    # Custom Resource Entry Box
    with st.expander("📤 Add Custom Resource or Manual Paper Entry to Basket"):
        st.caption("Upload or manually input custom research papers, PDFs, or abstracts to add to your Research Basket.")
        c_title = st.text_input("Paper Title", placeholder="e.g. Fine-Tuning IBM Granite for Enterprise RAG")
        c_authors = st.text_input("Authors (comma-separated)", placeholder="e.g. Alice Smith, Bob Johnson")
        c_abstract = st.text_area("Abstract / Executive Summary / Notes", placeholder="Paste abstract or paper text here...")
        col_y, col_v = st.columns([0.5, 0.5])
        c_year = col_y.number_input("Publication Year", min_value=1900, max_value=2030, value=2026)
        c_venue = col_v.text_input("Venue / Journal", value="Custom Laboratory Upload")
        uploaded_file = st.file_uploader("Upload PDF or Text File (Optional)", type=["pdf", "txt", "md"])

        if st.button("➕ Add Custom Resource to Research Basket", use_container_width=True):
            file_text = ""
            if uploaded_file is not None:
                try:
                    file_bytes = uploaded_file.read()
                    file_text = file_bytes.decode("utf-8", errors="ignore")[:2000]
                except Exception:
                    file_text = f"Uploaded file: {uploaded_file.name}"

            combined_abstract = c_abstract if c_abstract else file_text or "Custom research paper uploaded to workspace."
            new_paper = fetcher.parse_uploaded_resource(
                title=c_title or (uploaded_file.name if uploaded_file else "Untitled Custom Paper"),
                authors=c_authors.split(",") if c_authors else ["User Upload"],
                abstract=combined_abstract,
                year=int(c_year),
                venue=c_venue
            )
            st.session_state.workspace_papers.append(new_paper)
            st.success(f"Added custom resource '{new_paper['title']}' to Research Basket!")
            st.rerun()

    # Display Search Results
    if st.session_state.search_results:
        st.divider()
        st.markdown(f"### Found Results ({len(st.session_state.search_results)} Papers)")

        for p_idx, paper in enumerate(st.session_state.search_results):
            with st.container():
                st.markdown(f"""
                <div class="paper-card">
                    <div class="paper-title">[{paper['source']}] {paper['title']}</div>
                    <div class="paper-meta">
                        <b>Authors:</b> {', '.join(paper.get('authors', []))} | 
                        <b>Year:</b> {paper.get('year')} | 
                        <b>Venue:</b> {paper.get('venue')} | 
                        <b>Citations:</b> {paper.get('citations')}
                    </div>
                    <p style="color: #CBD5E1; font-size: 0.95rem; line-height: 1.5;">{paper.get('abstract')[:320]}...</p>
                </div>
                """, unsafe_allow_html=True)

                b_col1, b_col2, b_col3, b_col4 = st.columns([0.25, 0.25, 0.25, 0.25])
                
                # Add to Basket
                is_in_basket = any(p["id"] == paper["id"] for p in st.session_state.workspace_papers)
                if is_in_basket:
                    b_col1.button("✓ In Basket", key=f"in_{p_idx}", disabled=True)
                else:
                    if b_col1.button("➕ Add to Basket", key=f"add_{p_idx}"):
                        st.session_state.workspace_papers.append(paper)
                        st.success(f"Added paper to basket!")
                        st.rerun()

                # View PDF / Link
                b_col2.link_button("📄 View PDF", paper.get("pdf_url", paper.get("url", "#")))
                
                # Copy BibTeX
                if b_col3.button("📋 Copy BibTeX", key=f"bib_{p_idx}"):
                    bib_str = CitationManager.to_bibtex(paper)
                    st.code(bib_str, language="bibtex")

                # Inspect DOI
                b_col4.caption(f"DOI: {paper.get('doi', 'N/A')}")
    else:
        st.info("Enter a query above to start searching scientific literature.")

# ================= TAB 2: PAPER SUMMARIZER & QA =================
with tab_summary:
    st.subheader("IBM Granite Paper Breakdown & Interactive Q&A")
    st.caption("Perform multi-perspective analysis on any paper or ask direct questions to IBM Granite.")

    if not st.session_state.workspace_papers:
        st.warning("⚠️ Your Research Basket is empty! Search for papers in Tab 1 and click 'Add to Basket'.")
    else:
        selected_paper_title = st.selectbox(
            "Select Paper for Summarization",
            options=[p["title"] for p in st.session_state.workspace_papers]
        )
        target_paper = next(p for p in st.session_state.workspace_papers if p["title"] == selected_paper_title)

        st.markdown(f"**Target Paper:** *{target_paper['title']}* ({target_paper['year']})")
        st.caption(f"Authors: {', '.join(target_paper.get('authors', []))}")

        sum_col1, sum_col2 = st.columns([0.5, 0.5])

        with sum_col1:
            if st.button("⚡ Generate IBM Granite Breakdown", type="primary", use_container_width=True):
                with st.spinner("IBM Granite analyzing paper methodology and key findings..."):
                    prompt = f"Summarize paper title: {target_paper['title']}\nAbstract: {target_paper['abstract']}"
                    res = granite_engine.generate(prompt=prompt, max_new_tokens=1024)
                    
                    st.markdown('<div class="granite-box">', unsafe_allow_html=True)
                    st.markdown(res["generated_text"])
                    st.divider()
                    st.caption(f"🤖 **Model**: {res['model_name']} | ⏱️ **Latency**: {res['latency_seconds']}s | 🎯 **Confidence**: {res['confidence_score']}")
                    st.markdown('</div>', unsafe_allow_html=True)

        with sum_col2:
            st.markdown("#### 💬 Ask IBM Granite About This Paper")
            user_qa_prompt = st.text_input("Question regarding methodology, dataset, or results:", placeholder="What dataset was used in this study?")
            if st.button("Submit Question to IBM Granite", use_container_width=True):
                if user_qa_prompt:
                    with st.spinner("Processing question..."):
                        qa_prompt = f"Paper Context: {target_paper['title']}\nAbstract: {target_paper['abstract']}\nQuestion: {user_qa_prompt}"
                        qa_res = granite_engine.generate(prompt=qa_prompt)
                        st.markdown('<div class="granite-box">', unsafe_allow_html=True)
                        st.markdown(qa_res["generated_text"])
                        st.caption(f"Mode: {qa_res['mode']}")
                        st.markdown('</div>', unsafe_allow_html=True)

# ================= TAB 3: SYNTHESIS & HYPOTHESES =================
with tab_studio:
    st.subheader("Multi-Paper Synthesis & Scientific Hypothesis Generator")
    st.caption("Leverage IBM Granite 3.0 to find research gaps across papers, generate literature reviews, and draft paper sections.")

    if not st.session_state.workspace_papers:
        st.warning("⚠️ Add at least 1-2 papers to your Research Basket to perform multi-paper synthesis.")
    else:
        st.markdown(f"**Active Workspace Papers ({len(st.session_state.workspace_papers)})**")

        st_tab1, st_tab2, st_tab3 = st.tabs(["📚 Literature Review Drafter", "🧪 Hypothesis Formulator", "✍️ Section Drafter"])

        with st_tab1:
            focus_topic = st.text_input("Review Focus Topic / Domain Theme", value="Advances in RAG and IBM Granite LLMs")
            if st.button("🚀 Synthesize Literature Review", type="primary"):
                with st.spinner("IBM Granite synthesizing across retrieved papers..."):
                    rev_res = generator.generate_literature_review(st.session_state.workspace_papers, focus_topic)
                    st.markdown('<div class="granite-box">', unsafe_allow_html=True)
                    st.markdown(rev_res["generated_text"])
                    st.markdown('</div>', unsafe_allow_html=True)

        with st_tab2:
            st.markdown("Formulate testable scientific hypotheses by cross-analyzing gaps in retrieved papers.")
            if st.button("🧪 Formulate Hypotheses with IBM Granite", type="primary"):
                with st.spinner("Analyzing research gaps and formulating hypotheses..."):
                    hyp_res = generator.generate_scientific_hypothesis(st.session_state.workspace_papers)
                    st.markdown('<div class="granite-box">', unsafe_allow_html=True)
                    st.markdown(hyp_res["generated_text"])
                    st.markdown('</div>', unsafe_allow_html=True)

        with st_tab3:
            sec_name = st.selectbox("Target Section", ["Abstract", "Introduction", "Related Work", "Methodology", "Discussion & Conclusion"])
            sec_topic = st.text_input("Paper Title / Core Topic", value="Scalable RAG with IBM Granite")
            sec_keypoints = st.text_area("Key Points to emphasize:", value="1. Reduced latency\n2. Low memory footprint\n3. High factual grounding")
            
            if st.button("✍️ Draft Paper Section"):
                with st.spinner("Drafting section..."):
                    sec_res = generator.draft_paper_section(sec_name, sec_topic, sec_keypoints)
                    st.markdown('<div class="granite-box">', unsafe_allow_html=True)
                    st.markdown(sec_res["generated_text"])
                    st.markdown('</div>', unsafe_allow_html=True)

# ================= TAB 4: COMPARATIVE MATRIX =================
with tab_matrix:
    st.subheader("Comparative Paper Matrix & Data Extraction")
    st.caption("Side-by-side comparison table generated across key scientific dimensions.")

    if not st.session_state.workspace_papers:
        st.info("Add papers to your Research Basket to generate a comparative matrix.")
    else:
        df_summary = MatrixExtractor.build_comparison_dataframe(st.session_state.workspace_papers)
        st.dataframe(df_summary, use_container_width=True)

        st.divider()

        st.markdown("#### Deep Methodology Comparison Matrix")
        df_method = MatrixExtractor.build_deep_methodology_matrix(st.session_state.workspace_papers)
        st.dataframe(df_method, use_container_width=True)

        exp_col1, exp_col2 = st.columns([0.5, 0.5])
        csv_data = df_summary.to_csv(index=False).encode('utf-8')
        exp_col1.download_button(
            label="📥 Download Comparative Summary (CSV)",
            data=csv_data,
            file_name="granite_scholar_matrix.csv",
            mime="text/csv",
            use_container_width=True
        )

        json_data = json.dumps(st.session_state.workspace_papers, indent=2).encode('utf-8')
        exp_col2.download_button(
            label="📥 Download Full Corpus (JSON)",
            data=json_data,
            file_name="granite_scholar_corpus.json",
            mime="application/json",
            use_container_width=True
        )

# ================= TAB 5: CITATION MANAGER =================
with tab_citations:
    st.subheader("Reference & Citation Manager")
    st.caption("Generate formatted citations and export your library to BibTeX (`.bib`).")

    if not st.session_state.workspace_papers:
        st.info("Add papers to your Research Basket to format citations.")
    else:
        cit_style = st.selectbox("Select Citation Format", options=["APA", "IEEE", "Harvard", "Chicago", "BibTeX"])

        st.divider()
        st.markdown(f"### Formatted Citations ({cit_style})")

        for paper in st.session_state.workspace_papers:
            formatted_cit = CitationManager.format_citation(paper, style=cit_style)
            st.markdown(f"- {formatted_cit}")

        st.divider()

        bib_library = CitationManager.generate_bibtex_library(st.session_state.workspace_papers)
        st.markdown("#### Complete BibTeX Library Export")
        st.code(bib_library, language="bibtex")

        st.download_button(
            label="📥 Download .bib File (BibTeX Library)",
            data=bib_library.encode("utf-8"),
            file_name="granite_scholar_references.bib",
            mime="text/plain",
            use_container_width=True
        )

# ================= TAB 6: IBM ARCHITECTURE VIEW =================
with tab_arch:
    st.subheader("IBM Cloud Lite & IBM Granite Solution Architecture")
    st.caption("Technical architectural diagram demonstrating end-to-end IBM Cloud Lite service integration.")

    st.markdown("""
    ```
    ┌────────────────────────────────────────────────────────────────────────────────────────┐
    │                                GraniteScholar UI (Streamlit)                            │
    └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
    ┌──────────────────────────┐                                 ┌──────────────────────────┐
    │  arXiv / PubMed APIs     │                                 │   IBM watsonx.ai         │
    │  (Literature Fetcher)    │                                 │   REST Endpoint          │
    └────────────┬─────────────┘                                 └─────────────┬────────────┘
                 │                                                             │
                 ▼                                                             ▼
    ┌──────────────────────────┐                                 ┌──────────────────────────┐
    │  Pandas & Matrix Engine  │                                 │  IBM Granite 3.0 Model   │
    │  (Structured Extraction) │                                 │  (8B / 2B / Vision)      │
    └────────────┬─────────────┘                                 └─────────────┬────────────┘
                 │                                                             │
                 └──────────────────────────────┬──────────────────────────────┘
                                                ▼
                               ┌──────────────────────────────────┐
                               │  IBM Cloud Object Storage &      │
                               │  IBM Watson Discovery (Vector)   │
                               └──────────────────────────────────┘
    ```
    """)

    arch_c1, arch_c2 = st.columns([0.5, 0.5])

    with arch_c1:
        st.markdown("""
        #### ☁️ IBM Cloud Lite Services Leveraged:
        1. **IBM watsonx.ai (IBM Granite Models)**:
           - Powers core paper summarization, gap identification, scientific hypothesis generation, and literature synthesis.
        2. **IBM Watson Discovery (RAG Vector Indexing)**:
           - Provides dense semantic searching across indexed scientific PDF documents.
        3. **IBM Cloud Object Storage (COS)**:
           - Stores raw PDF research paper repositories, dataset embeddings, and user library metadata.
        4. **IBM Code Engine / Cloud Functions**:
           - Serverless execution of literature API retrieval scripts and background citation formatting.
        """)

    with arch_c2:
        st.markdown("""
        #### 🎯 Key Architectural Advantages:
        - **Enterprise AI Safety**: IBM Granite models are filtered for data governance, copyright compliance, and academic rigor.
        - **Zero Latency Bottlenecks**: Hybrid caching & fallbacks ensure uninterrupted agent performance.
        - **Modular & Extensible**: Easily scales from personal reference manager to lab-wide multi-agent discovery pipelines.
        """)
