# GraniteScholar - IBM Granite AI Research Agent (Python)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![IBM Cloud Lite](https://img.shields.io/badge/IBM_Cloud-Lite_Services-052FAD?style=flat&logo=ibm&logoColor=white)](https://cloud.ibm.com)
[![IBM Granite](https://img.shields.io/badge/IBM_Granite-3.0_Instruct-1261FE?style=flat&logo=ibm&logoColor=white)](https://www.ibm.com/granite)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)

An autonomous AI Research Agent designed for academic and scientific research tasks using **IBM Cloud Lite services & IBM Granite LLMs**.

---

## 🌟 Key Features

1. **Autonomous Literature Search & Retrieval**: Real-time integration with **arXiv API** and **NCBI PubMed API**, plus curated offline domain repositories.
2. **IBM Granite Paper Summarizer & Interactive QA**: Multi-perspective paper breakdown (Core Problem, Methodology, Key Findings, Limitations) and interactive Q&A assistant powered by `ibm/granite-3-8b-instruct`.
3. **Multi-Paper Synthesis & Research Studio**:
   - Literature Review Synthesizer (Generates themed literature reviews from selected papers).
   - Scientific Hypothesis Formulator (Discovers research gaps and constructs testable scientific hypotheses).
   - Academic Section Drafter (Abstract, Related Work, Methodology, Conclusion).
4. **Comparative Paper Matrix**: Side-by-side comparative grid built using **Pandas** with one-click CSV and JSON export.
5. **Reference & Citation Manager**: Citation format generator for **APA 7th, IEEE, Harvard, Chicago, and BibTeX** with complete `.bib` library export.
6. **IBM Cloud Lite Architecture Showcase**: Interactive visualization of IBM Cloud Lite service integration (watsonx.ai Granite, Watson Discovery, Object Storage, Code Engine).

---

## 🚀 Quick Start

### 1. Installation
```bash
cd C:\Users\A\.gemini\antigravity\scratch\research-agent-python
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
python test_agent.py
```

### 3. Launch Streamlit Web Application
```bash
streamlit run app.py
```

---

## ☁️ Live IBM Cloud Lite / Watsonx Credentials Configuration

To connect live IBM Cloud Lite services:
1. Open the sidebar in the Streamlit web dashboard.
2. Expand **☁️ IBM Cloud Lite Credentials**.
3. Input your **IBM Cloud API Key** and **watsonx Project ID**.
4. Select your Cloud Region (default `us-south`).

*If no API key is provided, the application runs seamlessly in **IBM Granite Engine Simulation Mode**.*

---

## 📂 Project Structure

```
research-agent-python/
├── app.py                   # Main Streamlit Web Application
├── ibm_granite_engine.py    # IBM watsonx.ai REST & Granite Simulation Engine
├── literature_fetcher.py   # arXiv & PubMed API Paper Search Engine
├── research_generator.py   # IBM Granite Multi-Paper Synthesizer & Hypothesis Generator
├── matrix_extractor.py     # Pandas Comparative Paper Matrix Extractor
├── citation_manager.py     # APA, IEEE, BibTeX Citation & Library Generator
├── test_agent.py           # Unit Verification Test Suite
├── requirements.txt        # Python Dependencies
└── .streamlit/
    └── config.toml          # Streamlit Custom Dark Glassmorphism Theme
```
