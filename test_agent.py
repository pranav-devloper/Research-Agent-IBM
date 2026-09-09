"""
Unit Test Suite for GraniteScholar Research Agent Python Modules.
"""

import sys
import unittest
from ibm_granite_engine import IBMGraniteEngine
from literature_fetcher import LiteratureFetcher
from research_generator import ResearchGenerator
from matrix_extractor import MatrixExtractor
from citation_manager import CitationManager

class TestGraniteScholar(unittest.TestCase):

    def setUp(self):
        self.engine = IBMGraniteEngine()
        self.fetcher = LiteratureFetcher()
        self.generator = ResearchGenerator(self.engine)

    def test_literature_fetcher_curated(self):
        results = self.fetcher.search("IBM Granite", source="All", max_results=5)
        self.assertGreater(len(results), 0)
        self.assertIn("title", results[0])
        self.assertIn("authors", results[0])
        print(f"[OK] Literature Fetcher Test Passed: Retrieved {len(results)} papers.")

    def test_ibm_granite_engine_simulation(self):
        res = self.engine.generate("Summarize deep learning paper", max_new_tokens=200)
        self.assertIn("generated_text", res)
        self.assertIn("latency_seconds", res)
        self.assertGreater(res["confidence_score"], 0.8)
        print(f"[OK] IBM Granite Engine Test Passed ({res['model_name']} - {res['latency_seconds']}s).")

    def test_matrix_extractor(self):
        papers = self.fetcher.search("RAG", max_results=3)
        df = MatrixExtractor.build_comparison_dataframe(papers)
        self.assertEqual(len(df), len(papers))
        self.assertIn("Title", df.columns)
        print("[OK] Matrix Extractor Test Passed.")

    def test_citation_manager(self):
        papers = self.fetcher.search("Granite", max_results=1)
        paper = papers[0]
        apa_cit = CitationManager.format_citation(paper, style="APA")
        bib_cit = CitationManager.format_citation(paper, style="BibTeX")
        self.assertIn(str(paper["year"]), apa_cit)
        self.assertIn("@article", bib_cit)
        print("[OK] Citation Manager Test Passed.")

    def test_research_generator(self):
        papers = self.fetcher.search("Granite", max_results=2)
        rev_res = self.generator.generate_literature_review(papers, focus_topic="Granite Models")
        hyp_res = self.generator.generate_scientific_hypothesis(papers)
        self.assertIn("generated_text", rev_res)
        self.assertIn("generated_text", hyp_res)
        print("[OK] Research Generator Test Passed.")

if __name__ == "__main__":
    unittest.main()
