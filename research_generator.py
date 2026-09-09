"""
Research Generator Module for GraniteScholar Research Agent.
Uses IBM Granite LLM to generate literature reviews, hypotheses, and paper draft sections.
"""

from typing import List, Dict, Any, Optional
from ibm_granite_engine import IBMGraniteEngine

class ResearchGenerator:
    """
    Orchestrates high-level academic generation tasks powered by IBM Granite.
    """

    def __init__(self, granite_engine: IBMGraniteEngine):
        self.engine = granite_engine

    def generate_literature_review(self, papers: List[Dict[str, Any]], focus_topic: str = "General Synthesis") -> Dict[str, Any]:
        """
        Synthesizes multiple selected papers into a cohesive literature review draft.
        """
        if not papers:
            return {
                "generated_text": "Please select at least one research paper to synthesize.",
                "model_name": self.engine.AVAILABLE_MODELS[self.engine.model_id]["name"]
            }

        context_blocks = []
        for idx, paper in enumerate(papers, 1):
            authors_str = ", ".join(paper.get("authors", []))
            context_blocks.append(
                f"Paper [{idx}]: {paper.get('title')}\n"
                f"Authors: {authors_str} ({paper.get('year')})\n"
                f"Abstract: {paper.get('abstract')}\n"
            )

        full_context = "\n---\n".join(context_blocks)
        
        system_prompt = (
            "You are IBM Granite 3.0, a senior academic AI research agent. "
            "Synthesize the provided papers into a rigorous, well-structured literature review. "
            "Group literature by thematic paradigms, highlight critical differences, and maintain formal scientific prose."
        )

        user_prompt = (
            f"Focus Topic: {focus_topic}\n\n"
            f"Retrieved Papers Corpus:\n{full_context}\n\n"
            f"Task: Generate a 4-section Literature Review (1. Introduction & Context, 2. Methodological Paradigms, "
            f"3. Comparative Synthesis & Research Gaps, 4. Strategic Recommendations)."
        )

        return self.engine.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_new_tokens=2048,
            temperature=0.7
        )

    def generate_scientific_hypothesis(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyzes paper corpus to identify open gaps and formulate testable scientific hypotheses.
        """
        if not papers:
            return {
                "generated_text": "Please select papers to formulate hypotheses.",
                "model_name": self.engine.AVAILABLE_MODELS[self.engine.model_id]["name"]
            }

        summaries = [f"Title: {p['title']}\nAbstract: {p['abstract']}" for p in papers]
        context = "\n---\n".join(summaries)

        system_prompt = (
            "You are IBM Granite 3.0, an expert scientific hypothesis formulator. "
            "Analyze existing research findings, pinpoint latent gaps, and construct testable scientific hypotheses."
        )

        user_prompt = (
            f"Target Literature Set:\n{context}\n\n"
            f"Task:\n1. Identify 2 key unaddressed research gaps across these papers.\n"
            f"2. Formulate 2 explicit, testable Hypotheses (H1 and H2).\n"
            f"3. Provide theoretical rationale for each hypothesis.\n"
            f"4. Propose an empirical verification strategy and metric benchmark."
        )

        return self.engine.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_new_tokens=1500,
            temperature=0.8
        )

    def draft_paper_section(
        self,
        section_name: str,
        topic: str,
        key_points: str,
        tone: str = "Academic Formal"
    ) -> Dict[str, Any]:
        """
        Drafts specific paper sections (Abstract, Introduction, Related Work, Methodology, Results, Conclusion).
        """
        system_prompt = (
            f"You are IBM Granite 3.0, an expert academic writer drafting a publication-ready paper section. "
            f"Maintain an '{tone}' writing style with rigorous logical flow and technical precision."
        )

        user_prompt = (
            f"Target Section: {section_name}\n"
            f"Research Topic: {topic}\n"
            f"Key Points & Constraints to Include:\n{key_points}\n\n"
            f"Task: Write a comprehensive, publication-ready {section_name} draft complete with subheadings and bullet points where appropriate."
        )

        return self.engine.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_new_tokens=1800,
            temperature=0.6
        )
