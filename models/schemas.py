from typing import TypedDict, List, Dict, Any


class ResearchState(TypedDict, total=False):

    # INPUT
    research_question: str

    # PLANNER
    research_plan: Dict[str, Any]

    # LITERATURE
    literature: List[Dict[str, Any]]

    claims: List[Dict[str, Any]]

    # RAG EVIDENCE
    retrieved_evidence: List[Dict[str, Any]]
    retrieval_queries: List[str]
    retrieval_attempts: int
    evidence_sufficient: bool
    evidence_sufficiency_reason: str
    grounded_claims: List[Dict[str, Any]]

    # CLINICAL GUIDELINES
    clinical_guidelines: List[Dict[str, Any]]

    # STATISTICS
    statistical_results: Dict[str, Any]

    # VALIDATION
    validated_claims: List[Dict[str, Any]]

    # CONTRADICTIONS
    contradictions: Dict[str, Any]

    # EVIDENCE GRADING

    evidence_scores: List[Dict[str, Any]]

    # CLAIMS CLEARED FOR DOWNSTREAM USE
    supported_claims: List[Dict[str, Any]]

    # FINAL OUTPUT

    final_report: str
