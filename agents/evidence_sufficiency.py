from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from config import get_llm


class EvidenceSufficiency(BaseModel):
    sufficient: bool
    reason: str
    refined_queries: list[str] = Field(default_factory=list)


quality_llm = get_llm().with_structured_output(EvidenceSufficiency)


def evidence_sufficiency_agent(state):
    """Decide whether one additional, bounded retrieval pass is needed."""
    attempts = state.get("retrieval_attempts", 0)
    if attempts >= 2:
        return {
            "evidence_sufficient": True,
            "evidence_sufficiency_reason": "Retrieval retry limit reached.",
        }
    result = quality_llm.invoke([
        SystemMessage(content="You are a conservative biomedical evidence quality gate. Evidence is sufficient only if it is relevant and claim-level supported. Do not invent sources."),
        HumanMessage(content=f"""Research question: {state['research_question']}
Supported claims: {state.get('supported_claims', [])}
Retrieved evidence: {state.get('retrieved_evidence', [])}

Decide whether the evidence base is sufficient for a cautious synthesis. If not,
return 1-3 focused PICO-oriented refined PubMed/guideline search queries."""),
    ])
    return {
        "evidence_sufficient": result.sufficient,
        "evidence_sufficiency_reason": result.reason,
        "retrieval_queries": result.refined_queries or state.get("retrieval_queries", []),
    }
