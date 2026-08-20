"""Convert retrieved source passages into conservative, citable claim candidates."""

from typing import List

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from config import get_llm


class GroundedClaim(BaseModel):
    claim: str
    population: str = ""
    intervention: str = ""
    comparator: str = ""
    outcome: str = ""
    study_type: str = ""
    limitations: List[str] = Field(default_factory=list)


grounding_llm = get_llm().with_structured_output(GroundedClaim)
MAX_GROUNDED_CHUNKS = 6


def grounding_agent(state):
    """Generate only claims explicitly supported by top retrieved passages."""
    claims = []
    chunks = sorted(
        state.get("retrieved_evidence", []),
        key=lambda item: item.get("relevance_score", 0),
        reverse=True,
    )[:MAX_GROUNDED_CHUNKS]

    for chunk in chunks:
        # PubMed records already have a separate structured-claim extraction path.
        if chunk.get("source_type") == "pubmed_abstract":
            continue
        result = grounding_llm.invoke([
            SystemMessage(content=(
                "You extract one conservative biomedical evidence claim. Use only "
                "the supplied passage; do not infer a recommendation, effect, or "
                "population that is not explicit."
            )),
            HumanMessage(content=f"""Research question: {state['research_question']}
Source title: {chunk.get('title', '')}
Source section: {chunk.get('section', '')}
Evidence ID: {chunk['evidence_id']}
Passage: {chunk['content']}

Extract one claim relevant to the research question. If the passage is not
relevant, return a claim stating that no relevant claim is available and list
"not relevant" as a limitation."""),
        ])
        claim = result.model_dump()
        claim.update({
            "evidence_id": chunk["evidence_id"],
            "title": chunk.get("title", ""),
            "source": chunk.get("organization", ""),
            "pmid": chunk.get("pmid", ""),
        })
        claims.append(claim)
    return {"grounded_claims": claims}
