from typing import List

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from config import get_llm

llm = get_llm()

class EvidenceScore(BaseModel):

    claim: str

    study_design_quality: str

    population_relevance: str

    outcome_relevance: str

    consistency: str

    evidence_strength: str

    confidence: float = Field(
        ge=0,
        le=1
    )

    reasoning: str


class EvidenceScoreList(BaseModel):

    scores: List[EvidenceScore]



grader_llm = (
    llm.with_structured_output(
        EvidenceScoreList
    )
)


def evidence_grading_agent(state):

    supported_claims = (
        state.get(
            "supported_claims",
            []
        )
    )

    contradictions = (
        state.get(
            "contradictions",
            []
        )
    )

    if not supported_claims:
        return {"evidence_scores": []}

    messages = [
        SystemMessage(content="""You are a biomedical evidence grading agent.
Grade evidence conservatively. Do not give high confidence merely because an LLM considers a claim plausible."""),
        HumanMessage(content=f"""

Supported claims:

{supported_claims}

Contradiction analysis:

{contradictions}

Grade the evidence conservatively.

Consider:

- Study design
- Population relevance
- Outcome relevance
- Consistency across studies
- Validation result
- Contradictory evidence

Confidence must be between 0 and 1.
""")
    ]

    result = grader_llm.invoke(
        messages
    )

    scores = [
        score.model_dump()
        for score in result.scores
    ]

    return {
        "evidence_scores":
            scores
    }
