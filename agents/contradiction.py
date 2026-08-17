from pydantic import BaseModel, Field
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage

from config import get_llm

llm = get_llm()
class ContradictionAnalysis(BaseModel):

    insufficient_evidence: bool = False

    agreement: List[str] = Field(
        default_factory=list
    )

    disagreements: List[str] = Field(
        default_factory=list
    )

    possible_reasons: List[str] = Field(
        default_factory=list
    )

    conclusion: str = ""




contradiction_llm = (
    llm.with_structured_output(
        ContradictionAnalysis
    )
)


def contradiction_agent(state):

    supported_claims = state.get(
        "supported_claims",
        []
    )

    if len(
        supported_claims
    ) < 2:

        return {
            "contradictions": ContradictionAnalysis(
                insufficient_evidence=True,
                conclusion=(
                    "Not enough supported claims to assess contradictions."
                )
            ).model_dump()
        }

    messages = [
        SystemMessage(content="""You are a biomedical contradiction detection agent.
Return a conservative scientific assessment. Do not declare studies contradictory merely because effect sizes differ."""),
        HumanMessage(content=f"""

Compare the following validated claims:

{supported_claims}

Identify:

1. Claims that agree
2. Claims that disagree
3. Potential reasons for disagreement
4. Population differences
5. Study design differences
6. Differences in interventions
7. Differences in outcomes
8. Differences in follow-up periods

""")
    ]

    result = contradiction_llm.invoke(
        messages
    )

    return {
        "contradictions":
            result.model_dump()
    }
