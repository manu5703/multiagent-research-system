from typing import List
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from config import get_llm


class ValidationResult(BaseModel):

    supported: bool
    population_match: bool
    intervention_match: bool
    outcome_match: bool
    study_design_quality: str
    evidence_strength: str
    reason: str
    supporting_evidence_ids: List[str] = Field(default_factory=list)

llm = get_llm()

validator_llm = llm.with_structured_output(
    ValidationResult
)
def validate_claim(claim, article, evidence_chunks, research_question):

    messages = [
        SystemMessage(
            content="""You are a biomedical evidence validation agent.
Return a conservative assessment and do not assume information that is not provided."""
        ),
        HumanMessage(content=f"""

Research question:
{research_question}

Evaluate this extracted claim:
{claim}

Source title:
{article.get('title', '')}

Source abstract:
{article.get('abstract', '')}

Retrieved evidence chunks (use only their evidence_id values):
{evidence_chunks}

Determine whether the claim is supported by
the cited paper information.

Check:

1. Does the population match?
2. Does the intervention match?
3. Does the outcome match?
4. Is the study design appropriate?
5. Is the claim actually supported by the
   information extracted from the paper?

Return a conservative assessment.
If supported, include every evidence_id that directly supports the claim. If no
retrieved evidence chunk supports it, set supported to false.
""")
    ]

    result = validator_llm.invoke(
        messages
    )

    return result.model_dump()


def validation_agent(state):

    claims = [
        *state.get("claims", []),
        *state.get("grounded_claims", []),
    ]

    question = state[
        "research_question"
    ]

    validated_claims = []

    articles_by_pmid = {
        article["pmid"]: article
        for article in state.get("literature", [])
        if article.get("pmid")
    }

    evidence_by_pmid = {}
    for chunk in state.get("retrieved_evidence", []):
        if chunk.get("pmid"):
            evidence_by_pmid.setdefault(
                str(chunk["pmid"]),
                []
            ).append(chunk)

    for claim in claims:

        article = articles_by_pmid.get(
            claim.get("pmid"),
            {}
        )

        direct_evidence = []
        if claim.get("evidence_id"):
            direct_evidence = [
                chunk for chunk in state.get("retrieved_evidence", [])
                if chunk.get("evidence_id") == claim["evidence_id"]
            ]

        validation = validate_claim(
            claim,
            article,
            direct_evidence or evidence_by_pmid.get(str(claim.get("pmid", "")), []),
            question
        )

        updated_claim = claim.copy()

        updated_claim[
            "validation"
        ] = validation

        validated_claims.append(
            updated_claim
        )

    supported_claims = [
        claim for claim in validated_claims
        if claim["validation"]["supported"]
        and claim["validation"]["supporting_evidence_ids"]
    ]

    return {
        "validated_claims": validated_claims,
        "supported_claims": supported_claims
    }
