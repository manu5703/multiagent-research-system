from typing import List
from pydantic import BaseModel
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



llm = get_llm()


validator_llm = llm.with_structured_output(
    ValidationResult
)


def validate_claim(claim, article, research_question):

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
""")
    ]

    result = validator_llm.invoke(
        messages
    )

    return result.model_dump()


def validation_agent(state):

    claims = state.get(
        "claims",
        []
    )

    question = state[
        "research_question"
    ]

    validated_claims = []

    articles_by_pmid = {
        article["pmid"]: article
        for article in state.get("literature", [])
        if article.get("pmid")
    }

    for claim in claims:

        article = articles_by_pmid.get(
            claim.get("pmid"),
            {}
        )

        validation = validate_claim(
            claim,
            article,
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
    ]

    return {
        "validated_claims": validated_claims,
        "supported_claims": supported_claims
    }
