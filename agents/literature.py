from typing import List

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from tools.pubmed import (
    search_pubmed,
    fetch_pubmed_articles
)
from config import get_llm

llm = get_llm()

class EvidenceClaim(BaseModel):
    claim: str
    population: str = ""
    intervention: str = ""
    comparator: str = ""
    outcome: str = ""
    study_type: str = ""
    sample_size: str = ""
    effect: str = ""
    limitations: List[str] = Field(
        default_factory=list
    )

claim_llm = llm.with_structured_output(
    EvidenceClaim
)

def extract_claim(article):

    messages = [
        SystemMessage(content="""You are a biomedical evidence extraction agent.
Extract only information explicitly supported by the supplied source. Do not invent missing information."""),
        HumanMessage(content=f"""

TITLE:
{article['title']}

ABSTRACT:
{article['abstract']}

Extract:

- Main claim
- Population
- Intervention
- Comparator
- Outcome
- Study design
- Sample size if available
- Effect if available
- Limitations if available
""")
    ]
    result = claim_llm.invoke(
        messages
    )

    return result.model_dump()

def literature_agent(state):

    plan = state[
        "research_plan"
    ]
    literature_tasks = plan.get(
        "literature_tasks",
        []
    )

    all_articles = []

    # Search PubMed for each research task

    for task in literature_tasks:

        pmids = search_pubmed(task)

        articles = fetch_pubmed_articles(
            pmids
        )

        all_articles.extend(
            articles
        )

    # Remove duplicate papers

    unique_articles = {}

    for article in all_articles:

        unique_articles[
            article["pmid"]
        ] = article

    all_articles = list(
        unique_articles.values()
    )

    # Extract claims
    claims = []

    for article in all_articles:

        if not article[
            "abstract"
        ]:
            continue

        extracted = extract_claim(
            article
        )

        extracted[
            "pmid"
        ] = article[
            "pmid"
        ]

        extracted[
            "title"
        ] = article[
            "title"
        ]

        extracted[
            "source"
        ] = article[
            "source"
        ]

        claims.append(
            extracted
        )

    return {

        "literature":
            all_articles,

        "claims":
            claims
    }
