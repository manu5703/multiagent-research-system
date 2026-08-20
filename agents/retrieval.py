from tools.rag import load_approved_documents, pubmed_documents, retrieve_evidence
from tools.pubmed import fetch_pubmed_articles, search_pubmed
from agents.literature import extract_claim


def retrieval_agent(state):
    """Retrieve source passages for the question and planner subquestions."""
    plan = state.get("research_plan", {})
    queries = state.get("retrieval_queries") or [
        state["research_question"],
        *plan.get("literature_tasks", []),
        *plan.get("clinical_tasks", []),
    ]
    articles = list(state.get("literature", []))
    claims = list(state.get("claims", []))

    # On a refinement pass, retrieve new PubMed records instead of repeatedly
    # ranking the same evidence. The retry limit is enforced by the quality gate.
    if state.get("retrieval_attempts", 0) > 0:
        existing_pmids = {str(article.get("pmid", "")) for article in articles}
        refined_pmids = []
        for query in queries:
            refined_pmids.extend(search_pubmed(query))
        new_pmids = list(dict.fromkeys(
            pmid for pmid in refined_pmids if pmid not in existing_pmids
        ))
        new_articles = fetch_pubmed_articles(new_pmids)
        articles = [*articles, *new_articles]
        for article in new_articles:
            if not article.get("abstract"):
                continue
            claim = extract_claim(article)
            claim.update({
                "pmid": article["pmid"],
                "title": article["title"],
                "source": article["source"],
            })
            claims.append(claim)

    documents = [*pubmed_documents(articles), *load_approved_documents()]
    evidence_by_id = {}
    for query in queries:
        for chunk in retrieve_evidence(query, documents):
            evidence_by_id[chunk["evidence_id"]] = chunk
    return {
        "retrieval_queries": queries,
        "retrieved_evidence": list(evidence_by_id.values()),
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        "literature": articles,
        "claims": claims,
    }
