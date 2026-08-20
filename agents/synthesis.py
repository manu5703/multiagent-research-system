from config import get_llm
from langchain_core.messages import HumanMessage, SystemMessage

llm = get_llm()

def synthesis_agent(state):

    question = state[
        "research_question"
    ]

    supported_claims = state.get(
        "supported_claims",
        []
    )

    clinical_guidelines = state.get(
        "clinical_guidelines",
        []
    )

    statistical_results = state.get(
        "statistical_results",
        {}
    )

    contradictions = state.get(
        "contradictions",
        []
    )

    evidence_scores = state.get(
        "evidence_scores",
        []
    )

    messages = [
        SystemMessage(content="""You are a senior biomedical research scientist.
Generate an evidence-grounded report. Do not invent results, citations, or guideline recommendations. Do not claim causality from observational data."""),
        HumanMessage(content=f"""

RESEARCH QUESTION:
{question}

SUPPORTED LITERATURE CLAIMS:
{supported_claims}

CLINICAL SOURCE METADATA:
{clinical_guidelines}

PATIENT DATA ANALYSIS:
{statistical_results}

CONTRADICTION ANALYSIS:
{contradictions}

EVIDENCE GRADES:
{evidence_scores}

RETRIEVED EVIDENCE WITH PROVENANCE:
{state.get('retrieved_evidence', [])}

EVIDENCE-SUFFICIENCY DECISION:
{state.get('evidence_sufficiency_reason', '')}

Write the report using these sections:

1. Research Question
2. Background
3. Research Approach
4. Literature Findings
5. Clinical Guideline Sources Available
6. Patient Cohort Findings
7. Statistical Findings
8. Comparison Between Patient Data and Literature
9. Conflicting Evidence
10. Limitations
11. Research Gaps
12. Conclusion

IMPORTANT RULES:

- Clearly distinguish association from causation.
- Do not describe a guideline's recommendations unless its content is supplied.
- Clearly identify uncertainty.
- Cite literature using PMID values when available.
- For every clinical or research finding, cite its supplied PMID or evidence_id.
- Do not use a source merely because it was retrieved; use only validated claims.
- Use only the evidence supplied to you.
""")
    ]

    response = llm.invoke(
        messages
    )

    return {
        "final_report":
            response.content
    }
