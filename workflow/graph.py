from langgraph.graph import (
    StateGraph,
    START,
    END
)

from models.schemas import ResearchState

from agents.planner import (
    planner_agent
)

from agents.literature import (
    literature_agent
)

from agents.clinical import (
    clinical_agent
)

from agents.statistics import (
    statistics_agent
)

from agents.validator import (
    validation_agent
)

from agents.retrieval import (
    retrieval_agent
)

from agents.grounding import (
    grounding_agent
)

from agents.evidence_sufficiency import (
    evidence_sufficiency_agent
)

from agents.contradiction import (
    contradiction_agent
)

from agents.grader import (
    evidence_grading_agent
)

from agents.synthesis import (
    synthesis_agent
)


# --------------------------------------------------
# Create graph
# --------------------------------------------------

workflow = StateGraph(
    ResearchState
)


# --------------------------------------------------
# Add agents
# --------------------------------------------------

workflow.add_node(
    "planner",
    planner_agent
)

workflow.add_node(
    "literature",
    literature_agent
)

workflow.add_node(
    "clinical",
    clinical_agent
)

workflow.add_node(
    "statistics",
    statistics_agent
)

workflow.add_node(
    "validation",
    validation_agent
)

workflow.add_node(
    "retrieval",
    retrieval_agent
)

workflow.add_node(
    "grounding",
    grounding_agent
)

workflow.add_node(
    "evidence_sufficiency",
    evidence_sufficiency_agent
)

workflow.add_node(
    "contradiction",
    contradiction_agent
)

workflow.add_node(
    "grading",
    evidence_grading_agent
)

workflow.add_node(
    "synthesis",
    synthesis_agent
)


# --------------------------------------------------
# Workflow
# --------------------------------------------------

workflow.add_edge(
    START,
    "planner"
)

workflow.add_edge("planner", "literature")
workflow.add_edge("planner", "clinical")
workflow.add_edge("planner", "statistics")

workflow.add_edge("literature", "retrieval")

workflow.add_edge("retrieval", "grounding")
workflow.add_edge("grounding", "validation")

workflow.add_edge("validation", "evidence_sufficiency")


def route_after_evidence_check(state):
    if state.get("evidence_sufficient"):
        return "contradiction"
    return "retrieval"


workflow.add_conditional_edges(
    "evidence_sufficiency",
    route_after_evidence_check,
    {
        "retrieval": "retrieval",
        "contradiction": "contradiction"
    }
)

workflow.add_edge(
    "contradiction",
    "grading"
)

# Synthesis needs the independently produced clinical and statistical results,
# as well as the evidence-grading result.
workflow.add_edge(["grading", "clinical", "statistics"], "synthesis")

workflow.add_edge(
    "synthesis",
    END
)


# --------------------------------------------------
# Compile
# --------------------------------------------------

graph = workflow.compile()
