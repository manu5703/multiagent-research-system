from pydantic import BaseModel, Field
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage
from config import get_llm


llm = get_llm()

class ResearchPlan(BaseModel):

    objective: str

    subquestions: List[str] = Field(
        default_factory=list
    )

    literature_tasks: List[str] = Field(
        default_factory=list
    )

    clinical_tasks: List[str] = Field(
        default_factory=list
    )

    data_tasks: List[str] = Field(
        default_factory=list
    )

    statistical_tasks: List[str] = Field(
        default_factory=list
    )




planner_llm = llm.with_structured_output(
    ResearchPlan
)


def planner_agent(state):

    question = state[
        "research_question"
    ]

    messages = [
        SystemMessage(content="""You are a healthcare research planning agent.
Create a rigorous research plan. Do not answer the research question."""),
        HumanMessage(content=f"""

Research question:

{question}

Create a rigorous research plan.

The research should consider:

1. Scientific literature
2. Clinical guidelines
3. Patient-level observational data
4. Statistical analysis
5. Evidence validation
6. Conflicting evidence

Create specific executable research tasks.
""")
    ]

    plan = planner_llm.invoke(
        messages
    )

    return {
        "research_plan":
            plan.model_dump()
    }
