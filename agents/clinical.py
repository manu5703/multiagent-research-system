from tools.guidelines import (
    search_guidelines
)


def clinical_agent(state):

    question = state[
        "research_question"
    ]

    guidelines = search_guidelines(
        question
    )

    return {
        "clinical_guidelines":
            guidelines
    }
