from tools.statistics import (
    run_statistics
)


DATA_PATH = (
    "data/diabetes.csv"
)


def statistics_agent(state):

    results = run_statistics(
        DATA_PATH
    )

    results["planned_tasks"] = state.get(
        "research_plan",
        {}
    ).get(
        "statistical_tasks",
        []
    )

    return {
        "statistical_results":
            results
    }
