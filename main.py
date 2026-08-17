from dotenv import load_dotenv

from workflow.graph import graph

load_dotenv()


RESEARCH_QUESTION = """
Among adults with Type 2 diabetes, which patient
characteristics are associated with better glycemic
response to different glucose-lowering medication
classes, and what does existing clinical evidence
suggest about treatment effectiveness across
patient subgroups?
"""


def main():

    print("=" * 80)
    print("HEALTHCARE AGENTIC RESEARCH SYSTEM")
    print("=" * 80)

    print("\nResearch Question:")
    print(RESEARCH_QUESTION)

    print("\nStarting research workflow...\n")

    result = graph.invoke({

        "research_question":
            RESEARCH_QUESTION

    })

    print("\n")
    print("=" * 80)
    print("FINAL RESEARCH REPORT")
    print("=" * 80)

    print(
        result.get(
            "final_report",
            "No report generated."
        )
    )

    # ------------------------------------------
    # Additional debugging information
    # ------------------------------------------

    print("\n")
    print("=" * 80)
    print("WORKFLOW SUMMARY")
    print("=" * 80)

    literature = result.get(
        "literature",
        []
    )

    claims = result.get(
        "claims",
        []
    )

    guidelines = result.get(
        "clinical_guidelines",
        []
    )

    print(
        f"Literature articles: "
        f"{len(literature)}"
    )

    print(
        f"Extracted claims: "
        f"{len(claims)}"
    )

    print(
        f"Clinical sources: "
        f"{len(guidelines)}"
    )


if __name__ == "__main__":
    main()
