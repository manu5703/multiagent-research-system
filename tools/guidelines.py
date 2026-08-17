import requests


TRUSTED_GUIDELINE_SOURCES = [

    {
        "organization": "American Diabetes Association",
        "name": "Standards of Care in Diabetes",
        "url": "https://diabetesjournals.org/care/issue",
        "source_type": "Clinical guideline"
    },

    {
        "organization": "National Institute of Diabetes and Digestive and Kidney Diseases",
        "name": "Diabetes Overview",
        "url": "https://www.niddk.nih.gov/health-information/diabetes",
        "source_type": "Government health resource"
    },

    {
        "organization": "U.S. Food and Drug Administration",
        "name": "Diabetes Medicines",
        "url": "https://www.fda.gov/drugs",
        "source_type": "Regulatory source"
    },

    {
        "organization": "Centers for Disease Control and Prevention",
        "name": "Diabetes",
        "url": "https://www.cdc.gov/diabetes/",
        "source_type": "Government health resource"
    }
]


def check_url(url: str) -> bool:
    """
    Check whether a trusted source is reachable.
    """

    try:

        response = requests.head(
            url,
            timeout=10,
            allow_redirects=True
        )

        return response.status_code < 400

    except requests.RequestException:

        return False


def search_guidelines(
    question: str,
    max_results: int = 5
):
    """
    Return trusted guideline sources relevant
    to the research question.

    This version does not scrape guideline content.
    It preserves source provenance and can later
    be extended with official document/API retrieval.
    """

    question_lower = question.lower()

    results = []

    diabetes_related = any(
        keyword in question_lower
        for keyword in [
            "diabetes",
            "hba1c",
            "glucose",
            "glycemic",
            "glucose-lowering"
        ]
    )

    medication_related = any(
        keyword in question_lower
        for keyword in [
            "medication",
            "drug",
            "treatment",
            "therapy"
        ]
    )

    for source in TRUSTED_GUIDELINE_SOURCES:

        if not diabetes_related:
            continue

        if (
            medication_related
            or source["organization"]
            != "U.S. Food and Drug Administration"
        ):

            reachable = check_url(
                source["url"]
            )

            result = source.copy()

            result["reachable"] = reachable

            results.append(result)

    return results[:max_results]