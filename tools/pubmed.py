import os
import ssl
import urllib.request
from urllib.error import URLError
from pathlib import Path

import certifi
from Bio import Entrez
from dotenv import load_dotenv

from config import MAX_PUBMED_RESULTS
load_dotenv()


PUBMED_EMAIL = os.getenv(
    "PUBMED_EMAIL",
    "researcher@example.com"
)

Entrez.email = PUBMED_EMAIL

def configure_ssl():
    """Configure urllib, which Biopython Entrez uses, with a trusted CA bundle."""

    ca_bundle = os.getenv("PUBMED_CA_BUNDLE") or os.getenv("SSL_CERT_FILE")

    if ca_bundle:
        ca_path = Path(ca_bundle).expanduser()

        if not ca_path.is_file():
            raise RuntimeError(
                "The configured PubMed CA bundle does not exist: "
                f"{ca_path}"
            )

        context = ssl.create_default_context(cafile=str(ca_path))
    else:
        context = ssl.create_default_context(cafile=certifi.where())

    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=context)
    )
    urllib.request.install_opener(opener)

configure_ssl()

def _entrez_request(operation, **kwargs):
    try:
        return operation(**kwargs)
    except URLError as error:
        reason = str(error.reason)

        if "CERTIFICATE_VERIFY_FAILED" in reason:
            raise RuntimeError(
                "PubMed's TLS certificate could not be verified. "
                "If your network uses a corporate proxy, set "
                "PUBMED_CA_BUNDLE to its root CA PEM file (or set "
                "SSL_CERT_FILE) and run again."
            ) from error

        raise


def search_pubmed(
    query: str,
    max_results: int = None
):
    if max_results is None:
        max_results = MAX_PUBMED_RESULTS

    handle = _entrez_request(
        Entrez.esearch,
        db="pubmed",
        term=query,
        retmax=max_results,
        sort="relevance"
    )

    record = Entrez.read(handle)

    handle.close()

    return [
        str(x)
        for x in record["IdList"]
    ]

def fetch_pubmed_articles(pmids):
    """
    Retrieve article metadata and abstracts from PubMed.
    """

    if not pmids:
        return []

    handle = _entrez_request(
        Entrez.efetch,
        db="pubmed",
        id=",".join(pmids),
        rettype="medline",
        retmode="xml"
    )

    records = Entrez.read(handle)
    handle.close()
    articles = []

    for article in records["PubmedArticle"]:

        citation = article["MedlineCitation"]
        article_data = citation["Article"]

        pmid = str(citation["PMID"])

        title = str(
            article_data.get(
                "ArticleTitle",
                ""
            )
        )

        abstract = ""

        if "Abstract" in article_data:

            abstract_parts = []

            for item in article_data["Abstract"]["AbstractText"]:

                text = str(item)

                label = getattr(
                    item,
                    "attributes",
                    {}
                ).get("Label")

                if label:
                    text = f"{label}: {text}"

                abstract_parts.append(text)

            abstract = " ".join(
                abstract_parts
            )

        # Journal

        journal = ""

        if "Journal" in article_data:

            journal = str(
                article_data["Journal"].get(
                    "Title",
                    ""
                )
            )
        # Publication year

        year = ""

        try:

            year = str(
                article_data[
                    "Journal"
                ][
                    "JournalIssue"
                ][
                    "PubDate"
                ].get(
                    "Year",
                    ""
                )
            )

        except Exception:
            year = ""

        # DOI
        doi = ""
        try:
            article_ids = article.get(
                "PubmedData",
                {}
            ).get(
                "ArticleIdList",
                []
            )

            for article_id in article_ids:

                if article_id.attributes.get(
                    "IdType"
                ) == "doi":

                    doi = str(article_id)

        except Exception:
            doi = ""

        articles.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "year": year,
            "doi": doi,
            "source": (
                f"https://pubmed.ncbi.nlm.nih.gov/"
                f"{pmid}/"
            )
        })

    return articles
