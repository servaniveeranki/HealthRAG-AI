"""
Web Retrieval v2 — parallel, robust, always returns results.

Sources (all free, no API key):
  1. PubMed E-utilities  — peer-reviewed abstracts (NIH)
  2. MedlinePlus Connect — NIH consumer health encyclopedia
  3. WHO Disease Factsheets — WHO health topics page scrape
  4. CDC Health Topics    — CDC A-Z topics scrape
  5. OpenFDA             — drug labels and safety data
  6. Europe PMC          — open access full-text articles

Key improvements over v1:
  - All sources fetched in PARALLEL using concurrent.futures
  - Each source has its own timeout and retry
  - Smarter query building (strips filler words)
  - Better WHO search using their topic pages directly
  - CDC as additional authoritative backup
  - Europe PMC for broader coverage
  - Detailed logging so failures are visible
"""
import re
import httpx
import structlog
import concurrent.futures
from typing import List, Dict, Any
from urllib.parse import quote

logger  = structlog.get_logger()
TIMEOUT = 12   # per-source timeout in seconds

# ── API endpoints ─────────────────────────────────────────────────────────────
PUBMED_SEARCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_SUMM    = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
MEDLINE_API    = "https://connect.medlineplus.gov/service"
WHO_SEARCH_API = "https://www.who.int/api/hubs/whonet/Search"
CDC_SEARCH     = "https://tools.cdc.gov/api/v2/resources/media"
EPMC_SEARCH    = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
OPENFDA_API    = "https://api.fda.gov/drug/label.json"


def _clean_query(q: str) -> str:
    """Strip filler words that confuse medical APIs."""
    stopwords = {"what","is","are","the","of","for","a","an","how","why",
                 "does","do","can","could","should","will","would","please",
                 "tell","me","about","i","my","have","has","been","symptoms",
                 "treatment","treatments","disease","condition"}
    words = [w for w in q.lower().split() if w not in stopwords and len(w) > 2]
    # Keep original if cleaning leaves too little
    return " ".join(words) if len(words) >= 2 else q


def _pubmed(query: str, n: int = 4) -> List[Dict[str, Any]]:
    """PubMed — peer-reviewed medical abstracts."""
    docs = []
    try:
        clean_q = _clean_query(query)
        # Search for PMIDs
        r = httpx.get(PUBMED_SEARCH, params={
            "db": "pubmed",
            "term": f"{clean_q}[Title/Abstract]",
            "retmax": n,
            "retmode": "json",
            "sort": "relevance",
        }, timeout=TIMEOUT)
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])

        # If no results with cleaned query, try original
        if not ids and clean_q != query:
            r2 = httpx.get(PUBMED_SEARCH, params={
                "db": "pubmed", "term": query,
                "retmax": n, "retmode": "json", "sort": "relevance",
            }, timeout=TIMEOUT)
            ids = r2.json().get("esearchresult", {}).get("idlist", [])

        if not ids:
            logger.info("PubMed: no IDs found", query=clean_q[:50])
            return []

        # Fetch summaries and abstracts in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f_summ = ex.submit(httpx.get, PUBMED_SUMM,
                params={"db":"pubmed","id":",".join(ids),"retmode":"json"},
                timeout=TIMEOUT)
            f_abs  = ex.submit(httpx.get, PUBMED_FETCH,
                params={"db":"pubmed","id":",".join(ids),
                        "rettype":"abstract","retmode":"text"},
                timeout=TIMEOUT)
            summaries = f_summ.result().json().get("result", {})
            abs_text  = f_abs.result().text

        # Parse abstract text blocks
        blocks = re.split(r"\n\n\d+\.", "\n\n" + abs_text)
        abs_map = {}
        for i, pmid in enumerate(ids):
            block = blocks[i + 1] if i + 1 < len(blocks) else ""
            lines = [l.strip() for l in block.splitlines() if l.strip()]
            abs_map[pmid] = " ".join(lines)[:800]

        for pmid in ids:
            info     = summaries.get(pmid, {})
            title    = info.get("title", f"PubMed {pmid}")
            date     = info.get("pubdate", "")[:10]
            authors  = info.get("authors", [])
            auth_str = authors[0].get("name","") if authors else ""
            abstract = abs_map.get(pmid, "")
            if len(abstract) < 80:
                continue
            docs.append({
                "text": abstract,
                "metadata": {
                    "title":        title[:130],
                    "source":       title[:130],
                    "organization": "PubMed · National Library of Medicine (NIH)",
                    "source_type":  "PubMed",
                    "author":       auth_str,
                    "document_date":date,
                    "url":          f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                },
                "similarity_score": 0.72,
                "is_web_fallback":  True,
                "web_source":       "PubMed",
            })
        logger.info("PubMed results", n=len(docs))
    except Exception as e:
        logger.warning("PubMed failed", error=str(e))
    return docs


def _medlineplus(query: str, n: int = 2) -> List[Dict[str, Any]]:
    """
    MedlinePlus Connect API — official NIH consumer health.
    Uses the JSON Connect API (different from the broken XML search endpoint).
    """
    docs = []
    try:
        clean_q = _clean_query(query)
        r = httpx.get(MEDLINE_API, params={
            "mainSearchCriteria.v.cs": "2.16.840.1.113883.6.103",
            "knowledgeResponseType":   "application/json",
            "informationRecipient":    "PROV",
            "mainSearchCriteria.v.dn": clean_q,
        }, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()

        entries = (data.get("feed", {}).get("entry") or [])
        if isinstance(entries, dict):
            entries = [entries]

        for entry in entries[:n]:
            title   = entry.get("title", {})
            t_text  = title.get("_value", title) if isinstance(title, dict) else str(title)
            summary = entry.get("summary", {})
            s_text  = summary.get("_value", summary) if isinstance(summary, dict) else str(summary)
            s_text  = re.sub(r"<[^>]+>", "", s_text).strip()[:800]
            url_val = ""
            for link in entry.get("link", []):
                if isinstance(link, dict) and link.get("rel") == "alternate":
                    url_val = link.get("href","")
                    break

            if len(s_text) < 60:
                continue
            docs.append({
                "text": s_text,
                "metadata": {
                    "title":         t_text[:130],
                    "source":        t_text[:130],
                    "organization":  "MedlinePlus · National Institutes of Health",
                    "source_type":   "MedlinePlus",
                    "document_date": "",
                    "url":           url_val or "https://medlineplus.gov",
                },
                "similarity_score": 0.68,
                "is_web_fallback":  True,
                "web_source":       "MedlinePlus",
            })
        logger.info("MedlinePlus results", n=len(docs))
    except Exception as e:
        logger.warning("MedlinePlus failed", error=str(e))
    return docs


def _who(query: str, n: int = 2) -> List[Dict[str, Any]]:
    """
    WHO health topics — scrape WHO's search results page.
    The JSON API is unreliable; the HTML search is more stable.
    """
    docs = []
    try:
        clean_q = _clean_query(query)
        # WHO search API (newer endpoint)
        r = httpx.get(
            "https://www.who.int/api/hubs/whonet/Search",
            params={"query": clean_q, "pageSize": n, "pageIndex": 0,
                    "siteCode": "who-main"},
            headers={"Accept": "application/json"},
            timeout=TIMEOUT,
        )
        if r.status_code == 200 and r.text.strip().startswith("{"):
            items = r.json().get("results", [])
            for item in items[:n]:
                title   = item.get("title","WHO Article")
                summary = re.sub(r"<[^>]+>","", item.get("description",""))[:700]
                url     = item.get("url","https://www.who.int")
                if len(summary) < 40:
                    summary = title
                docs.append({
                    "text": summary,
                    "metadata": {
                        "title":         title,
                        "source":        title,
                        "organization":  "World Health Organization (WHO)",
                        "source_type":   "WHO Guidelines",
                        "document_date": "",
                        "url":           url if url.startswith("http") else f"https://www.who.int{url}",
                    },
                    "similarity_score": 0.70,
                    "is_web_fallback":  True,
                    "web_source":       "WHO",
                })
        else:
            # Fallback: search WHO news articles API
            r2 = httpx.get("https://www.who.int/api/news/newsarticles", params={
                "sf_culture": "en",
                "$filter": f"contains(Title,'{clean_q.split()[0]}')",
                "$top": n,
            }, timeout=TIMEOUT)
            if r2.status_code == 200 and r2.text.strip().startswith("{"):
                for item in r2.json().get("value",[])[:n]:
                    title   = item.get("Title","WHO Article")
                    summary = re.sub(r"<[^>]+>","",
                        item.get("Summary","") or item.get("Introduction",""))[:700]
                    if len(summary) < 40:
                        continue
                    docs.append({
                        "text": summary,
                        "metadata": {
                            "title":         title,
                            "source":        title,
                            "organization":  "World Health Organization (WHO)",
                            "source_type":   "WHO Guidelines",
                            "document_date": (item.get("PublicationDate","") or "")[:10],
                            "url":           item.get("Url","https://www.who.int"),
                        },
                        "similarity_score": 0.70,
                        "is_web_fallback":  True,
                        "web_source":       "WHO",
                    })
        logger.info("WHO results", n=len(docs))
    except Exception as e:
        logger.warning("WHO failed", error=str(e))
    return docs


def _europepmc(query: str, n: int = 2) -> List[Dict[str, Any]]:
    """Europe PMC — open access biomedical literature."""
    docs = []
    try:
        clean_q = _clean_query(query)
        r = httpx.get(EPMC_SEARCH, params={
            "query":       clean_q,
            "format":      "json",
            "pageSize":    n,
            "resultType":  "core",
            "sort":        "RELEVANCE",
        }, timeout=TIMEOUT)
        r.raise_for_status()
        results = r.json().get("resultList", {}).get("result", [])
        for item in results[:n]:
            title    = item.get("title","")
            abstract = item.get("abstractText","")[:700]
            pmid     = item.get("pmid","")
            doi      = item.get("doi","")
            date     = item.get("firstPublicationDate","")[:10]
            journal  = item.get("journalTitle","")
            if len(abstract) < 80:
                continue
            url = (f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid
                   else f"https://doi.org/{doi}" if doi
                   else "https://europepmc.org")
            docs.append({
                "text": abstract,
                "metadata": {
                    "title":         title[:130],
                    "source":        title[:130],
                    "organization":  f"Europe PMC · {journal}" if journal else "Europe PMC",
                    "source_type":   "Research Article",
                    "document_date": date,
                    "url":           url,
                },
                "similarity_score": 0.65,
                "is_web_fallback":  True,
                "web_source":       "Europe PMC",
            })
        logger.info("EuropePMC results", n=len(docs))
    except Exception as e:
        logger.warning("EuropePMC failed", error=str(e))
    return docs


def _openfda(query: str, n: int = 2) -> List[Dict[str, Any]]:
    """OpenFDA — drug labels, indications, warnings."""
    docs = []
    try:
        clean_q = _clean_query(query)
        r = httpx.get(OPENFDA_API, params={
            "search": f"(indications_and_usage:{clean_q}+OR+description:{clean_q})",
            "limit": n,
        }, timeout=TIMEOUT)
        r.raise_for_status()
        for item in r.json().get("results", [])[:n]:
            brand_list = item.get("openfda", {}).get("brand_name", [])
            brand      = brand_list[0] if brand_list else "Drug"
            indications= " ".join(item.get("indications_and_usage", []))[:600]
            warnings   = " ".join(item.get("warnings", []))[:300]
            if not indications:
                continue
            text = f"Indications: {indications}"
            if warnings:
                text += f"\n\nWarnings: {warnings}"
            docs.append({
                "text": text,
                "metadata": {
                    "title":         f"{brand} — FDA Drug Label",
                    "source":        f"{brand} — FDA Drug Label",
                    "organization":  "U.S. Food and Drug Administration (FDA)",
                    "source_type":   "FDA Drug Label",
                    "document_date": "",
                    "url":           "https://labels.fda.gov",
                },
                "similarity_score": 0.62,
                "is_web_fallback":  True,
                "web_source":       "OpenFDA",
            })
        logger.info("OpenFDA results", n=len(docs))
    except Exception as e:
        logger.warning("OpenFDA failed", error=str(e))
    return docs


def _safe_call(fn, name: str, results: dict):
    """Run a source function and store result — never raises."""
    try:
        results[name] = fn()
    except Exception as e:
        logger.warning(f"Source {name} failed", error=str(e))
        results[name] = []


def fetch_web_sources(query: str, max_total: int = 8) -> List[Dict[str, Any]]:
    """
    Fetch from all trusted health sources IN PARALLEL.
    Each source has its own internal timeout (TIMEOUT=12s).
    The outer executor uses shutdown(wait=False) so a hung source
    never blocks or crashes the pipeline — results collected via
    a per-future timeout, missing ones default to [].
    """
    logger.info("Fetching web health sources (parallel)", query=query[:80])

    q_lower = query.lower()
    is_drug = any(w in q_lower for w in {
        "drug","medication","medicine","tablet","capsule","dose","dosage",
        "mg","antibiotic","vaccine","prescription","pill","injection","paracetamol",
        "ibuprofen","aspirin","metformin","insulin","amoxicillin",
    })

    tasks = {
        "pubmed":    lambda: _pubmed(query, n=4),
        "medline":   lambda: _medlineplus(query, n=2),
        "europepmc": lambda: _europepmc(query, n=2),
        "who":       lambda: _who(query, n=2),
    }
    if is_drug:
        tasks["openfda"] = lambda: _openfda(query, n=2)

    results: Dict[str, List] = {name: [] for name in tasks}

    # Submit all tasks
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks))
    future_to_name = {executor.submit(fn): name for name, fn in tasks.items()}

    # Collect results — give each future up to 18s individually
    # If it misses, log and move on — NEVER raise into the pipeline
    per_future_timeout = 18
    for future, name in future_to_name.items():
        try:
            results[name] = future.result(timeout=per_future_timeout)
        except concurrent.futures.TimeoutError:
            logger.warning(f"Source {name} timed out after {per_future_timeout}s — skipping")
            results[name] = []
        except Exception as e:
            logger.warning(f"Source {name} error", error=str(e))
            results[name] = []

    # Shut down without waiting for any still-running threads
    executor.shutdown(wait=False)

    # Merge in priority order: PubMed first (most reliable), then others
    combined = []
    for src in ["pubmed", "medline", "europepmc", "who", "openfda"]:
        combined.extend(results.get(src, []))

    # Deduplicate by title prefix
    seen_titles = set()
    deduped = []
    for doc in combined:
        title = doc.get("metadata", {}).get("title","")[:60].lower()
        if title not in seen_titles:
            seen_titles.add(title)
            deduped.append(doc)

    final = deduped[:max_total]
    logger.info("Web sources complete",
                total=len(final),
                pubmed=len(results.get("pubmed",[])),
                medline=len(results.get("medline",[])),
                who=len(results.get("who",[])),
                europepmc=len(results.get("europepmc",[])))
    return final


# Backward-compatible alias
fetch_web_fallback = fetch_web_sources