import json
import urllib.parse
import xml.etree.ElementTree as ET
import httpx
from duckduckgo_search import DDGS
from mcp.server.fastmcp import FastMCP

# Initialize Official MCP Server using FastMCP
mcp = FastMCP("ResearchDecisionTools")


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the live web using DuckDuckGo (free, no API key required).
    Returns JSON list of results with title, url, and snippet.
    """
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
            )
    return json.dumps(results, ensure_ascii=False)


@mcp.tool()
def arxiv_search(query: str, max_results: int = 4) -> str:
    """
    Search academic papers on arXiv via official public API (free, no key).
    Returns JSON list of papers with title, authors, published date, url, and abstract.
    """
    encoded = urllib.parse.quote(query)
    url = f"http://export.arxiv.org/api/query?search_query=all:{encoded}&start=0&max_results={max_results}"
    resp = httpx.get(url, timeout=15.0)
    resp.raise_for_status()

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)
    papers = []
    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip().replace("\n", " ")
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip().replace("\n", " ")
        link = entry.findtext("atom:id", default="", namespaces=ns) or ""
        published = entry.findtext("atom:published", default="", namespaces=ns) or ""
        authors = [
            a.findtext("atom:name", default="", namespaces=ns)
            for a in entry.findall("atom:author", ns)
        ]
        papers.append(
            {
                "title": title,
                "authors": authors[:4],
                "published": published[:10],
                "url": link,
                "snippet": summary[:600],
            }
        )
    return json.dumps(papers, ensure_ascii=False)


@mcp.tool()
def wikipedia_summary(topic: str) -> str:
    """
    Fetch concise factual summary and canonical URL from Wikipedia REST API (free).
    """
    slug = urllib.parse.quote(topic.strip().replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
    headers = {"User-Agent": "AgenticResearchSystem/1.0"}
    resp = httpx.get(url, headers=headers, timeout=10.0, follow_redirects=True)
    if resp.status_code != 200:
        return json.dumps({"error": f"Topic '{topic}' not found (HTTP {resp.status_code})"})
    data = resp.json()
    return json.dumps(
        {
            "title": data.get("title", topic),
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", url),
            "snippet": data.get("extract", ""),
        },
        ensure_ascii=False,
    )


@mcp.tool()
def decision_matrix_scorer(options_json: str, criteria_weights_json: str) -> str:
    """
    Deterministic multi-criteria decision analysis (MCDA) calculator.
    options_json: JSON dict mapping option_name -> {criterion: score_0_to_10}
    criteria_weights_json: JSON dict mapping criterion -> weight (float)
    """
    options = json.loads(options_json)
    weights = json.loads(criteria_weights_json)
    total_w = sum(weights.values()) or 1.0
    norm_weights = {k: v / total_w for k, v in weights.items()}

    ranked = []
    for opt_name, scores in options.items():
        weighted_score = sum(
            float(scores.get(crit, 0.0)) * w for crit, w in norm_weights.items()
        )
        ranked.append({"option": opt_name, "weighted_score": round(weighted_score, 3)})

    ranked.sort(key=lambda x: x["weighted_score"], reverse=True)
    return json.dumps({"ranking": ranked, "normalized_weights": norm_weights})


if __name__ == "__main__":
    mcp.run(transport="stdio")
