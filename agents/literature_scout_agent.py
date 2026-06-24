"""
SkinAnalytica — literature_scout_agent.py
Monitors PubMed + arXiv for new dermoscopy AI papers.
Connects findings to your specific models and metrics.
"""

import os, json, time, urllib.request, urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from base_agent import BaseAgent, BASE

OUT_DIR = os.path.join(BASE, "outputs", "literature")

PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ARXIV_URL         = "http://export.arxiv.org/api/query"

SEARCH_TERMS = [
    "dermoscopy deep learning melanoma",
    "skin lesion classification transformer",
    "ISIC challenge dermoscopy AI",
    "ViT melanoma detection",
    "EfficientNet skin cancer",
    "dermoscopy fairness bias deep learning",
]

class LiteratureScoutAgent(BaseAgent):
    """
    Queries PubMed and arXiv for recent papers.
    Flags papers relevant to SkinAnalytica's models and metrics.
    Runs weekly or on-demand.
    """

    def __init__(self):
        super().__init__("literature_scout_agent")
        os.makedirs(OUT_DIR, exist_ok=True)

    def _search_pubmed(self, query: str, days_back: int = 30, max_results: int = 5) -> list:
        """Search PubMed for recent papers."""
        papers = []
        try:
            since = (datetime.now() - timedelta(days=days_back)).strftime("%Y/%m/%d")
            params = urllib.parse.urlencode({
                "db": "pubmed", "term": query,
                "retmax": max_results, "sort": "date",
                "mindate": since, "retmode": "json",
            })
            req  = urllib.request.Request(f"{PUBMED_SEARCH_URL}?{params}",
                                          headers={"User-Agent": "SkinAnalytica/1.0"})
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read())
            ids  = data.get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []
            # Fetch titles
            fetch_params = urllib.parse.urlencode({
                "db":"pubmed","id": ",".join(ids),
                "rettype":"abstract","retmode":"xml",
            })
            req2 = urllib.request.Request(f"{PUBMED_FETCH_URL}?{fetch_params}",
                                          headers={"User-Agent":"SkinAnalytica/1.0"})
            with urllib.request.urlopen(req2, timeout=15) as r:
                xml_data = r.read()
            root = ET.fromstring(xml_data)
            for article in root.iter("PubmedArticle"):
                try:
                    pmid    = article.findtext(".//PMID", "")
                    title   = article.findtext(".//ArticleTitle", "")
                    journal = article.findtext(".//Journal/Title", "")
                    year    = article.findtext(".//PubDate/Year", "")
                    abstract= article.findtext(".//AbstractText", "")[:300] if article.findtext(".//AbstractText") else ""
                    papers.append({
                        "source" : "PubMed",
                        "pmid"   : pmid,
                        "title"  : title,
                        "journal": journal,
                        "year"   : year,
                        "abstract_excerpt": abstract,
                        "url"    : f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        "query"  : query,
                    })
                except: continue
        except Exception as e:
            self.logger.warning(f"PubMed search failed for '{query}': {e}")
        return papers

    def _search_arxiv(self, query: str, max_results: int = 3) -> list:
        """Search arXiv cs.CV and cs.LG."""
        papers = []
        try:
            params = urllib.parse.urlencode({
                "search_query": f"all:{query}",
                "start": 0, "max_results": max_results,
                "sortBy": "submittedDate", "sortOrder": "descending",
            })
            req  = urllib.request.Request(f"{ARXIV_URL}?{params}",
                                          headers={"User-Agent":"SkinAnalytica/1.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                xml_data = r.read()
            ns   = {"atom":"http://www.w3.org/2005/Atom"}
            root = ET.fromstring(xml_data)
            for entry in root.findall("atom:entry", ns):
                title   = entry.findtext("atom:title", "", ns).strip()
                summary = entry.findtext("atom:summary", "", ns)[:300].strip()
                url     = entry.findtext("atom:id", "", ns)
                papers.append({
                    "source"         : "arXiv",
                    "title"          : title,
                    "abstract_excerpt": summary,
                    "url"            : url,
                    "query"          : query,
                })
        except Exception as e:
            self.logger.warning(f"arXiv search failed for '{query}': {e}")
        return papers

    def _relevance_score(self, paper: dict) -> float:
        """Score paper relevance to SkinAnalytica."""
        keywords = [
            "melanoma","vit","vision transformer","efficientnet","convnext",
            "isic","dermoscopy","skin lesion","fairness","calibration",
            "grad-cam","explainability","auc","sensitivity","specificity",
        ]
        text  = (paper.get("title","") + " " + paper.get("abstract_excerpt","")).lower()
        score = sum(1 for kw in keywords if kw in text)
        return round(score / len(keywords), 3)

    def _run(self, days_back: int = 30, max_per_query: int = 3,
             output_name: str = "literature_scan") -> dict:

        self.logger.info(f"Scanning PubMed + arXiv (last {days_back} days)")
        all_papers = []

        for i, query in enumerate(SEARCH_TERMS):
            self.logger.info(f"Query {i+1}/{len(SEARCH_TERMS)}: {query}")
            pubmed_papers = self._search_pubmed(query, days_back, max_per_query)
            arxiv_papers  = self._search_arxiv(query, max_per_query)
            all_papers.extend(pubmed_papers + arxiv_papers)
            time.sleep(0.5)  # Be polite to APIs

        # Deduplicate by title
        seen   = set()
        unique = []
        for p in all_papers:
            t = p["title"].lower()[:60]
            if t not in seen:
                seen.add(t)
                p["relevance"] = self._relevance_score(p)
                unique.append(p)

        # Sort by relevance
        unique.sort(key=lambda x: x["relevance"], reverse=True)
        top_papers = unique[:20]

        result = {
            "days_back"    : days_back,
            "total_found"  : len(all_papers),
            "unique_papers": len(unique),
            "top_papers"   : top_papers,
            "queries"      : SEARCH_TERMS,
        }

        report_path = os.path.join(OUT_DIR, f"{output_name}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\nLITERATURE SCOUT: {output_name}")
        print("=" * 55)
        print(f"  Searched    : {len(SEARCH_TERMS)} queries")
        print(f"  Found       : {len(unique)} unique papers")
        print(f"\n  Top 5 most relevant:")
        for i, p in enumerate(top_papers[:5], 1):
            print(f"    {i}. [{p['source']}] {p['title'][:70]}")
            print(f"       Relevance: {p['relevance']:.2f}  URL: {p['url']}")
        print(f"\n  Full report : {report_path}")

        return result


if __name__ == "__main__":
    agent  = LiteratureScoutAgent()
    result = agent.run(days_back=30, output_name="weekly_literature")
