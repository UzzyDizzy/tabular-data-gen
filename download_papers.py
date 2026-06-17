"""Download all referred papers (arXiv PDFs) into ./referred-papers/.

Source of truth = the literature-survey workflow output (has exact arXiv URLs),
so we never fabricate IDs. A small SUPP map adds plan-only refs not in the survey.
Non-arXiv venues (Nature / PNAS / npj / Transformer Circuits / bioRxiv / ACM) are
listed in NON_ARXIV for manual note; most are paywalled and skipped automatically.

Run:  python download_papers.py
Idempotent: skips files already present (>10 KB).
"""
from __future__ import annotations
import json, os, re, time, urllib.request, urllib.error

SURVEY = (
    r"C:\Users\KIIT0001\AppData\Local\Temp\claude"
    r"\C--Users-KIIT0001-Desktop-gitclones-tabular-data-gen"
    r"\2bf7b968-8cd7-4e8d-ab7f-5fe54fd24100\tasks\wp015u30q.output"
)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "referred-papers")
os.makedirs(OUT, exist_ok=True)

ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", re.I)
ARXIV_COLON_RE = re.compile(r"ar[Xx]iv:\s*(\d{4}\.\d{4,5})")

# Plan-only references that may not appear verbatim in the survey URLs.
SUPP = {
    # name: arxiv_id  (only confident IDs; uncertain ones omitted on purpose)
}

def slug(s: str, n: int = 60) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")
    return s[:n] if s else "paper"

def collect() -> dict[str, str]:
    """Return {arxiv_id: title}. Title best-effort from the survey JSON."""
    raw = open(SURVEY, encoding="utf-8").read()
    ids: dict[str, str] = {}

    # 1) structured pass: map id -> paper title
    try:
        data = json.loads(raw)
        for sub in data.get("result", {}).get("survey", []):
            for p in sub.get("papers", []):
                url = p.get("url", "") or ""
                title = p.get("title", "") or ""
                for m in ARXIV_RE.findall(url):
                    ids.setdefault(m, title)
                # venues sometimes hold the arXiv id as text
                for field in (p.get("venue", ""), p.get("url", "")):
                    for m in ARXIV_COLON_RE.findall(field or ""):
                        ids.setdefault(m, title)
    except Exception as e:  # noqa
        print("WARN json parse:", e)

    # 2) raw regex sweep over the whole text (catches anything missed)
    for m in ARXIV_RE.findall(raw):
        ids.setdefault(m, "")
    for m in ARXIV_COLON_RE.findall(raw):
        ids.setdefault(m, "")

    # 3) supplementary plan-only refs
    for name, aid in SUPP.items():
        ids.setdefault(aid, name)

    return ids

def download(aid: str, title: str) -> str:
    fname = f"{aid}_{slug(title)}.pdf" if title else f"{aid}.pdf"
    dest = os.path.join(OUT, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
        return "skip"
    url = f"https://arxiv.org/pdf/{aid}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research; paper-fetch)"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            blob = r.read()
        if len(blob) < 10_000 or blob[:4] != b"%PDF":
            return f"bad ({len(blob)} bytes)"
        with open(dest, "wb") as f:
            f.write(blob)
        return f"ok ({len(blob)//1024} KB)"
    except urllib.error.HTTPError as e:
        return f"http {e.code}"
    except Exception as e:  # noqa
        return f"err {e}"

def main() -> None:
    ids = collect()
    print(f"Found {len(ids)} unique arXiv IDs -> {OUT}")
    rows = []
    for i, (aid, title) in enumerate(sorted(ids.items()), 1):
        status = download(aid, title)
        print(f"[{i:>3}/{len(ids)}] {aid}  {status}  {title[:70]}")
        rows.append((aid, status, title))
        if status.startswith("ok"):
            time.sleep(3)  # be polite to arXiv
    # index
    with open(os.path.join(OUT, "INDEX.md"), "w", encoding="utf-8") as f:
        f.write("# Referred papers (arXiv)\n\n")
        f.write("| arXiv | status | title |\n|---|---|---|\n")
        for aid, status, title in rows:
            f.write(f"| [{aid}](https://arxiv.org/abs/{aid}) | {status} | {title} |\n")
    ok = sum(1 for _, s, _ in rows if s.startswith(("ok", "skip")))
    print(f"\nDONE: {ok}/{len(rows)} available. Index -> referred-papers/INDEX.md")

if __name__ == "__main__":
    main()
