"""
scrape_rbc_faqs.py
-------------------------------------
JSON-driven Playwright scraper for RBC FAQ pages.

Features:
    • Loads selectors and rules from faq_extraction_patterns.json
    • Supports URL-specific overrides (per-page behavior)
    • Extracts FAQs using accordion structures, FAQ blocks, headings
    • Applies robust fallback markdown extraction (configurable)
    • Expands accordions interactively
    • Handles nested/compound answer blocks
    • Normalizes and deduplicates Q/A pairs
    • Clamps overly long answers to configured max length

Configuration:
    Global selectors / rules / interactive behavior are defined in:
        src/ingestion/faq_extraction_patterns.json

    Optional URL-specific overrides can be added under "overrides":
        {
          "overrides": {
            "https://www.rbcroyalbank.com/....": {
              "use_containers": true,
              "use_headings": true,
              "allow_markdown_fallback": false,
              "min_question_length": 8,
              "min_answer_length": 20,
              "container_selectors": [".some-class", ".other-class"],
              "question_selectors": ["h2", ".question"],
              "answer_selectors": ["p", ".answer"],
              "heading_fallbacks": ["h2", "h3"]
            }
          }
        }

    All keys are optional; unspecified values fall back to global config.
"""

import json
import re
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, Tag
from markdownify import markdownify as md
from tqdm import tqdm
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_RAW = BASE_DIR / "data" / "raw" / "rbc"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
LOG_DIR = BASE_DIR / "logs"
URL_FILE = Path(__file__).resolve().parent / "rbc_urls.txt"
PATTERN_FILE = Path(__file__).resolve().parent / "faq_extraction_patterns.json"

# Ensure directories exist
for d in [DATA_RAW, DATA_PROCESSED, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "scrape_rbc.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------------------------------------------------------
# LOAD EXTRACTION PATTERNS
# ---------------------------------------------------------
with open(PATTERN_FILE, "r", encoding="utf-8") as f:
    patterns = json.load(f)

selectors = patterns["selectors"]
rules = patterns["extraction"]
interactive = patterns["interactive_expansion"]
overrides = patterns.get("overrides", {})


def get_override(url: str) -> dict:
    """Return URL-specific override dict (may be empty)."""
    return overrides.get(url, {})


# ---------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------
def clean_text(text: str) -> str:
    """Normalize whitespace and strip noise characters."""
    if not text:
        return ""

    t = (
        text.replace("\xa0", " ")
        .replace("\u200b", "")
    )
    t = re.sub(r"\s+", " ", t.strip())
    return t


def normalize_question_answer(question: str, answer: str) -> tuple[str, str]:
    """
    Normalize Q/A pairs:
        • Clean whitespace
        • Strip question text from beginning of answer if duplicated
        • Clamp overly long answers to max_fallback_answer_chars
    """
    q = clean_text(question)
    a = clean_text(answer)

    # Strip duplicated question at the start of the answer
    if q and a and a.lower().startswith(q.lower()):
        a = a[len(q) :].lstrip(" :.-")

    # Clamp answer length to configured limit
    max_len = rules["max_fallback_answer_chars"]
    if len(a) > max_len:
        a = a[:max_len].strip()

    return q, a


def is_valid_faq(question: str, answer: str, url: str | None = None) -> bool:
    """
    Basic FAQ sanity checker using JSON rules + optional URL overrides.
    """
    if not question or not answer:
        return False

    override = get_override(url) if url else {}
    min_q_len = override.get("min_question_length", rules["min_question_length"])
    min_a_len = override.get("min_answer_length", rules["min_answer_length"])

    if len(question.strip()) < min_q_len:
        return False

    if len(answer.strip()) < min_a_len:
        return False

    combo = (question + " " + answer).lower()
    if any(noise in combo for noise in rules["noise_patterns"]):
        return False

    # numeric garbage
    if re.fullmatch(r"[0-9\s\-\+\(\)]+", answer.strip()):
        return False

    return True


# ---------------------------------------------------------
# EXTRACTION HELPERS (URL-AWARE)
# ---------------------------------------------------------
def parse_containers(soup: BeautifulSoup, url: str):
    """
    Dynamically gather FAQ containers based on global selectors
    or URL-specific overrides.
    """
    override = get_override(url)
    custom_containers = override.get("container_selectors")

    if custom_containers:
        css = custom_containers
    else:
        css = selectors["accordion_containers"] + selectors["faq_blocks"]

    joined = ", ".join(css)
    return soup.select(joined)


def extract_question(block: Tag, url: str):
    """
    Find the question element using global or URL-specific question selectors.
    """
    override = get_override(url)
    q_sel = override.get("question_selectors", selectors["question_elements"])
    q = block.select_one(", ".join(q_sel))
    return clean_text(q.get_text(" ", strip=True)) if q else None


def extract_answer(block: Tag, url: str):
    """
    Extract answer text using global or URL-specific answer selectors.
    """
    override = get_override(url)
    a_sel = override.get("answer_selectors", selectors["answer_elements"])
    a = block.select_one(", ".join(a_sel))
    return clean_text(a.get_text(" ", strip=True)) if a else None


def deep_collect_answer(start_tag: Tag, stop_tags: list[str]) -> str:
    """
    Collect answer text from siblings following a heading, stopping at
    the next heading-like tag.
    """
    texts = []
    node = start_tag.next_sibling

    while node:
        if isinstance(node, Tag) and node.name in stop_tags:
            break

        if isinstance(node, Tag):
            t = node.get_text(" ", strip=True)
            if t:
                texts.append(t)

        node = node.next_sibling

    return clean_text(" ".join(texts))


# ---------------------------------------------------------
# EXTRACTION MODES (URL-AWARE)
# ---------------------------------------------------------
def extract_from_containers(soup: BeautifulSoup, url: str):
    """Extract FAQs from accordion + FAQ block containers."""
    faqs = []
    containers = parse_containers(soup, url)

    for c in containers:
        raw_q = extract_question(c, url)
        raw_a = extract_answer(c, url)
        if not raw_q or not raw_a:
            continue

        q, a = normalize_question_answer(raw_q, raw_a)

        if is_valid_faq(q, a, url=url):
            faqs.append({"question": q, "answer": a})

    return faqs


def extract_from_headings(soup: BeautifulSoup, url: str):
    """Heading-based fallback extraction."""
    faqs = []
    override = get_override(url)
    heading_tags = override.get("heading_fallbacks", selectors["heading_fallbacks"])

    headings = soup.find_all(heading_tags)

    for h in headings:
        raw_q = h.get_text(" ", strip=True)
        if not raw_q:
            continue

        raw_a = deep_collect_answer(h, heading_tags)
        if not raw_a:
            continue

        q, a = normalize_question_answer(raw_q, raw_a)

        if is_valid_faq(q, a, url=url):
            faqs.append({"question": q, "answer": a})

    return faqs


def extract_as_markdown(soup: BeautifulSoup, url: str):
    """
    Final fallback extraction: full-page markdown.

    This is controlled by the URL override flag:
        allow_markdown_fallback (default: True)
    """
    md_text = md(str(soup))
    md_text = re.sub(r"<[^>]+>", " ", md_text)
    md_text = clean_text(md_text)

    # Clamp length using the same logic as normalize_question_answer
    max_len = rules["max_fallback_answer_chars"]
    if len(md_text) > max_len:
        md_text = md_text[:max_len].strip()

    q, a = normalize_question_answer("Full Page Content (fallback)", md_text)

    return [{"question": q, "answer": a}]


def extract_faq_pairs(html: str, url: str):
    """Main extraction pipeline (URL-aware)."""
    soup = BeautifulSoup(html, "html.parser")

    # strip noisy tags globally
    for tag_name in rules["strip_tags"]:
        for el in soup.find_all(tag_name):
            el.decompose()

    override = get_override(url)
    use_containers = override.get("use_containers", True)
    use_headings = override.get("use_headings", True)
    allow_markdown_fallback = override.get("allow_markdown_fallback", True)

    faqs: list[dict] = []

    # Mode 1: container-based extraction
    if use_containers:
        faqs.extend(extract_from_containers(soup, url))

    # Mode 2: heading-based extraction
    if use_headings and not faqs:
        faqs.extend(extract_from_headings(soup, url))

    # Mode 3: markdown fallback (optional)
    if allow_markdown_fallback and not faqs:
        faqs.extend(extract_as_markdown(soup, url))

    # Deduplicate by (question, answer)
    uniq: list[dict] = []
    seen = set()
    for f in faqs:
        key = (f["question"], f["answer"])
        if key not in seen:
            uniq.append(f)
            seen.add(key)

    return uniq


# ---------------------------------------------------------
# PAGE INTERACTION (ACCORDION EXPANSION)
# ---------------------------------------------------------
def expand_interactive(page):
    """Click accordion triggers + scroll to load lazy content."""
    try:
        click_js = f"""
        () => {{
            const sels = {interactive["click_selectors"]};
            sels.forEach(sel => {{
                document.querySelectorAll(sel).forEach(el => {{
                    try {{ el.click(); }} catch {{ }}
                }});
            }});
        }}
        """
        page.evaluate(click_js)

        # Scroll to trigger lazy-loading
        distance = interactive["scroll_distance"]
        steps = interactive["scroll_steps"]
        delay = interactive["scroll_delay_ms"]

        for _ in range(steps):
            page.evaluate(f"window.scrollBy(0, {distance});")
            page.wait_for_timeout(delay)

    except Exception:
        # Expansion is best-effort; failures should not break scraping
        pass


# ---------------------------------------------------------
# MASTER SCRAPER
# ---------------------------------------------------------
def scrape_rbc_faqs():
    logging.info("Starting RBC FAQ scraping...")

    with open(URL_FILE, "r") as f:
        urls = [u.strip() for u in f if u.strip()]

    faq_data: list[dict] = []
    skipped_pdfs: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()

        for url in tqdm(urls, desc="Scraping RBC FAQ pages"):
            if url.endswith(".pdf"):
                skipped_pdfs.append(url)
                logging.warning(f"Skipped PDF: {url}")
                continue

            try:
                page.goto(url, timeout=70000)
                page.wait_for_load_state("domcontentloaded")
                page.wait_for_load_state("networkidle")
                page.wait_for_timeout(1500)

                expand_interactive(page)
                page.wait_for_timeout(1200)

                html = page.content()
                extracted = extract_faq_pairs(html, url)

                now_iso = datetime.now().isoformat()
                for item in extracted:
                    item.update(
                        {
                            "url": url,
                            "source": "RBC",
                            "retrieved_at": now_iso,
                        }
                    )

                faq_data.extend(extracted)
                logging.info(f"Scraped {len(extracted)} items from {url}")

            except PlaywrightTimeoutError:
                logging.error(f"Timeout on {url}")
            except Exception as e:
                logging.error(f"Error scraping {url}: {e}")

        browser.close()

    # Save raw + processed
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = DATA_RAW / f"{ts}_rbc_raw.json"
    proc_path = DATA_PROCESSED / "rbc_faqs.parquet"

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(faq_data, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(faq_data).drop_duplicates(subset=["question", "answer"])
    df.to_parquet(proc_path, index=False)

    logging.info(f"Saved {len(df)} final FAQ entries → {proc_path}")
    print(f"Scraping complete: {len(df)} FAQ entries saved.")


if __name__ == "__main__":
    scrape_rbc_faqs()
