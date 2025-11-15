"""
scrape_rbc_faqs.py
-------------------------------------
JSON-driven Playwright scraper for RBC FAQ pages.

Features:
    • Loads all selectors from faq_extraction_patterns.json
    • Extracts FAQs using accordion structures, FAQ blocks, headings
    • Applies robust fallback markdown extraction
    • Expands accordions interactively
    • Handles nested/compound answer blocks
    • Saves raw + processed output
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
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# ---------------------------------------------------------
# LOAD EXTRACTION PATTERNS
# ---------------------------------------------------------
with open(PATTERN_FILE, "r", encoding="utf-8") as f:
    patterns = json.load(f)

selectors = patterns["selectors"]
rules = patterns["extraction"]
interactive = patterns["interactive_expansion"]


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


def is_valid_faq(question: str, answer: str) -> bool:
    """Basic FAQ sanity checker using JSON rules."""
    if not question or not answer:
        return False

    if len(question.strip()) < rules["min_question_length"]:
        return False

    if len(answer.strip()) < rules["min_answer_length"]:
        return False

    combo = (question + " " + answer).lower()
    if any(noise in combo for noise in rules["noise_patterns"]):
        return False

    # numeric garbage
    if re.fullmatch(r"[0-9\s\-\+\(\)]+", answer.strip()):
        return False

    return True


# ---------------------------------------------------------
# EXTRACTION HELPERS
# ---------------------------------------------------------
def parse_containers(soup: BeautifulSoup):
    """Dynamically gather FAQ containers based on JSON selectors."""
    css = (
        selectors["accordion_containers"]
        + selectors["faq_blocks"]
    )
    joined = ", ".join(css)
    return soup.select(joined)


def extract_question(block: Tag):
    """Find the question element using JSON selectors."""
    q_sel = selectors["question_elements"]
    q = block.select_one(", ".join(q_sel))
    return clean_text(q.get_text(" ", strip=True)) if q else None


def extract_answer(block: Tag):
    """Extract answer using JSON selectors."""
    a_sel = selectors["answer_elements"]
    a = block.select_one(", ".join(a_sel))
    return clean_text(a.get_text(" ", strip=True)) if a else None


def deep_collect_answer(start_tag: Tag) -> str:
    """Fallback multi-sibling DOM walk for heading-based answers."""
    texts = []
    node = start_tag.next_sibling

    while node:
        if isinstance(node, Tag) and node.name in selectors["heading_fallbacks"]:
            break

        if isinstance(node, Tag):
            t = node.get_text(" ", strip=True)
            if t:
                texts.append(t)

        node = node.next_sibling

    return clean_text(" ".join(texts))


# ---------------------------------------------------------
# EXTRACTION MODES
# ---------------------------------------------------------
def extract_from_containers(soup: BeautifulSoup):
    """Extract FAQs from accordion + faq block containers."""
    faqs = []
    containers = parse_containers(soup)

    for c in containers:
        q = extract_question(c)
        a = extract_answer(c)
        if q and a and is_valid_faq(q, a):
            faqs.append({"question": q, "answer": a})
    return faqs


def extract_from_headings(soup: BeautifulSoup):
    """Heading-based fallback extraction."""
    faqs = []
    headings = soup.find_all(selectors["heading_fallbacks"])

    for h in headings:
        q = clean_text(h.get_text(" ", strip=True))
        if not q:
            continue

        a = deep_collect_answer(h)
        if not a:
            continue

        if is_valid_faq(q, a):
            faqs.append({"question": q, "answer": a})

    return faqs


def extract_as_markdown(soup: BeautifulSoup):
    """Final fallback extraction."""
    md_text = md(str(soup))
    md_text = re.sub(r"<[^>]+>", " ", md_text)
    md_text = clean_text(md_text)

    if len(md_text) > rules["max_fallback_answer_chars"]:
        md_text = md_text[: rules["max_fallback_answer_chars"]]

    return [{
        "question": "Full Page Content (fallback)",
        "answer": md_text
    }]


def extract_faq_pairs(html: str):
    """Main extraction pipeline."""
    soup = BeautifulSoup(html, "html.parser")

    # strip noisy tags
    for tag in rules["strip_tags"]:
        for el in soup.find_all(tag):
            el.decompose()

    faqs = []

    # Mode 1: containers → most accurate
    faqs.extend(extract_from_containers(soup))

    # Mode 2: heading fallback
    if not faqs:
        faqs.extend(extract_from_headings(soup))

    # Mode 3: last-resort markdown fallback
    if not faqs:
        faqs.extend(extract_as_markdown(soup))

    # Deduplicate
    uniq = []
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
        pass


# ---------------------------------------------------------
# MASTER SCRAPER
# ---------------------------------------------------------
def scrape_rbc_faqs():
    logging.info("Starting RBC FAQ scraping...")

    with open(URL_FILE, "r") as f:
        urls = [u.strip() for u in f if u.strip()]

    faq_data = []
    skipped_pdfs = []

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
                extracted = extract_faq_pairs(html)

                now_iso = datetime.now().isoformat()
                for item in extracted:
                    item.update({
                        "url": url,
                        "source": "RBC",
                        "retrieved_at": now_iso
                    })

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
