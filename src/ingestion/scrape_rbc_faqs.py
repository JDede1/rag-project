"""
scrape_rbc_faqs.py
-------------------------------------
Robust Playwright-based scraper for RBC FAQ pages.

Enhancements:
    • Handles accordion, tab, and custom FAQ modules
    • Supports JS-rendered and lazy-loaded pages
    • Uses multiple extraction strategies with fallbacks
    • Stabilized load waits for slow RBC pages
    • Ensures safe directory creation
    • Outputs:
          data/raw/rbc/<timestamp>_rbc_raw.json
          data/processed/rbc_faqs.parquet
          logs/scrape_rbc.log
"""

import re
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
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

# Ensure all directories exist
for d in [DATA_RAW, DATA_PROCESSED, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "scrape_rbc.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------------------------------------------------------
# TEXT CLEANING UTILITIES
# ---------------------------------------------------------
def clean_text(text: str) -> str:
    """Normalize whitespace and filter boilerplate strings."""
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text.strip())
    return t


def is_valid_faq(question: str, answer: str) -> bool:
    """Filter out noise, empty content, and non-FAQ elements."""
    if not question or not answer:
        return False
    if len(question) < 8 or len(answer) < 20:
        return False

    noise = [
        "cookie", "privacy", "footer", "terms", "sign up",
        "©", "javascript", "email", "contact us"
    ]
    combo = (question + answer).lower()
    if any(n in combo for n in noise):
        return False

    # Reject pure numeric garbage
    if re.fullmatch(r"[0-9\s\-\+]+", answer):
        return False

    return True


# ---------------------------------------------------------
# FAQ EXTRACTION ROUTINES
# ---------------------------------------------------------
def extract_from_accordion(soup):
    """Extract using RBC's accordion-style markup."""
    faqs = []
    panels = soup.select(".accordion-panel, .faq-item, .panel, .accordion-content")

    for p in panels:
        q = p.select_one("button, h2, h3, h4, strong, .accordion-title")
        a = p.select_one("p, div, .accordion-body, .panel-body")

        if not q or not a:
            continue

        question = clean_text(q.get_text())
        answer = clean_text(a.get_text())

        if is_valid_faq(question, answer):
            faqs.append({"question": question, "answer": answer})

    return faqs


def extract_from_heading_pairs(soup):
    """Fallback: any H2/H3 followed by a descriptive block."""
    faqs = []
    headings = soup.find_all(["h2", "h3", "dt", "strong"])

    for h in headings:
        question = clean_text(h.get_text())
        a = h.find_next_sibling(["p", "div", "section"])

        if not a:
            continue

        answer = clean_text(a.get_text())

        if is_valid_faq(question, answer):
            faqs.append({"question": question, "answer": answer})

    return faqs


def extract_from_markdown(soup):
    """Final fallback: extract entire page as markdown (last resort)."""
    markdown_text = md(str(soup))
    return [{
        "question": "Full Page Content",
        "answer": clean_text(markdown_text)
    }]


def extract_faq_pairs(html: str):
    """Run multi-mode extraction pipeline with fallback tiers."""
    soup = BeautifulSoup(html, "html.parser")

    # Cleanup script/style/nav
    for tag in soup(["script", "style", "footer", "nav", "header"]):
        tag.decompose()

    # Tier 1: Accordion extraction
    faqs = extract_from_accordion(soup)

    # Tier 2: Heading-based extraction
    if not faqs:
        faqs = extract_from_heading_pairs(soup)

    # Tier 3: Markdown fallback
    if not faqs:
        faqs = extract_from_markdown(soup)

    # Deduplicate
    uniq = []
    seen = set()
    for f in faqs:
        key = (f["question"], f["answer"])
        if key not in seen:
            seen.add(key)
            uniq.append(f)

    return uniq


# ---------------------------------------------------------
# SCRAPER ENGINE
# ---------------------------------------------------------
def scrape_rbc_faqs():
    logging.info("Starting RBC FAQ scraping...")

    with open(URL_FILE, "r") as f:
        urls = [u.strip() for u in f if u.strip()]

    faq_data = []
    pdf_links = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for url in tqdm(urls, desc="Scraping RBC FAQ pages"):
            if url.endswith(".pdf"):
                logging.warning(f"Skipped PDF: {url}")
                pdf_links.append(url)
                continue

            try:
                # Load with extended wait for JS-heavy pages
                page.goto(url, timeout=70000)
                page.wait_for_load_state("domcontentloaded")
                page.wait_for_load_state("networkidle")
                page.wait_for_timeout(1500)

                html = page.content()
                extracted = extract_faq_pairs(html)

                # Annotate
                for item in extracted:
                    item.update({
                        "url": url,
                        "source": "RBC",
                        "retrieved_at": datetime.now().isoformat(),
                    })

                faq_data.extend(extracted)
                logging.info(f"Scraped {len(extracted)} items from {url}")

            except PlaywrightTimeoutError:
                logging.error(f"Timeout on {url}")
            except Exception as e:
                logging.error(f"Error scraping {url}: {e}")

        browser.close()

    # Save data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = DATA_RAW / f"{timestamp}_rbc_raw.json"
    processed_path = DATA_PROCESSED / "rbc_faqs.parquet"

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(faq_data, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(faq_data).drop_duplicates(subset=["question", "answer"])
    df.to_parquet(processed_path, index=False)

    logging.info(f"Saved {len(df)} cleaned FAQs to {processed_path}")
    print(f"Scraping complete: {len(df)} FAQ entries saved.")


if __name__ == "__main__":
    scrape_rbc_faqs()
