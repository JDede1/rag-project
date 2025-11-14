"""
scrape_rbc_faqs.py
-------------------------------------
Playwright-based scraper for Royal Bank of Canada (RBC) FAQ pages.

Features:
    • Uses .accordion-panel DOM structure where available
    • Falls back to heading + paragraph layout
    • Handles JavaScript-rendered content
    • Cleans and filters Q/A pairs
    • Ensures deterministic directory creation
    • Produces:
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
# CONFIGURATION
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_RAW = BASE_DIR / "data" / "raw" / "rbc"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
LOG_DIR = BASE_DIR / "logs"
URL_FILE = Path(__file__).resolve().parent / "rbc_urls.txt"

# Robust directory creation (prevents mkdir recursion errors)
for d in [DATA_PROCESSED, LOG_DIR, DATA_RAW]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "scrape_rbc.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def clean_text(text: str) -> str:
    """Normalize whitespace and remove redundant spacing."""
    return re.sub(r"\s+", " ", text.strip()) if text else ""


def is_valid_faq(question: str, answer: str) -> bool:
    """Filter out extremely short or noisy entries."""
    if not question or not answer:
        return False
    if len(question) < 10 or len(answer) < 20:
        return False

    noise = ["cookie", "privacy", "footer", "email", "sign up", "©"]
    if any(n in (question + answer).lower() for n in noise):
        return False

    if re.match(r"^[0-9\s\-\+]+$", answer):
        return False

    return True


def extract_faq_pairs(html: str) -> list:
    """
    Extract Q/A pairs using three fallback strategies:
        1. Accordion panel (.accordion-panel)
        2. Headings followed by paragraphs
        3. Markdown fallback (entire page)
    """
    soup = BeautifulSoup(html, "html.parser")
    faqs = []

    # Remove irrelevant tags
    for tag in soup(["script", "style", "footer", "nav", "header"]):
        tag.decompose()

    # Primary: accordion panels
    panels = soup.select(".accordion-panel")
    for item in panels:
        q = item.select_one("button, h2, h3, h4, strong")
        a = item.select_one("p, div")
        if q and a:
            question = clean_text(q.get_text())
            answer = clean_text(a.get_text())
            if is_valid_faq(question, answer):
                faqs.append({"question": question, "answer": answer})

    # Fallback: headings + paragraphs
    if not faqs:
        for q in soup.find_all(["h2", "h3", "dt", "strong"]):
            question = clean_text(q.get_text())
            a = q.find_next_sibling(["p", "div", "section"])
            if not a:
                continue
            answer = clean_text(a.get_text())
            if is_valid_faq(question, answer):
                faqs.append({"question": question, "answer": answer})

    # Final fallback: markdown entire page
    if not faqs:
        markdown_text = md(str(soup))
        faqs.append({
            "question": "Full Page Content",
            "answer": clean_text(markdown_text)
        })

    # Deduplicate
    faqs = [f for f in faqs if 50 < len(f["answer"]) < 2000]
    seen = set()
    unique = []
    for f in faqs:
        key = (f["question"], f["answer"])
        if key not in seen:
            seen.add(key)
            unique.append(f)

    return unique


# ---------------------------------------------------------
# MAIN SCRAPER
# ---------------------------------------------------------
def scrape_rbc_faqs():
    logging.info("Starting RBC FAQ scraping...")
    faq_data = []
    pdf_links = []

    with open(URL_FILE, "r") as f:
        urls = [u.strip() for u in f if u.strip()]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for url in tqdm(urls, desc="Scraping RBC FAQ pages"):
            if url.endswith(".pdf"):
                pdf_links.append(url)
                logging.warning(f"Skipped PDF: {url}")
                continue

            try:
                page.goto(url, timeout=60000)
                page.wait_for_load_state("domcontentloaded")
                page.wait_for_timeout(2000)

                html = page.content()
                extracted = extract_faq_pairs(html)

                for item in extracted:
                    item.update({
                        "url": url,
                        "source": "RBC",
                        "retrieved_at": datetime.now().isoformat()
                    })

                faq_data.extend(extracted)
                logging.info(f"Scraped {len(extracted)} FAQs from {url}")

            except PlaywrightTimeoutError:
                logging.error(f"Timeout on {url}")
            except Exception as e:
                logging.error(f"Error scraping {url}: {e}")

        browser.close()

    # Write output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = DATA_RAW / f"{timestamp}_rbc_raw.json"
    processed_path = DATA_PROCESSED / "rbc_faqs.parquet"

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(faq_data, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(faq_data).drop_duplicates(subset=["question", "answer"])
    df.to_parquet(processed_path, index=False)

    logging.info(f"Saved {len(df)} cleaned FAQs to {processed_path}")
    print(f"RBC FAQ scraping completed. Saved {len(df)} entries to {processed_path}")


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    scrape_rbc_faqs()
