"""
scrape_rbc_faqs.py
-------------------------------------
Final Playwright-based scraper for Royal Bank of Canada (RBC) FAQ pages.

Key features:
    • Uses verified .accordion-panel DOM structure for FAQs
    • Falls back to standard HTML heading-paragraph FAQ layout
    • Captures dynamic content rendered by JavaScript
    • Cleans, filters, and stores structured Q&A data
    • Skips PDF URLs and logs scraping progress

Outputs:
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


# -------------------------
# CONFIGURATION
# -------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_RAW = BASE_DIR / "data" / "raw" / "rbc"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
LOG_DIR = BASE_DIR / "logs"
URL_FILE = Path(__file__).resolve().parent / "rbc_urls.txt"

DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "scrape_rbc.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# -------------------------
# HELPER FUNCTIONS
# -------------------------
def clean_text(text: str) -> str:
    """Normalize whitespace and remove extra line breaks."""
    return re.sub(r"\s+", " ", text.strip()) if text else ""


def is_valid_faq(question: str, answer: str) -> bool:
    """Filter out generic or noisy entries."""
    if not question or not answer:
        return False
    if len(question) < 10 or len(answer) < 20:
        return False
    bad_words = ["cookie", "privacy", "footer", "email", "sign up", "©"]
    if any(b in (question + answer).lower() for b in bad_words):
        return False
    if re.match(r"^[0-9\s\-\+]+$", answer):
        return False
    return True


def extract_faq_pairs(html: str) -> list:
    """
    Extract FAQ-style question–answer pairs using multiple fallback strategies.
    1️⃣ Primary: RBC accordion structure (.accordion-panel)
    2️⃣ Fallback: heading + paragraph pattern
    3️⃣ Final: full markdown fallback
    """
    soup = BeautifulSoup(html, "html.parser")
    faqs = []

    # Remove scripts, navs, footers
    for tag in soup(["script", "style", "footer", "nav", "header"]):
        tag.decompose()

    # --- PRIMARY STRUCTURE ---
    panels = soup.select(".accordion-panel")
    for item in panels:
        q = item.select_one("button, h2, h3, h4, strong")
        a = item.select_one("p, div")
        if q and a:
            question = clean_text(q.get_text())
            answer = clean_text(a.get_text())
            if is_valid_faq(question, answer):
                faqs.append({"question": question, "answer": answer})

    # --- FALLBACK STRUCTURE ---
    if not faqs:
        question_tags = soup.find_all(["h2", "h3", "dt", "strong"])
        for q in question_tags:
            question = clean_text(q.get_text())
            answer_tag = q.find_next_sibling(["p", "div", "section"])
            if not answer_tag:
                continue
            answer = clean_text(answer_tag.get_text())
            if is_valid_faq(question, answer):
                faqs.append({"question": question, "answer": answer})

    # --- FINAL FALLBACK (markdown) ---
    if not faqs:
        markdown_text = md(str(soup))
        faqs.append({
            "question": "Full Page Content",
            "answer": clean_text(markdown_text)
        })

    # --- Clean + Deduplicate ---
    faqs = [f for f in faqs if 50 < len(f["answer"]) < 2000]
    seen = set()
    unique_faqs = []
    for f in faqs:
        pair = (f["question"], f["answer"])
        if pair not in seen:
            seen.add(pair)
            unique_faqs.append(f)

    return unique_faqs


# -------------------------
# MAIN SCRAPER
# -------------------------
def scrape_rbc_faqs():
    logging.info("Starting RBC FAQ scraping...")
    faq_data, pdf_links = [], []

    with open(URL_FILE, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

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
                faqs = extract_faq_pairs(html)

                for item in faqs:
                    item.update({
                        "url": url,
                        "source": "RBC",
                        "retrieved_at": datetime.now().isoformat()
                    })
                faq_data.extend(faqs)

                logging.info(f"Scraped {len(faqs)} FAQs from {url}")

            except PlaywrightTimeoutError:
                logging.error(f"Timeout on {url}")
                continue
            except Exception as e:
                logging.error(f"Error scraping {url}: {e}")
                continue

        browser.close()

    # Save outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = DATA_RAW / f"{timestamp}_rbc_raw.json"
    processed_path = DATA_PROCESSED / "rbc_faqs.parquet"

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(faq_data, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(faq_data).drop_duplicates(subset=["question", "answer"])
    df.to_parquet(processed_path, index=False)

    logging.info(f"Saved {len(df)} cleaned FAQs to {processed_path}")
    print(f"✅ RBC FAQ scraping completed. Saved {len(df)} entries to {processed_path}")


# -------------------------
# ENTRY POINT
# -------------------------
if __name__ == "__main__":
    scrape_rbc_faqs()
