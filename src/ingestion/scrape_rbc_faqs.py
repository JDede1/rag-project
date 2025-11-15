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
    """Normalize whitespace and strip basic boilerplate noise."""
    if not text:
        return ""
    # Normalize common whitespace artifacts
    t = (
        text.replace("\xa0", " ")
        .replace("\u200b", "")
    )
    t = re.sub(r"\s+", " ", t.strip())
    return t


def is_valid_faq(question: str, answer: str) -> bool:
    """Filter out noise, empty content, and non-FAQ elements."""
    if not question or not answer:
        return False

    # Minimal length constraints (keep slightly short answers)
    if len(question.strip()) < 8 or len(answer.strip()) < 15:
        return False

    # Noise patterns – keep this conservative to avoid dropping valid FAQs
    noise = [
        "cookie",
        "privacy",
        "footer",
        "terms and conditions",
        "all rights reserved",
        "javascript",
    ]
    combo = (question + " " + answer).lower()
    if any(n in combo for n in noise):
        return False

    # Reject pure numeric / symbol garbage
    if re.fullmatch(r"[0-9\s\-\+\(\)]+", answer.strip()):
        return False

    return True


# ---------------------------------------------------------
# FAQ EXTRACTION ROUTINES
# ---------------------------------------------------------
def extract_from_accordion(soup: BeautifulSoup):
    """
    Extract using accordion-style markup commonly used in FAQs.

    Uses selectors that match what we already inspect in test_playwright_visual.py:
        .accordion-panel, .accordion-item, .accordion, .accordion-content,
        .panel, .panel-body
    """
    faqs = []
    panels = soup.select(
        ".accordion-panel, .accordion-item, .accordion, "
        ".accordion-content, .panel, .panel-body"
    )

    for p in panels:
        q = p.select_one("button, h2, h3, h4, strong, .accordion-title, .question, .faq-question")
        a = p.select_one("p, div, .accordion-body, .panel-body, .answer, .faq-answer")

        if not q or not a:
            continue

        question = clean_text(q.get_text(separator=" ", strip=True))
        answer = clean_text(a.get_text(separator=" ", strip=True))

        if is_valid_faq(question, answer):
            faqs.append({"question": question, "answer": answer})

    return faqs


def extract_from_faq_blocks(soup: BeautifulSoup):
    """
    Extract FAQs from generic FAQ block structures:
        .faq, .faq-item, .faq-container, .faq-block, .collapse-item
    """
    faqs = []
    containers = soup.select(".faq, .faq-item, .faq-container, .faq-block, .collapse-item")

    for block in containers:
        q = block.select_one(
            "h2, h3, h4, strong, button, .question, .faq-question"
        )
        a = block.select_one(
            "p, div, .answer, .faq-answer, .accordion-content, .panel-body"
        )

        if not q or not a:
            continue

        question = clean_text(q.get_text(separator=" ", strip=True))
        answer = clean_text(a.get_text(separator=" ", strip=True))

        if is_valid_faq(question, answer):
            faqs.append({"question": question, "answer": answer})

    return faqs


def _collect_answer_block(start_tag: Tag) -> str:
    """
    Collect answer text from the siblings following a heading until another heading-like
    element is reached. This allows answers that are wrapped in multiple nested <div>s.
    """
    texts = []
    node = start_tag.next_sibling

    while node:
        # Stop if we hit another heading-like tag (new question)
        if isinstance(node, Tag) and node.name in ["h1", "h2", "h3", "h4", "dt", "strong"]:
            break

        if isinstance(node, Tag):
            node_text = node.get_text(separator=" ", strip=True)
            if node_text:
                texts.append(node_text)

        node = node.next_sibling

    return clean_text(" ".join(texts))


def extract_from_heading_pairs(soup: BeautifulSoup):
    """
    Fallback: any H2/H3/DT/STRONG followed by a descriptive block.
    Uses a deeper sibling walk to capture nested answer blocks.
    """
    faqs = []
    headings = soup.find_all(["h2", "h3", "dt", "strong"])

    for h in headings:
        question = clean_text(h.get_text(separator=" ", strip=True))
        if not question:
            continue

        answer = _collect_answer_block(h)
        if not answer:
            continue

        if is_valid_faq(question, answer):
            faqs.append({"question": question, "answer": answer})

    return faqs


def extract_from_markdown(soup: BeautifulSoup):
    """
    Final fallback: extract entire page as markdown (last resort).

    We cap the length at 1500 characters to avoid triggering validation
    failures on extremely long answers and strip any residual HTML tags.
    """
    markdown_text = md(str(soup))
    # Remove any residual HTML tags if markdownify left some in
    markdown_text = re.sub(r"<[^>]+>", " ", markdown_text)
    markdown_text = clean_text(markdown_text)

    if len(markdown_text) > 1500:
        markdown_text = markdown_text[:1500]

    return [
        {
            "question": "Full Page Content (fallback)",
            "answer": markdown_text,
        }
    ]


def extract_faq_pairs(html: str):
    """Run multi-mode extraction pipeline with fallback tiers."""
    soup = BeautifulSoup(html, "html.parser")

    # Cleanup script/style/nav/header/footer for less noise
    for tag in soup(["script", "style", "footer", "nav", "header"]):
        tag.decompose()

    faqs = []

    # Tier 1: Accordion extraction
    faqs.extend(extract_from_accordion(soup))

    # Tier 2: FAQ block extraction (.faq, .faq-item, etc.)
    more_faqs = extract_from_faq_blocks(soup)
    faqs.extend(more_faqs)

    # Tier 3: Heading-based extraction
    if not faqs:
        faqs = extract_from_heading_pairs(soup)

    # Tier 4: Markdown fallback (only if we still have nothing)
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
def _expand_interactive_elements(page):
    """
    Best-effort expansion of accordions / FAQ items before scraping HTML.

    We only use generic selectors we already rely on in the visual tester:
        button, .accordion-title, .faq-question, .question
    """
    try:
        page.evaluate(
            """
            () => {
                const selectors = ['button', '.accordion-title', '.faq-question', '.question'];
                selectors.forEach(sel => {
                    document.querySelectorAll(sel).forEach(el => {
                        try {
                            el.click();
                        } catch (e) {
                            // ignore click failures
                        }
                    });
                });

                // Scroll through the page to trigger lazy-loaded content
                let totalHeight = 0;
                const distance = 400;
                const timer = setInterval(() => {
                    const scrollHeight = document.body.scrollHeight;
                    window.scrollBy(0, distance);
                    totalHeight += distance;
                    if (totalHeight >= scrollHeight) {
                        clearInterval(timer);
                    }
                }, 200);
            }
            """
        )
    except Exception:
        # Expansion is best-effort; failures should not break scraping
        pass


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

                # Try to expand accordions / FAQ sections
                _expand_interactive_elements(page)
                page.wait_for_timeout(1500)

                html = page.content()
                extracted = extract_faq_pairs(html)

                # Annotate
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
