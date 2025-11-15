"""
test_playwright_visual.py
-------------------------------------
Interactive Playwright helper to visually debug RBC FAQ pages.

Improvements:
    • Accepts dynamic URL (CLI arg or default)
    • Highlights all FAQ-related selectors with visual CSS borders
    • Expands accordions, FAQ blocks, and custom RBC toggle elements
    • Logs all discovered DOM classes containing 'accordion' or 'faq'
    • Saves:
          - DOM class report
          - HTML snapshot
          - pre- and post-expansion screenshots
    • Extracts Q/A previews using the same selector strategy used in scraper
    • Provides a stable toolkit for debugging problematic RBC pages
"""

import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

# --------------------------------------------------------
# CONFIG
# --------------------------------------------------------

# Default URL if none is supplied
DEFAULT_URL = (
    "https://www.rbcroyalbank.com/credit-cards/cardholders/"
    "frequently-asked-questions/general-questions.html"
)

# Reuse only selectors already known in your repo (no hallucinations)
FAQ_SELECTORS = [
    ".accordion-panel",
    ".accordion-item",
    ".accordion",
    ".accordion-content",
    ".faq",
    ".faq-item",
    ".faq-container",
    ".faq-block",
    ".panel",
    ".panel-body",
    ".collapse-item",
    ".question",
    ".answer",
    "button",
]

DEBUG_DIR = Path("logs/playwright_debug")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------
# UTILITY HELPERS
# --------------------------------------------------------

def save_screenshot(page, name):
    """Save PNG screenshot to logs/playwright_debug directory."""
    out_path = DEBUG_DIR / f"{name}.png"
    page.screenshot(path=str(out_path), full_page=True)
    print(f"[Saved] Screenshot → {out_path}")


def save_html(page, name):
    """Save HTML snapshot for offline inspection."""
    out_path = DEBUG_DIR / f"{name}.html"
    html = page.content()
    out_path.write_text(html, encoding="utf-8")
    print(f"[Saved] HTML snapshot → {out_path}")


def save_class_report(class_list, name="dom_classes.txt"):
    """Write discovered classes with 'accordion' or 'faq' substrings."""
    out_path = DEBUG_DIR / name
    out_path.write_text("\n".join(class_list), encoding="utf-8")
    print(f"[Saved] DOM class report → {out_path}")


# --------------------------------------------------------
# VISUAL TEST RUNNER
# --------------------------------------------------------

def run_visual_test(url: str):
    print("Launching interactive Chromium browser...")
    print(f"Loading URL: {url}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=250)
        page = browser.new_page()
        page.goto(url, timeout=60000)

        page.wait_for_load_state("domcontentloaded")
        page.wait_for_timeout(1500)

        # Save initial state
        save_screenshot(page, "01_initial")
        save_html(page, "01_initial")

        print("\nHighlighting FAQ-related elements...\n")

        # ------------------------------------------------------------
        # Highlight elements for each selector
        # ------------------------------------------------------------
        for selector in FAQ_SELECTORS:
            elements = page.query_selector_all(selector)
            if not elements:
                continue

            print(f"Found {len(elements)} elements for selector: {selector}")

            for el in elements:
                try:
                    page.evaluate(
                        """
                        (el) => {
                            el.style.border = "3px solid red";
                            el.style.padding = "4px";
                            el.style.backgroundColor = "rgba(255, 0, 0, 0.15)";
                        }
                        """,
                        el,
                    )
                except Exception:
                    pass

        # ------------------------------------------------------------
        # Attempt to expand all accordion / FAQ items
        # ------------------------------------------------------------
        print("\nAttempting to expand accordion + FAQ elements...\n")

        try:
            page.evaluate(
                """
                () => {
                    const selectors = [
                        'button',
                        '.accordion-title',
                        '.faq-question',
                        '.question',
                        '.accordionHeadline'
                    ];

                    selectors.forEach(sel => {
                        document.querySelectorAll(sel).forEach(el => {
                            try { el.click(); } catch(e) {}
                        });
                    });

                    // Scroll to trigger lazy-loading
                    let totalHeight = 0;
                    const distance = 400;
                    const timer = setInterval(() => {
                        const scrollHeight = document.body.scrollHeight;
                        window.scrollBy(0, distance);
                        totalHeight += distance;
                        if (totalHeight >= scrollHeight) {
                            clearInterval(timer);
                        }
                    }, 150);
                }
                """
            )

            page.wait_for_timeout(2500)

        except Exception:
            print("Expansion script encountered a minor issue (ignored).")

        # Save state after expansion
        save_screenshot(page, "02_after_expansion")
        save_html(page, "02_after_expansion")

        # ------------------------------------------------------------
        # Detect classes containing 'accordion' or 'faq'
        # ------------------------------------------------------------
        print("\nDiscovering DOM classes containing 'accordion' or 'faq':\n")

        class_list = page.eval_on_selector_all(
            "body *",
            """
            els => {
                const classes = new Set();
                els.forEach(el => {
                    if (el.className && typeof el.className === "string") {
                        const cls = el.className.toLowerCase();
                        if (cls.includes("accordion") || cls.includes("faq")) {
                            classes.add(cls);
                        }
                    }
                });
                return Array.from(classes);
            }
            """
        )

        for cls in class_list:
            print(f"- {cls}")

        save_class_report(class_list)

        # ------------------------------------------------------------
        # Extract Q/A preview after expansion
        # ------------------------------------------------------------
        print("\nExtracting preview Q&A pairs from DOM...\n")

        extract_script = """
        () => {
            const results = [];

            const extract = (root) => {
                const q = root.querySelector(
                    'button, h1, h2, h3, h4, strong, .question, .faq-question'
                );
                const a = root.querySelector(
                    'p, div, .answer, .faq-answer, .accordion-content, .panel-body'
                );
                if (q && a) {
                    results.push({
                        question: q.innerText.trim(),
                        answer: a.innerText.trim().slice(0, 350)
                    });
                }
            };

            [
                '.accordion-panel',
                '.accordion-item',
                '.accordion',
                '.faq-item',
                '.faq',
                '.faq-block',
                '.panel'
            ].forEach(sel => {
                document.querySelectorAll(sel).forEach(extract);
            });

            return results.slice(0, 10);
        }
        """

        samples = page.eval_on_selector_all("body", extract_script)

        if samples:
            for i, qa in enumerate(samples, 1):
                print(f"Q{i}: {qa['question']}")
                print(f"A{i}: {qa['answer']}\n{'-'*60}")
        else:
            print("No structured FAQ elements detected. Try adjusting selectors.")

        print("\nChromium browser is ready. Interact manually to inspect layout.")
        print("Close the browser window to finish.\n")

        try:
            page.wait_for_timeout(3600000)
        except KeyboardInterrupt:
            print("Test manually stopped.")

        browser.close()


# --------------------------------------------------------
# CLI ENTRY POINT
# --------------------------------------------------------

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    run_visual_test(url)
