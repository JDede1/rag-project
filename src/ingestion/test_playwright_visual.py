"""
test_playwright_visual.py
-------------------------------------
Interactive Playwright helper to visually debug RBC FAQ pages.

Purpose:
    • Launch Chromium (non-headless) to view dynamic FAQ rendering
    • Highlight accordion, FAQ, and panel elements
    • Auto-expand all accordion items (RBC hides content by default)
    • Print discovered DOM class names containing FAQ/accordion patterns
    • Extract and preview multiple FAQ formats used across RBC
"""

from playwright.sync_api import sync_playwright

# --------------------------------------------------------
# CONFIG
# --------------------------------------------------------

TEST_URL = (
    "https://www.rbcroyalbank.com/credit-cards/cardholders/frequently-asked-questions/general-questions.html"
)

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


# --------------------------------------------------------
# RUN VISUAL TEST
# --------------------------------------------------------
def run_visual_test():
    print("Launching interactive Chromium browser...")
    print(f"Loading URL: {TEST_URL}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=300)
        page = browser.new_page()
        page.goto(TEST_URL, timeout=60000)

        page.wait_for_load_state("domcontentloaded")
        page.wait_for_timeout(3000)

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
                            el.style.border = '3px solid red';
                            el.style.padding = '4px';
                            el.style.backgroundColor = 'rgba(255, 0, 0, 0.10)';
                        }
                        """,
                        el,
                    )
                except Exception:
                    pass

        # ------------------------------------------------------------
        # Auto-expand all accordions
        # ------------------------------------------------------------
        print("\nAttempting to expand all accordion items...\n")

        try:
            page.evaluate(
                """
                () => {
                    const buttons = document.querySelectorAll('button, .accordion-title');
                    buttons.forEach(btn => btn.click());
                }
                """
            )
            page.wait_for_timeout(2000)
        except Exception:
            pass

        # ------------------------------------------------------------
        # List all DOM classes containing 'accordion' or 'faq'
        # ------------------------------------------------------------
        print("\nListing DOM class names containing 'accordion' or 'faq':\n")

        class_list = page.eval_on_selector_all(
            "body *",
            """
            elements => {
                const classes = new Set();
                elements.forEach(el => {
                    if (el.className && typeof el.className === 'string') {
                        const cls = el.className.toLowerCase();
                        if (cls.includes('accordion') || cls.includes('faq')) {
                            classes.add(cls);
                        }
                    }
                });
                return Array.from(classes);
            }
            """,
        )

        for cls in class_list:
            print(f"- {cls}")

        # ------------------------------------------------------------
        # Extract sample Q&A from multiple known RBC structures
        # ------------------------------------------------------------
        print("\nExtracting preview Q&A pairs from DOM...\n")

        extract_script = """
        () => {
            const results = [];

            const extract = (root) => {
                const q = root.querySelector(
                    'button, h1, h2, h3, h4, strong, .question'
                );
                const a = root.querySelector(
                    'p, div, .answer, .accordion-content, .panel-body'
                );
                if (q && a) {
                    results.push({
                        question: q.innerText.trim(),
                        answer: a.innerText.trim().slice(0, 300),
                    });
                }
            };

            document.querySelectorAll('.accordion-panel').forEach(extract);
            document.querySelectorAll('.faq-item').forEach(extract);
            document.querySelectorAll('.faq, .faq-block, .panel').forEach(extract);

            return results.slice(0, 10);
        }
        """

        sample_faqs = page.eval_on_selector_all("body", extract_script)

        if sample_faqs:
            for i, faq in enumerate(sample_faqs, 1):
                print(f"Q{i}: {faq['question']}")
                print(f"A{i}: {faq['answer']}\n{'-'*60}")
        else:
            print("No structured FAQ elements detected. Expand elements manually or update selectors.")

        print("\nChromium browser is ready. Interact manually to inspect layout and behavior.")
        print("Close the browser window to end the test.\n")

        try:
            page.wait_for_timeout(3600000)
        except KeyboardInterrupt:
            print("Test interrupted by user.")

        browser.close()


if __name__ == "__main__":
    run_visual_test()
