"""
test_playwright_visual.py
-------------------------------------
Interactive Playwright helper to visually debug RBC FAQ pages.

Purpose:
    • Launch Chromium (non-headless) to view dynamic FAQ rendering
    • Test which selectors (accordion buttons, FAQ items) are clickable
    • Automatically print and highlight FAQ/accordion elements
    • Extract and preview sample Q&A text directly from the DOM

Usage:
    python src/ingestion/test_playwright_visual.py
"""

from playwright.sync_api import sync_playwright


# ------------------------------------
# CONFIG
# ------------------------------------
TEST_URL = "https://www.rbcroyalbank.com/credit-cards/cardholders/frequently-asked-questions/general-questions.html"

FAQ_SELECTORS = [
    "button",
    ".accordion",
    ".accordion-panel",
    ".otmodal-accordion",
    ".faq-item",
    ".collapse-item",
    ".faq-question",
]


# ------------------------------------
# RUN INTERACTIVE SESSION
# ------------------------------------
def run_visual_test():
    print(f"🌐 Launching interactive Chromium browser...")
    print(f"🔗 Loading URL: {TEST_URL}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=400)
        page = browser.new_page()
        page.goto(TEST_URL, timeout=60000)

        # Wait until DOM content is ready
        page.wait_for_load_state("domcontentloaded")
        page.wait_for_timeout(3000)

        # 1️⃣ Highlight visible FAQ elements
        print("\n🔍 Highlighting FAQ-related elements on the page...\n")
        for selector in FAQ_SELECTORS:
            elements = page.query_selector_all(selector)
            if not elements:
                continue
            print(f"🧩 Found {len(elements)} elements for selector: {selector}")
            for el in elements:
                try:
                    page.evaluate(
                        """(el) => { el.style.border='3px solid red'; el.style.padding='3px'; }""",
                        el,
                    )
                except Exception:
                    pass

        # 2️⃣ Print all DOM class names containing “accordion” or “faq”
        print("\n🧠 Listing all DOM classes containing 'accordion' or 'faq':\n")
        class_names = page.eval_on_selector_all(
            "body *",
            """elements => {
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
            }""",
        )
        for cls in class_names:
            print(f"• {cls}")

        # 3️⃣ Extract and preview first few Q&A pairs from .accordion-panel blocks
        print("\n💬 Extracting first few Q&A pairs from the DOM...\n")
        sample_faqs = page.eval_on_selector_all(
            "body",
            """() => {
                const faqs = [];
                document.querySelectorAll('.accordion-panel').forEach(item => {
                    const q = item.querySelector('button, h3, h2, strong');
                    const a = item.querySelector('p, div');
                    if (q && a) {
                        faqs.push({
                            question: q.innerText.trim(),
                            answer: a.innerText.trim().slice(0, 300)
                        });
                    }
                });
                return faqs.slice(0, 5);
            }""",
        )

        if sample_faqs:
            for i, faq in enumerate(sample_faqs, 1):
                print(f"Q{i}: {faq['question']}\nA{i}: {faq['answer']}\n{'-'*50}")
        else:
            print("⚠️ No structured FAQ elements detected yet. Try expanding accordions manually or adjust selectors.")

        print("\n✅ Page loaded. Interact with it manually to observe expand/collapse behavior.")
        print("💡 When done, close the browser window to end the test.\n")
        print("🕐 Waiting indefinitely — close the browser window when ready.\n")

        # 4️⃣ Keep browser open indefinitely until user closes it
        try:
            page.wait_for_timeout(3600000)  # Wait up to 1 hour
        except KeyboardInterrupt:
            print("🛑 Test interrupted by user.")
        finally:
            browser.close()


if __name__ == "__main__":
    run_visual_test()
