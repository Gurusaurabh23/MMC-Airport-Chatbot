"""
One-off script to capture real screenshots of the running Streamlit app for
the report (Section 8). Assumes `streamlit run app.py` is already running on
localhost:8501.

Run: python evaluation/capture_screenshots.py
Output: evaluation/plots/screenshot_ui_*.png
"""
from pathlib import Path
from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"
OUT = Path(__file__).resolve().parent / "plots"
OUT.mkdir(parents=True, exist_ok=True)


def snap_full(page, path):
    """Streamlit's scrollable content lives in an inner container, not
    document.body, so Playwright's full_page screenshot (which measures
    body height) clips content below the fold. Resize the viewport to the
    actual content height first instead. Streamlit also re-renders content
    a beat after the DOM update settles, so wait for network/layout to be
    idle before measuring."""
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(500)
    height = page.evaluate(
        """() => {
            const candidates = [
                document.querySelector('[data-testid="stMain"]'),
                document.querySelector('[data-testid="stAppViewContainer"]'),
                document.querySelector('.main'),
                document.body,
                document.documentElement,
            ].filter(Boolean);
            return Math.max(...candidates.map(el => el.scrollHeight));
        }"""
    )
    page.set_viewport_size({"width": 1280, "height": min(int(height) + 60, 8000)})
    page.wait_for_timeout(300)
    page.screenshot(path=path)


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(URL, wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(2000)

        # 1. Home screen
        snap_full(page, str(OUT / "screenshot_ui_home.png"))
        print("saved screenshot_ui_home.png")

        # 2. Type a text query and submit
        text_input = page.get_by_placeholder("e.g. Where is gate B12?")
        text_input.click()
        text_input.fill("Where is gate B12?")
        ask_button = page.get_by_role("button", name="Ask the assistant")
        ask_button.click()
        # First inference call loads all pipelines (CLIP/DistilBERT/Whisper)
        # from disk, so wait for the response panel rather than a fixed delay.
        page.wait_for_selector("text=Here's what I found", timeout=60000)
        page.wait_for_timeout(1000)
        snap_full(page, str(OUT / "screenshot_ui_answered.png"))
        print("saved screenshot_ui_answered.png")

        # 3. Expand the debug panel
        debug_expander = page.get_by_text("Debug: routing details", exact=False)
        debug_expander.click()
        page.wait_for_timeout(1000)
        snap_full(page, str(OUT / "screenshot_ui_debug.png"))
        print("saved screenshot_ui_debug.png")

        browser.close()


if __name__ == "__main__":
    main()
