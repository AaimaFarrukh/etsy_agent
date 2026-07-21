"""
LangGraph + Playwright agent that logs into Etsy and searches keywords.

SAFETY NOTES (read this before running):
- Etsy's Terms of Service prohibit automated access. Careful, human-like
  automation reduces (but never eliminates) the risk of your account being
  flagged. Use this for light personal use only — not for scraping at scale
  or running many times a day.
- This script never tries to solve CAPTCHAs or bypass verification. If Etsy
  challenges the login, the script pauses and waits for YOU to solve it in
  the visible browser window.
- It reuses a persistent browser profile (cookies/session) so it doesn't
  need to log in every run, which is the single biggest thing you can do
  to look "normal."
- Keep the browser headed (visible), keep concurrency at 1, keep delays in
  place. Don't remove the random delays to "speed things up."
"""

import asyncio
import os
import random
from typing import TypedDict, List, Optional

from dotenv import load_dotenv
from playwright.async_api import (
    async_playwright,
    BrowserContext,
    Page,
    Error as PWError,
    TimeoutError as PWTimeoutError,
)
from langgraph.graph import StateGraph, END

load_dotenv()

ETSY_EMAIL = os.getenv("ETSY_EMAIL")
ETSY_PASSWORD = os.getenv("ETSY_PASSWORD")

# Persistent profile directory -> keeps you logged in across runs so the
# script doesn't need to authenticate every single time.
USER_DATA_DIR = os.path.expanduser("~/.etsy_agent_profile")


class AgentState(TypedDict):
    keywords: List[str]
    keyword_index: int
    logged_in: bool
    results: List[dict]
    status: str
    needs_manual_action: bool


class EtsyAgent:
    def __init__(self):
        self.playwright = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    # ---------- human-like helpers ----------

    async def human_delay(self, a: float = 0.8, b: float = 2.2):
        await asyncio.sleep(random.uniform(a, b))

    async def human_type(self, selector: str, text: str):
        el = self.page.locator(selector).first
        await el.click()
        for ch in text:
            await el.type(ch, delay=random.uniform(60, 180))
        await self.human_delay(0.3, 0.9)

    # ---------- graph nodes ----------

    async def setup_browser(self, state: AgentState) -> AgentState:
        self.playwright = await async_playwright().start()
        # Persistent context = reuses cookies/local storage across runs.
        self.context = await self.playwright.chromium.launch_persistent_context(
            USER_DATA_DIR,
            headless=False,  # headed browsing looks far less bot-like
            no_viewport=False,
            viewport={"width": 1366, "height": 850},  # let the real window size drive rendering —
                                # a fixed viewport on a headed browser can
                                # mismatch the actual window and cause
                                # broken/blank layouts
            locale="en-US",
            args=[
                "--disable-blink-features=AutomationControlled",
                "--window-size=1366,850",
                "--window-position=0,0",
                # NOTE: --disable-gpu was tried as a fix for a blank-render
                # issue and made it worse on this machine (fully blank
                # instead of just mis-positioned) — deliberately NOT used.
            ],
        )
        # Reduce the most common automation fingerprint. This is a single
        # standard property override, not fingerprint spoofing — widely
        # used in ordinary browser testing.
        await self.context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )
        # Etsy's sign-in / verification flow can open a new tab or replace
        # the current one. Track whichever page is newest/active instead of
        # pinning to a single Page object, so we don't end up waiting on a
        # tab that's since been closed.
        self.context.on("page", self._on_new_page)
        self.page = self.context.pages[0] if self.context.pages else await self.context.new_page()
        await self.page.goto("https://www.etsy.com", wait_until="domcontentloaded")
        await self.human_delay()
        await self._nudge_repaint(self.page)
        print("✅ Browser ready")
        state["status"] = "browser_ready"
        return state

    async def _nudge_repaint(self, page: Page):
        """Workaround for a Chromium-under-automation bug (mainly on
        Windows) where the visible window can render blank until something
        forces a repaint. Harmless no-op if not needed."""
        try:
            await page.bring_to_front()
            await page.evaluate("window.dispatchEvent(new Event('resize'))")
        except Exception:
            pass

    def _on_new_page(self, page: Page):
        self.page = page

    async def _first_match(self, page: Page, selectors: list):
        """Try each selector in order, return the first Locator that
        actually matches something on the page. None if nothing matches."""
        for sel in selectors:
            try:
                loc = page.locator(sel).first
                if await loc.count() > 0:
                    return loc
            except Exception:
                continue
        return None

    def _active_page(self) -> Optional[Page]:
        """Return a page that's still open, preferring self.page, falling
        back to any other open tab in the context."""
        if self.page and not self.page.is_closed():
            return self.page
        open_pages = [p for p in (self.context.pages if self.context else []) if not p.is_closed()]
        if open_pages:
            self.page = open_pages[-1]
            return self.page
        return None

    LOGGED_IN_SELECTORS = [
        '[data-selector="user-menu"]',
        'a[href*="/your/account"]',
        'a[href*="/signout"]',
        'button[aria-label*="account" i]',
        '#gnav-account-icon',
        '[data-gnav-tab="account"]',
        'a[href*="/your/purchases"]',
    ]

    async def _is_logged_in(self, page: Page) -> bool:
        for sel in self.LOGGED_IN_SELECTORS:
            try:
                if await page.locator(sel).count() > 0:
                    return True
            except Exception:
                continue
        return False

    async def _wait_until_logged_in(self, timeout_ms: int, poll_interval: float = 1.0):
        """Poll _is_logged_in across all known selectors until true or the
        timeout elapses. Raises TimeoutError-like RuntimeError on timeout,
        or RuntimeError if the browser closes."""
        elapsed = 0
        while elapsed < timeout_ms:
            page = self._active_page()
            if page is None:
                raise RuntimeError("browser_closed")
            if await self._is_logged_in(page):
                return
            await asyncio.sleep(poll_interval)
            elapsed += int(poll_interval * 1000)
        raise TimeoutError("login_wait_timeout")

    async def check_login(self, state: AgentState) -> AgentState:
        try:
            await self.page.wait_for_load_state("domcontentloaded", timeout=8000)
        except Exception:
            pass
        signed_in = await self._is_logged_in(self.page)
        state["logged_in"] = signed_in
        state["status"] = "checked_login"
        print(f"✅ Already logged in: {signed_in}")
        return state

    async def login(self, state: AgentState) -> AgentState:
        if not ETSY_EMAIL or not ETSY_PASSWORD:
            raise RuntimeError(
                "ETSY_EMAIL / ETSY_PASSWORD are not set. Add them to your .env file "
                "(see .env.example) before running the agent."
            )

        await self.page.goto("https://www.etsy.com/signin", wait_until="domcontentloaded")
        await self.human_delay()

        # If Etsy simply redirected us back because we were already
        # authenticated, there's no form to fill — skip straight to done.
        if await self._is_logged_in(self.page):
            state["logged_in"] = True
            state["needs_manual_action"] = False
            state["status"] = "logged_in"
            print("✅ Already logged in (no form needed)")
            return state

        await self.human_type('input#join_neu_email_field, input[name="email"]', ETSY_EMAIL)
        await self.page.locator('button:has-text("Continue"), button[type="submit"]').first.click()
        await self.human_delay(1.2, 2.5)

        await self.human_type('input#join_neu_password_field, input[name="password"]', ETSY_PASSWORD)
        await self.page.locator('button:has-text("Sign in"), button[type="submit"]').first.click()
        await self.human_delay(2, 4)

        # --- Human-in-the-loop checkpoint ---
        # The ONLY point where a human is expected to step in: if Etsy throws
        # a CAPTCHA, email code, or 2FA prompt, we detect it and pause here
        # instead of trying to automate around it.
        await self._wait_for_human_if_challenged()

        state["logged_in"] = True
        state["needs_manual_action"] = False
        state["status"] = "logged_in"
        print("✅ Logged in")
        return state

    async def _wait_for_human_if_challenged(self, timeout_ms: int = 300_000):
        """After submitting the login form: if a real CAPTCHA or 2FA/OTP
        prompt is detected, pause and wait for a human to resolve it. If
        not, just wait briefly for the normal post-login page — no pause,
        no message, since nothing needs a human."""
        page = self._active_page()
        if page is None:
            raise RuntimeError(
                "The browser window was closed. Re-run the script — a "
                "persistent profile is in use, so you shouldn't need to log "
                "in from scratch."
            )

        # Give the page a moment to settle after submit, then check once
        # for an unambiguous CAPTCHA/2FA challenge.
        await self.human_delay(1.0, 2.0)

        if await self._page_shows_challenge():
            page = self._active_page()
            print("\n" + "=" * 60)
            print("🛑 HUMAN INPUT NEEDED: real CAPTCHA or 2FA/verification-code")
            print("   prompt detected.")
            print(f"   Current page: {page.url}")
            print("   Please complete it in the browser window — don't close it.")
            print(f"   Waiting up to {timeout_ms // 1000}s for you to finish...")
            print("=" * 60 + "\n")
            try:
                await self._wait_until_logged_in(timeout_ms)
                print("✅ Verified — continuing automatically.\n")
            except TimeoutError:
                raise RuntimeError(
                    "Timed out waiting for you to complete the challenge. "
                    "Re-run the script and finish it faster."
                )
            except RuntimeError:
                raise RuntimeError(
                    "The browser window was closed before the challenge was "
                    "completed. Re-run the script — a persistent profile is "
                    "in use, so you shouldn't need to log in from scratch."
                )
            return

        # No challenge — just a normal login. Wait briefly for it to land,
        # no human involved.
        if page is None:
            raise RuntimeError(
                "The browser window was closed. Re-run the script — a "
                "persistent profile is in use, so you shouldn't need to log "
                "in from scratch."
            )
        try:
            await self._wait_until_logged_in(15_000)
        except TimeoutError:
            page = self._active_page()
            url = page.url if page else "(browser closed)"
            raise RuntimeError(
                f"Login didn't complete and no CAPTCHA/2FA challenge was "
                f"detected either. Current page: {url}\n"
                "The sign-in form's field selectors may not match what "
                "Etsy served this time — check the browser window and "
                "update the selectors in login() if needed."
            )
        except RuntimeError:
            raise RuntimeError(
                "The browser window was closed. Re-run the script — a "
                "persistent profile is in use, so you shouldn't need to log "
                "in from scratch."
            )

    async def _page_shows_challenge(self) -> bool:
        """True only if the page shows something unambiguously a CAPTCHA or
        a 2FA/one-time-code verification step. Deliberately narrow — this is
        the only thing allowed to trigger the human-input pause."""
        page = self._active_page()
        if page is None:
            return False

        # 1) Known CAPTCHA / bot-protection iframes and widgets.
        captcha_hits = await page.locator(
            'iframe[src*="captcha" i], iframe[title*="captcha" i], '
            'iframe[src*="perimeterx" i], iframe[src*="arkose" i], '
            'iframe[title*="challenge" i], #px-captcha, '
            'div[id*="challenge" i][id*="captcha" i]'
        ).count()
        if captcha_hits > 0:
            return True

        # 2) A one-time-code / 2FA input field — the standard, unambiguous
        # markup for this step, regardless of exact wording.
        otp_field_hits = await page.locator(
            'input[autocomplete="one-time-code"], '
            'input[name*="otp" i], input[id*="otp" i], '
            'input[name*="verification_code" i], input[id*="verification_code" i], '
            'input[name*="two_factor" i], input[id*="two_factor" i]'
        ).count()
        if otp_field_hits > 0:
            return True

        # 3) A small set of exact phrases that only appear on a real
        # verification screen, not in ordinary page copy.
        exact_phrases = [
            "press and hold",
            "prove you're not a robot",
            "we sent a code to",
            "enter the 6-digit code",
            "enter the code we sent",
        ]
        try:
            body_text = (await page.locator("body").inner_text()).lower()
        except Exception:
            body_text = ""
        return any(phrase in body_text for phrase in exact_phrases)

    async def search_keyword(self, state: AgentState) -> AgentState:
        keyword = state["keywords"][state["keyword_index"]]
        print(f"🔎 Searching: {keyword}")
        page = self._active_page()
        if page is None:
            raise RuntimeError("The browser window was closed. Re-run the script.")

        await page.goto("https://www.etsy.com", wait_until="domcontentloaded")
        await self.human_delay()

        search_box = await self._first_match(page, [
            'input[name="search_query"]',
            'input[type="search"]',
            'input#global-enhancements-search-query',
            'input[data-search-input]',
            'input[placeholder*="Search" i]',
            'input[aria-label*="Search" i]',
        ])
        if search_box is None:
            raise RuntimeError(
                f"Couldn't find Etsy's search box — the site markup may have "
                f"changed. Current page: {page.url}\n"
                "Open the browser window, inspect the search input, and "
                "update the selector list in search_keyword()."
            )

        await search_box.click()
        for ch in keyword:
            await search_box.type(ch, delay=random.uniform(70, 200))
        await self.human_delay(0.4, 1.0)
        await search_box.press("Enter")

        await page.wait_for_load_state("domcontentloaded")
        await self._nudge_repaint(page)

        # Give JS-driven content (Etsy's results grid) a chance to settle.
        try:
            await page.wait_for_load_state("networkidle", timeout=10_000)
        except Exception:
            pass

        # Wait for actual listing results to render, not just a fixed
        # delay — Etsy's results grid loads in via JS after navigation.
        try:
            await page.wait_for_selector(
                '[data-listing-id], li[data-search-results-item]',
                timeout=15_000,
            )
        except Exception:
            debug_dir = os.path.expanduser("~/.etsy_agent_debug")
            os.makedirs(debug_dir, exist_ok=True)
            shot_path = os.path.join(debug_dir, f"debug_{keyword.replace(' ', '_')}.png")
            html_path = os.path.join(debug_dir, f"debug_{keyword.replace(' ', '_')}.html")
            try:
                await page.screenshot(path=shot_path, full_page=True)
                html = await page.content()
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html)
                print(f"⚠️  No listing results appeared for '{keyword}' within 15s "
                      f"(page: {page.url})")
                print(f"    Saved debug screenshot: {shot_path}")
                print(f"    Saved debug HTML: {html_path}")
            except Exception as diag_err:
                print(f"⚠️  No listing results appeared, and diagnostics failed: {diag_err}")

        await self.human_delay(1.0, 2.0)

        # small human-like scroll
        await page.mouse.wheel(0, random.randint(400, 900))
        await self.human_delay(0.5, 1.5)

        state["status"] = f"searched:{keyword}"
        return state

    async def extract_results(self, state: AgentState) -> AgentState:
        page = self._active_page()
        if page is None:
            raise RuntimeError("The browser window was closed. Re-run the script.")

        titles = await page.locator(
            '[data-listing-id] h3, li[data-search-results-item] h3, '
            'a[data-listing-id], .wt-text-caption'
        ).all_text_contents()
        # de-dupe while preserving order, drop empties
        seen = set()
        clean_titles = []
        for t in titles:
            t = t.strip()
            if t and t not in seen:
                seen.add(t)
                clean_titles.append(t)

        keyword = state["keywords"][state["keyword_index"]]
        state.setdefault("results", [])
        state["results"].append({
            "keyword": keyword,
            "titles": clean_titles[:10],
        })
        print(f"✅ Found {len(clean_titles)} listings for '{keyword}'")
        state["keyword_index"] += 1
        return state

    async def cooldown(self, state: AgentState) -> AgentState:
        # Longer, randomized pause between searches — avoids a robotic
        # fixed-interval pattern.
        await asyncio.sleep(random.uniform(6, 14))
        return state

    async def close(self):
        try:
            if self.context:
                await self.context.close()
        except Exception:
            pass
        try:
            if self.playwright:
                await self.playwright.stop()
        except Exception:
            pass


def build_graph(agent: EtsyAgent):
    graph = StateGraph(AgentState)

    graph.add_node("setup_browser", agent.setup_browser)
    graph.add_node("check_login", agent.check_login)
    graph.add_node("login", agent.login)
    graph.add_node("search", agent.search_keyword)
    graph.add_node("extract", agent.extract_results)
    graph.add_node("cooldown", agent.cooldown)

    graph.set_entry_point("setup_browser")
    graph.add_edge("setup_browser", "check_login")

    graph.add_conditional_edges(
        "check_login",
        lambda s: "search" if s["logged_in"] else "login",
        {"search": "search", "login": "login"},
    )

    graph.add_edge("login", "search")

    graph.add_edge("search", "extract")

    graph.add_conditional_edges(
        "extract",
        lambda s: "cooldown" if s["keyword_index"] < len(s["keywords"]) else END,
        {"cooldown": "cooldown", END: END},
    )

    graph.add_edge("cooldown", "search")

    return graph.compile()


async def main():
    agent = EtsyAgent()
    graph = build_graph(agent)

    init_state: AgentState = {
        "keywords": [
            "boho earrings",
            "personalized mug",
            "wall art print",
        ],
        "keyword_index": 0,
        "logged_in": False,
        "results": [],
        "status": "start",
        "needs_manual_action": False,
    }

    try:
        final_state = await graph.ainvoke(init_state, config={"recursion_limit": 100})
        for r in final_state["results"]:
            print(f"\n🔍 {r['keyword']}")
            for t in r["titles"]:
                print("   -", t)
    except Exception as e:
        print(f"\n❌ {type(e).__name__}: {e}")
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())