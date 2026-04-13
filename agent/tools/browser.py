from __future__ import annotations

import yaml
from pathlib import Path
from playwright.async_api import async_playwright, Browser, BrowserContext, Page

from agent.tools.base import Tool

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "agent.yaml"

# Selecteurs pour extraire le contenu principal (tries dans l'ordre)
_CONTENT_SELECTORS: list[str] = [
    "article",
    "main",
    "[role='main']",
    "#content",
    "#main-content",
    ".post-content",
    ".article-content",
    ".entry-content",
]

# Elements de bruit a supprimer avant extraction
_NOISE_SELECTORS: list[str] = [
    "nav", "header", "footer", "aside",
    "[role='navigation']", "[role='banner']", "[role='contentinfo']",
    ".sidebar", ".menu", ".cookie", ".popup", ".modal", ".ad",
]


def _load_browser_config() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("browser", {})


class BrowserTool(Tool):
    name = "browser"
    description = (
        "Navigue sur le web via Camoufox (anti-detect, Docker). "
        "Actions : navigate, extract_text, click, screenshot."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["navigate", "extract_text", "click", "screenshot"],
                "description": "Action a effectuer",
            },
            "url": {"type": "string", "description": "URL pour navigate ou extract_text"},
            "selector": {"type": "string", "description": "Selecteur CSS pour click"},
        },
        "required": ["action"],
    }

    def __init__(self) -> None:
        self._pw = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        bcfg = _load_browser_config()
        self._navigate_timeout: int = bcfg.get("navigate_timeout", 20_000)
        self._click_timeout: int = bcfg.get("click_timeout", 5000)
        self._extract_max_chars: int = bcfg.get("extract_max_chars", 30_000)

    async def _ensure_browser(self) -> None:
        """Connecte au Camoufox Docker via WebSocket si pas deja fait."""
        if self._page is not None:
            return
        ws_url = _load_browser_config().get("camoufox_ws", "")
        self._pw = await async_playwright().start()
        try:
            self._browser = await self._pw.firefox.connect(ws_url)
        except Exception:
            self._browser = await self._pw.chromium.connect(ws_url)
        # Context isole (cookies, storage separes de toute autre session)
        self._context = await self._browser.new_context()
        self._page = await self._context.new_page()

    async def execute(self, action: str, url: str = "", selector: str = "", **_) -> str:
        await self._ensure_browser()
        page = self._page

        if action == "navigate":
            if not url:
                return "[ERREUR] URL requise pour navigate"
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=self._navigate_timeout)
                title = await page.title()
                return f"Page chargee : {title} ({url})"
            except Exception as exc:
                return f"[ERREUR] Navigation echouee : {exc}"

        if action == "extract_text":
            if url:
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=self._navigate_timeout)
                except Exception as exc:
                    return f"[ERREUR] Navigation echouee : {exc}"

            text = await self._extract_content(page)
            if len(text.strip()) < 50:
                return "[ERREUR] Page vide ou contenu insuffisant"
            if len(text) > self._extract_max_chars:
                return text[:self._extract_max_chars] + f"\n\n... (tronque, {len(text)} chars)"
            return text

        if action == "click":
            if not selector:
                return "[ERREUR] Selecteur CSS requis pour click"
            try:
                await page.click(selector, timeout=self._click_timeout)
                return f"Click sur '{selector}' effectue."
            except Exception as exc:
                return f"[ERREUR] Click echoue sur '{selector}' : {exc}"

        if action == "screenshot":
            buf = await page.screenshot()
            out_path = Path("screenshot.png")
            out_path.write_bytes(buf)
            return f"Screenshot sauvegarde : {out_path.absolute()} ({len(buf)} bytes)"

        return f"[ERREUR] Action inconnue : {action}"

    async def _extract_content(self, page: Page) -> str:
        """Extrait le contenu principal de la page, en evitant le bruit.

        1. Cherche un element de contenu principal (article, main, etc.)
        2. Si trouve : retourne son texte
        3. Sinon : supprime le bruit (nav, header, footer, aside) et retourne body
        """
        # Essayer les selecteurs de contenu principal
        for sel in _CONTENT_SELECTORS:
            try:
                element = page.locator(sel).first
                if await element.count() > 0:
                    text = await element.inner_text()
                    if len(text) > 100:  # Assez de contenu pour etre utile
                        title = await page.title()
                        return f"# {title}\n\n{text}"
            except Exception:
                continue

        # Fallback : supprimer les elements de bruit puis extraire body
        for noise_sel in _NOISE_SELECTORS:
            try:
                await page.evaluate(
                    f"document.querySelectorAll('{noise_sel}').forEach(el => el.remove())"
                )
            except Exception:
                continue

        text = await page.inner_text("body")
        title = await page.title()
        return f"# {title}\n\n{text}"

    async def close(self) -> None:
        if self._page:
            await self._page.close()
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()
