import os
import re


# Path to Brave browser executable
BRAVE_PATH = r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"


class BrowserAdapter:
    """
    Adapter REAL usando Playwright com o navegador Brave.
    Navega a páginas da whitelist, extrai o texto principal.
    Brave bloqueia ads nativamente — essencial para YouTube e outros sites.
    """

    def __init__(self):
        self._browser = None
        self._context = None

    def _ensure_browser(self):
        """Lazy-launch Brave via Playwright."""
        if self._browser is not None:
            return
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            executable_path=BRAVE_PATH,
            headless=True,
            args=["--disable-gpu", "--no-sandbox"],
        )
        self._context = self._browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        print("[BROWSER] Brave launched (headless)")

    def _extract_main_text(self, page) -> str:
        """Extract the main readable text from a page, stripping nav/footer/ads."""
        # Try common article selectors first
        selectors = [
            "article", "main", "#content", "#mw-content-text",
            ".mw-parser-output", ".post-content", ".entry-content",
            "#bodyContent", ".article-body",
        ]
        for sel in selectors:
            try:
                el = page.query_selector(sel)
                if el:
                    text = el.inner_text()
                    if len(text) > 200:
                        return text[:8000]
            except:
                continue

        # Fallback: get body text
        try:
            text = page.inner_text("body")
            return text[:8000]
        except:
            return ""

    def query(self, topic: str, source: str = "Wikipedia EN") -> str:
        """
        Navigate to a source and extract text about a topic.
        """
        self._ensure_browser()

        # Map source names to search URLs
        source_urls = {
            "Wikipédia PT": f"https://pt.wikipedia.org/wiki/{topic.replace(' ', '_')}",
            "Wikipedia EN": f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}",
            "Wikilivros PT": f"https://pt.wikibooks.org/w/index.php?search={topic}",
            "Wikibooks EN": f"https://en.wikibooks.org/w/index.php?search={topic}",
            "SciELO Brasil": f"https://search.scielo.org/?q={topic}&lang=pt",
            "Domínio Público": f"http://www.dominiopublico.gov.br/pesquisa/ResultadaPesquisaObraForm.do?first=1&co_obra=&co_idioma=1&co_autor=&co_categoria=&pagina=1&select_action=&co_midia=2&ds_titulo={topic}",
            "arXiv": f"https://arxiv.org/search/?query={topic}&searchtype=all",
            "Project Gutenberg": f"https://www.gutenberg.org/ebooks/search/?query={topic}",
        }

        url = source_urls.get(source)
        if not url:
            # Generic search via Brave
            url = f"https://search.brave.com/search?q={topic}+site:{source}"

        try:
            page = self._context.new_page()
            page.goto(url, timeout=15000, wait_until="domcontentloaded")
            text = self._extract_main_text(page)
            page.close()

            if not text or len(text) < 50:
                return f"[Browser] Página sem conteúdo útil: {url}"

            return text

        except Exception as e:
            return f"[Browser] Erro ao navegar para {url}: {e}"

    def close(self):
        """Cleanup browser resources."""
        if self._browser:
            self._browser.close()
        if hasattr(self, '_pw') and self._pw:
            self._pw.stop()
