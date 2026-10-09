import requests
import re


class WikipediaAdapter:
    """
    Adapter REAL para a API da Wikipedia.
    Busca o conteúdo completo de um artigo em PT ou EN.
    """

    ENDPOINTS = {
        "pt": "https://pt.wikipedia.org/w/api.php",
        "en": "https://en.wikipedia.org/w/api.php",
    }

    def query(self, topic: str, source: str = "Wikipédia PT") -> str:
        lang = "pt" if "PT" in source.upper() else "en"
        endpoint = self.ENDPOINTS[lang]

        # 1. Search for the best matching article title
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": topic,
            "srlimit": 1,
            "format": "json",
        }
        headers = {
            "User-Agent": "NovaMindAGILearner/1.0 (contact@novamind.org)"
        }
        try:
            resp = requests.get(endpoint, params=search_params, headers=headers, timeout=10)
            resp.raise_for_status()
            results = resp.json().get("query", {}).get("search", [])
            if not results:
                return f"[Wikipedia] Nenhum artigo encontrado para: {topic}"
            title = results[0]["title"]
        except Exception as e:
            return f"[Wikipedia] Erro na busca: {e}"

        # 2. Get the full plain-text extract of the article
        extract_params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "exlimit": 1,
            "format": "json",
        }
        try:
            resp = requests.get(endpoint, params=extract_params, headers=headers, timeout=15)
            resp.raise_for_status()
            pages = resp.json().get("query", {}).get("pages", {})
            page = next(iter(pages.values()))
            text = page.get("extract", "")
            if not text:
                return f"[Wikipedia] Artigo '{title}' sem conteúdo extraível."

            # Truncate to ~8000 chars to avoid overloading a single think() cycle
            # The manager will break this into sentences anyway
            text = text[:8000]
            return text

        except Exception as e:
            return f"[Wikipedia] Erro ao extrair artigo '{title}': {e}"
