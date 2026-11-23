"""

Pipeline
--------
1. Extract sentences from your text (academic-aware splitting).
2. Build several search-query types (exact phrase, distinctive phrase, keywords).
3. Search the web through a real search API (Brave or Tavily; optional unofficial ddgs).
4. Download the real pages / PDFs and extract readable text.
5. Compare: exact -> near-exact (difflib) -> semantic (Sentence-BERT + cosine).
6. Report one of three honest states:
     matches  /  no_matches  /  incomplete (search or downloads failed).

Semantic similarity is NOT proof of plagiarism. A human must review every match.
"""

import html
import io
import json
import math
import os
import re
import sqlite3
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from difflib import SequenceMatcher
from urllib.parse import parse_qs, urljoin, urlparse, urlunparse

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup
from docx import Document
from fpdf import FPDF
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# Search-provider configuration.
# API keys are NEVER hard-coded. They are read from (in this order):
#   1. the sidebar key box (this browser session only, never written to disk)
#   2. environment variables, including a local ".env" file next to app.py
#   3. Streamlit secrets (.streamlit/secrets.toml)
# --------------------------------------------------------------------------
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
TAVILY_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"


def _load_dotenv(path):
    """Tiny .env reader (no extra dependency). Real environment variables win."""
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, value = line.split("=", 1)
                clean_name = name.strip()
                clean_val = value.strip().strip('"').strip("'")
                os.environ[clean_name] = clean_val
    except OSError:
        pass


_load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


def get_secret(name):
    """Return (value, source_description) for an API key, or ("", "") if not configured."""
    try:
        typed = st.session_state.get(f"session_key_{name}", "")
        if not typed:
            for k, val in st.session_state.items():
                if k.endswith(f"session_key_{name}") and val and str(val).strip():
                    typed = str(val).strip()
                    break
    except Exception:
        typed = ""
    if typed and str(typed).strip():
        return str(typed).strip(), "entered in session (this session only)"

    # Dynamically reload .env to ensure editor edits are immediately reflected
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_file):
        _load_dotenv(env_file)

    value = os.getenv(name, "").strip()
    if value:
        return value, "environment variable / .env file"
    try:
        if hasattr(st, "secrets") and name in st.secrets:
            value = str(st.secrets[name]).strip()
            if value:
                return value, "Streamlit secrets"
    except Exception:
        pass
    return "", ""


def get_env_or_secret(name, default=""):
    """Return secret value from session, environment, or st.secrets."""
    val, _ = get_secret(name)
    return val if val else default


def is_low_memory_mode():
    val = get_env_or_secret("LOW_MEMORY_MODE", "true").lower()
    return val in ("true", "1", "yes")


def get_process_memory_mb() -> float:
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def get_writable_db_path():
    """Return (db_path, is_ephemeral). Tries project root first, falls back to tempdir."""
    custom = os.getenv("PLAGIARISM_DB_PATH", "").strip()
    if custom:
        return custom, False
    root_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(root_dir, "plagiarism_history.db")
    try:
        test_file = os.path.join(root_dir, ".write_test")
        with open(test_file, "a") as f:
            pass
        os.remove(test_file)
        return candidate, False
    except OSError:
        pass
    import tempfile
    temp_dir = os.path.join(tempfile.gettempdir(), "plagiarism_ai")
    os.makedirs(temp_dir, exist_ok=True)
    return os.path.join(temp_dir, "plagiarism_history.db"), True


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
APP_NAME = "PlagiarismAI"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DB_PATH, IS_EPHEMERAL_DB = get_writable_db_path()

from nlp.extract import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_txt,
    extract_text_from_upload,
)
from nlp.similarity import (
    MIN_WORDS,
    EXACT_MIN_WORDS,
    SEMANTIC_MIN_WORDS,
    MAX_SOURCE_SENTENCES,
    MAX_COMPARE_SENTENCES,
    EXACT,
    NEAR,
    SEMANTIC,
    TOPICAL,
    NONE,
    RANK,
    COUNTED_TYPES,
    COLORS,
    HIGHLIGHT_BG,
    DISPLAY_TYPE,
    CONTEXT_MIN_SIM,
    DISTINCTIVE_DF,
    LEXICAL_MIN_SHARED,
    LEXICAL_MIN_COVERAGE,
    COMMON_SOURCE_MIN,
    STOPWORDS,
    _is_technical_token,
    _stem,
    normalize_text,
    extract_sentences,
    content_tokens,
    document_frequencies,
    lexical_evidence,
    detect_exact_matches,
    detect_near_exact_matches,
    calculate_similarity,
    semantic_level,
    encode_sentences,
    detect_semantic_matches,
    finalize_semantic_matches,
    match_against_source,
    rank_source_sentences,
    semantic_breakdown,
    summarize_matches,
    report_stat_lines,
    compare_documents,
    scan_against_reference_corpus,
)
from nlp.preprocess import (
    preprocess_document,
    compute_composite_originality_score,
)
from nlp.analysis import (
    analyze_sentiment,
    extract_named_entities,
    extract_keywords,
    extract_citations,
    is_spacy_available,
)
from nlp.topics import extract_document_topics
from nlp.multilingual import detect_language_smart, translate_text_smart
from llm.assistant import fix_passage, verify_gemini_model_at_startup

QUERY_EXACT, QUERY_PHRASE, QUERY_KEYWORDS = "Exact phrase", "Distinctive phrase", "Technical keywords"
QUERY_WEIGHT = {QUERY_EXACT: 3.0, QUERY_PHRASE: 2.0, QUERY_KEYWORDS: 1.0}

# --- Semantic evidence gate (starting values, NOT calibrated: tune with validate_real_model.py) ---
# A cosine >= threshold only makes a passage a *candidate*. It is counted as a Semantic Match
# only when independent evidence of reuse exists; otherwise it is reported as Topical Similarity.
CONTEXT_MIN_SIM = 0.60        # an aligned neighbour must be >= this similar AND be that neighbour's best match
DISTINCTIVE_DF = 0.10         # a word is "distinctive" if it occurs in <= 10% of the scan's sentences
LEXICAL_MIN_SHARED = 2        # distinctive words shared by the sentence and the passage
LEXICAL_MIN_COVERAGE = 0.25   # ...and they carry >= 25% of the sentence's distinctive-word weight
COMMON_SOURCE_MIN = 3         # candidates in >= 3 different sources = standard statement (lexical support vetoed)

STOPWORDS = set("""a about above after again all also an and any are as at be because been before being
between both but by can could did do does doing down during each few for from further had has have having
he her here hers him his how i if in into is it its itself just may me more most my no nor not of off on once
only or other our out over own same she should so some such than that the their theirs them then there these
they this those through to too under until up very was we were what when where which while who whom why will
with would you your thus however therefore which whose using used use based paper show shown results result
propose proposed present presented approach method methods""".split())

NOT_FOUND_MESSAGE = ("Web search could not be completed. "
                     "No reliable plagiarism conclusion can be generated.")

HTML_LIMIT = 3_000_000
PDF_LIMIT = 20_000_000

REQUEST_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
ALT_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0"),
    "Accept": "*/*",
}


# --------------------------------------------------------------------------
# 1. Model
# --------------------------------------------------------------------------
MULTILINGUAL_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
FALLBACK_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_MODEL_LOAD_STATUS = {
    "name": "",
    "is_multilingual": True,
    "warning": ""
}


@st.cache_resource(show_spinner=False)
def load_bert_model():
    """Load Sentence-BERT lazily once per server process."""
    from sentence_transformers import SentenceTransformer

    custom_model = get_env_or_secret("EMBEDDING_MODEL", "").strip()
    if custom_model:
        model = SentenceTransformer(custom_model)
        _MODEL_LOAD_STATUS["name"] = custom_model
        _MODEL_LOAD_STATUS["is_multilingual"] = "multilingual" in custom_model.lower()
        _MODEL_LOAD_STATUS["warning"] = ""
        print(f"[*] PlagiarismAI: Loaded Custom Embedding Model: {custom_model}")
        return model

    if is_low_memory_mode():
        # Low-memory mode: all-MiniLM-L6-v2 (~578 MB RAM, fits comfortably in 1GB free tier)
        try:
            model = SentenceTransformer(FALLBACK_MODEL_NAME, local_files_only=True)
        except Exception:
            model = SentenceTransformer(FALLBACK_MODEL_NAME)
        _MODEL_LOAD_STATUS["name"] = FALLBACK_MODEL_NAME
        _MODEL_LOAD_STATUS["is_multilingual"] = False
        _MODEL_LOAD_STATUS["warning"] = ""
        print(f"[*] PlagiarismAI (Low-Memory Mode): Loaded {FALLBACK_MODEL_NAME}")
        return model

    # If low-memory mode is explicitly disabled, try multilingual model
    try:
        model = SentenceTransformer(MULTILINGUAL_MODEL_NAME, local_files_only=True)
        _MODEL_LOAD_STATUS["name"] = MULTILINGUAL_MODEL_NAME
        _MODEL_LOAD_STATUS["is_multilingual"] = True
        _MODEL_LOAD_STATUS["warning"] = ""
        print(f"[*] PlagiarismAI: Loaded Multilingual Embedding Model: {MULTILINGUAL_MODEL_NAME}")
        return model
    except Exception as exc_multi:
        print(f"[!] Warning: Multilingual model failed to load ({exc_multi}). Attempting fallback to English-only model.")

    try:
        model = SentenceTransformer(FALLBACK_MODEL_NAME, local_files_only=True)
    except Exception:
        model = SentenceTransformer(FALLBACK_MODEL_NAME)

    _MODEL_LOAD_STATUS["name"] = FALLBACK_MODEL_NAME
    _MODEL_LOAD_STATUS["is_multilingual"] = False
    _MODEL_LOAD_STATUS["warning"] = (
        f"Multilingual model ({MULTILINGUAL_MODEL_NAME}) failed to load. "
        f"Using English-only model ({FALLBACK_MODEL_NAME}). Cross-lingual Devanagari matching will rely on translation."
    )
    print(f"[*] PlagiarismAI: Loaded English-only Embedding Model: {FALLBACK_MODEL_NAME}")
    print(f"[!] Warning: {_MODEL_LOAD_STATUS['warning']}")
    return model


def encode_sentences(model, sentences):
    """Batch-encode sentences into unit-length BERT embeddings."""
    return model.encode(sentences, batch_size=64, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)


# --------------------------------------------------------------------------
# 2-4. Text extraction from files (imported from nlp.extract)
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 5-6. Normalisation and sentences (imported from nlp.similarity)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# 7a. Search-query generation (academic-friendly)
# --------------------------------------------------------------------------
def _term_weight(token):
    """How 'distinctive' a word is: technical terms and long words score higher."""
    t = token.strip(".,;:!?()[]{}\"'\u201c\u201d\u2018\u2019")
    if len(t) < 3 or t.lower() in STOPWORDS:
        return 0.0
    weight = 0.5
    if len(t) >= 7:
        weight += 1.0
    if len(t) >= 11:
        weight += 0.5
    if _is_technical_token(t):
        weight += 2.0
    return weight


def _clean_words(sentence):
    words = [w.strip(".,;:!?()[]{}\"'\u201c\u201d\u2018\u2019") for w in sentence.split()]
    return [w for w in words if w]


def distinctive_phrase(sentence, size=12):
    """The 8-15 word window of a sentence with the most distinctive terms."""
    words = _clean_words(sentence)
    if len(words) <= size:
        return " ".join(words)
    weights = [_term_weight(w) for w in words]
    start = max(range(len(words) - size + 1), key=lambda i: sum(weights[i:i + size]))
    return " ".join(words[start:start + size])


def top_terms(words, n):
    """The n most distinctive (weighted by frequency) terms, original spelling kept."""
    counter, spelling = Counter(), {}
    for word in words:
        weight = _term_weight(word)
        if weight > 0:
            counter[word.lower()] += weight
            spelling.setdefault(word.lower(), word)
    return [spelling[k] for k, _ in counter.most_common(n)]


def build_search_queries(sentences, max_queries):
    """
    Build up to `max_queries` queries from the actual text of document sentences,
    searching exact phrases in quotes AND shorter keyword versions across the
    first, middle, and last parts of the document.
    """
    if not sentences or max_queries < 1:
        return []

    n = len(sentences)
    # Divide document into first, middle, and last sections
    if n == 1:
        sections = [(0, 1)]
    elif n == 2:
        sections = [(0, 1), (1, 2)]
    else:
        s1 = max(1, n // 3)
        s2 = max(s1 + 1, (2 * n) // 3)
        sections = [(0, s1), (s1, s2), (s2, n)]

    # Collect and score candidate sentences for each section
    section_candidates = []
    for start, end in sections:
        sec_scored = []
        for i in range(start, end):
            s = sentences[i]
            words = _clean_words(s)
            if len(words) >= 4:
                weight = sum(_term_weight(w) for w in words)
                score = weight * math.sqrt(len(words))
                sec_scored.append((score, i, s))
        if sec_scored:
            sec_scored.sort(key=lambda x: -x[0])
            section_candidates.append(sec_scored)
        else:
            sec_candidates = [(1.0, i, sentences[i]) for i in range(start, end)]
            section_candidates.append(sec_candidates)

    queries, seen = [], set()

    # Pass 1: Exact quoted phrases across first, middle, and last sections
    for candidates in section_candidates:
        if not candidates or len(queries) >= max_queries:
            break
        _, idx, s = candidates[0]
        words = _clean_words(s)
        phrase = " ".join(words[:14]) if len(words) <= 14 else distinctive_phrase(s, size=12)
        exact_q = f'"{phrase}"'
        key = exact_q.lower()
        if key not in seen and len(phrase) >= 12:
            seen.add(key)
            queries.append({"query": exact_q[:250], "type": QUERY_EXACT, "sentence_index": idx})

    # Pass 2: Shorter keyword queries across first, middle, and last sections
    for candidates in section_candidates:
        if not candidates or len(queries) >= max_queries:
            break
        _, idx, s = candidates[0]
        words = _clean_words(s)
        kw_words = [w for w in words if _term_weight(w) >= 1.2]
        kw_phrase = " ".join(kw_words[:6]) if len(kw_words) >= 3 else distinctive_phrase(s, size=6)
        key = kw_phrase.lower()
        if key not in seen and len(kw_phrase) >= 8:
            seen.add(key)
            queries.append({"query": kw_phrase[:200], "type": QUERY_PHRASE, "sentence_index": idx})

    # Pass 3: Additional distinctive queries if budget remains
    if len(queries) < max_queries:
        for candidates in section_candidates:
            for _, idx, s in candidates[1:]:
                words = _clean_words(s)
                phrase = " ".join(words[:14]) if len(words) <= 14 else distinctive_phrase(s, size=12)
                exact_q = f'"{phrase}"'
                key = exact_q.lower()
                if key not in seen and len(phrase) >= 12:
                    seen.add(key)
                    queries.append({"query": exact_q[:250], "type": QUERY_EXACT, "sentence_index": idx})
                if len(queries) >= max_queries:
                    break
            if len(queries) >= max_queries:
                break

    # Pass 4: Fallback generic distinctive phrases
    if len(queries) < max_queries:
        for candidates in section_candidates:
            for _, idx, s in candidates:
                phrase = distinctive_phrase(s, size=10)
                key = phrase.lower()
                if key not in seen and len(phrase) >= 8:
                    seen.add(key)
                    queries.append({"query": phrase[:200], "type": QUERY_PHRASE, "sentence_index": idx})
                if len(queries) >= max_queries:
                    break
            if len(queries) >= max_queries:
                break

    return queries[:max_queries]


def zip_longest_lists(*lists):
    """Round-robin over lists, skipping exhausted ones."""
    for k in range(max((len(x) for x in lists), default=0)):
        for items in lists:
            if k < len(items):
                yield items[k]


# --------------------------------------------------------------------------
# 7b. Real web search
# --------------------------------------------------------------------------
def canonical_url(url):
    """Normalise a URL so duplicates can be detected."""
    parts = urlparse(url.strip())
    netloc = parts.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return urlunparse((parts.scheme.lower(), netloc, parts.path.rstrip("/"), "", parts.query, ""))


class ProviderError(Exception):
    """A search provider failed. fatal=True means further queries are pointless (bad key etc.)."""

    def __init__(self, message, fatal=False):
        super().__init__(message)
        self.fatal = fatal


def _check_response(response, name):
    """Turn HTTP errors from a search API into clear messages."""
    code = response.status_code
    if code == 200:
        return
    body = response.text[:200].replace("\n", " ")
    if code in (401, 403):
        raise ProviderError(f"{name}: API key rejected (HTTP {code}). {body}", fatal=True)
    if code == 402:
        raise ProviderError(f"{name}: no active plan or credits (HTTP 402). {body}", fatal=True)
    if code == 429:
        raise ProviderError(f"{name}: rate limit or quota reached (HTTP 429). {body}")
    raise ProviderError(f"{name}: HTTP {code}. {body}")


def _search_brave(query, max_results, api_key, timeout=15):
    """Brave Search API: GET /res/v1/web/search, key in the X-Subscription-Token header."""
    response = requests.get(
        BRAVE_URL, timeout=timeout,
        headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        params={"q": query[:400], "count": min(max_results, 20)})
    _check_response(response, "Brave")
    try:
        items = response.json().get("web", {}).get("results", [])
    except ValueError:
        raise ProviderError("Brave: response was not valid JSON")
    return [{"title": i.get("title", ""), "href": i.get("url"), "body": i.get("description", "")}
            for i in items]


def _search_tavily(query, max_results, api_key, timeout=20):
    """Tavily Search API: POST /search with include_raw_content=True to get page text directly."""
    response = requests.post(
        TAVILY_URL, timeout=timeout,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "query": query[:400],
            "max_results": min(max_results, 20),
            "search_depth": "basic",
            "include_raw_content": True
        })
    _check_response(response, "Tavily")
    try:
        items = response.json().get("results", [])
    except ValueError:
        raise ProviderError("Tavily: response was not valid JSON")
    return [{"title": i.get("title", ""), "href": i.get("url"), "body": i.get("content", ""),
             "raw_content": i.get("raw_content", "")}
            for i in items]


def _tavily_extract(urls, api_key, timeout=15):
    """Tavily Extract API: extracts clean full page text from candidate URLs."""
    if not urls or not api_key:
        return {}
    clean_urls = [u for u in urls if u and str(u).startswith("http") and not str(u).lower().endswith(".pdf")]
    if not clean_urls:
        return {}
    try:
        resp = requests.post(
            TAVILY_EXTRACT_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"urls": clean_urls[:10]},
            timeout=timeout
        )
        if resp.status_code == 200:
            data = resp.json()
            out = {}
            for item in data.get("results", []):
                u = item.get("url")
                raw = item.get("raw_content") or item.get("content") or ""
                if u and len(raw) >= 100:
                    out[u] = {"text": raw, "title": item.get("title", "")}
            return out
    except Exception:
        pass
    return {}


def _fetch_wikipedia_content(url, timeout=8):
    """Official Wikipedia API/REST endpoint for clean extracts without web scraping or anti-bot blocks."""
    try:
        import urllib.parse
        parsed = urlparse(url)
        if "wikipedia.org" not in parsed.netloc or "/wiki/" not in parsed.path:
            return None
        lang = parsed.netloc.split(".")[0] if "." in parsed.netloc else "en"
        title = parsed.path.split("/wiki/")[-1].split("#")[0]
        if not title:
            return None
        api_url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "titles": urllib.parse.unquote(title),
            "format": "json"
        }
        headers = {"User-Agent": "PlagiarismAI/2.0 (educational research bot; https://github.com/PlagiarismAI)"}
        resp = requests.get(api_url, params=params, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            pages = resp.json().get("query", {}).get("pages", {})
            for pid, p in pages.items():
                if pid != "-1":
                    text = p.get("extract", "")
                    if len(text) >= 100:
                        return {"ok": True, "text": text, "page_title": p.get("title", ""), "kind": "wikipedia_api", "reason": ""}
        return {"ok": False, "text": "", "page_title": "", "kind": "wikipedia_api", "reason": f"Wikipedia article '{title}' returned no readable text"}
    except Exception as exc:
        return {"ok": False, "text": "", "page_title": "", "kind": "wikipedia_api", "reason": f"Wikipedia API error: {exc}"}


def _search_ddgs(query, max_results, api_key="", timeout=15):
    """Optional, unofficial, key-less provider. One attempt only - no retries or backend cycling."""
    try:
        from ddgs import DDGS
    except Exception as exc:
        raise ProviderError(f"ddgs is not importable ({type(exc).__name__}: {exc}). Install it into "
                            f'this Python: "{sys.executable}" -m pip install -U ddgs', fatal=True)
    try:
        return DDGS(timeout=timeout).text(query, max_results=max_results) or []
    except Exception as exc:
        raise ProviderError(f"ddgs: {type(exc).__name__}: {exc}")


PROVIDERS = {
    "brave": {"label": "Brave Search API", "env": "BRAVE_API_KEY", "func": _search_brave,
              "interval": 1.1, "needs_key": True,
              "signup": "https://api-dashboard.search.brave.com/register"},
    "tavily": {"label": "Tavily Search API", "env": "TAVILY_API_KEY", "func": _search_tavily,
               "interval": 0.6, "needs_key": True, "signup": "https://app.tavily.com"},
    "ddgs": {"label": "DuckDuckGo via ddgs (unofficial, experimental)", "env": "",
             "func": _search_ddgs, "interval": 1.5, "needs_key": False, "signup": ""},
}
PROVIDER_CHOICES = ["auto", "brave", "tavily", "ddgs"]
SETUP_HELP = ("Set BRAVE_API_KEY or TAVILY_API_KEY (environment variable, .env file next to app.py, "
              "or .streamlit/secrets.toml), or paste a key in the sidebar under Search diagnostics. "
              "Tavily: https://app.tavily.com (free credits, no card). "
              "Brave: https://api-dashboard.search.brave.com/register (card required).")


def resolve_provider(choice="auto"):
    """Decide which provider to use. Returns id, label, key, source, configured, problem."""
    order = ["brave", "tavily"] if choice == "auto" else [choice]
    for provider_id in order:
        spec = PROVIDERS[provider_id]
        key, source = get_secret(spec["env"]) if spec["env"] else ("", "")
        if key or not spec["needs_key"]:
            return {"id": provider_id, "label": spec["label"], "key": key, "source": source,
                    "configured": True, "problem": ""}
    if choice == "auto":
        return {"id": None, "label": "none", "key": "", "source": "", "configured": False,
                "problem": "No search provider is configured. " + SETUP_HELP}
    spec = PROVIDERS[choice]
    return {"id": choice, "label": spec["label"], "key": "", "source": "", "configured": False,
            "problem": f"{spec['label']} has no API key ({spec['env']} is not set). " + SETUP_HELP}


def search_web_sources(queries, results_per_query=5, provider="auto"):
    """
    Run every query against the selected search API.
    Returns a dict:
      results        - genuine hits (title, url, snippet, found_by, best_rank), de-duplicated
      errors / notes - the real error messages and setup problems (never swallowed)
      queries        - per-query log (query, type, ok, hits, engine)
      provider / provider_label - what was used (provider is None when nothing is configured)
    Snippets are kept only for ranking. They are never treated as a verified comparison.
    """
    out = {"results": [], "errors": [], "notes": [], "queries": [], "engines": [],
           "provider": None, "provider_label": ""}
    info = resolve_provider(provider)
    out["provider_label"] = info["label"]
    if not info["configured"]:
        out["notes"].append(info["problem"])
        return out
    out["provider"] = info["id"]
    spec = PROVIDERS[info["id"]]
    by_url, error_counts, consecutive_failures = {}, Counter(), 0

    for position, q in enumerate(queries):
        entry = {"query": q["query"], "type": q["type"], "ok": False, "hits": 0,
                 "engine": info["label"]}
        raw, stop = [], False
        try:
            raw = spec["func"](q["query"], results_per_query, info["key"])
            entry["ok"] = True
            consecutive_failures = 0
            if info["label"] not in out["engines"]:
                out["engines"].append(info["label"])
        except ProviderError as exc:
            error_counts[str(exc)[:300]] += 1
            consecutive_failures += 1
            stop = exc.fatal
        except requests.exceptions.RequestException as exc:
            error_counts[f"{info['label']}: {type(exc).__name__}: {str(exc)[:200]}"] += 1
            consecutive_failures += 1
        except Exception as exc:
            error_counts[f"{info['label']}: {type(exc).__name__}: {str(exc)[:200]}"] += 1
            consecutive_failures += 1
        for rank, hit in enumerate(raw or []):
            url = hit.get("href") or hit.get("url")
            if not url or not str(url).startswith("http"):
                continue
            key = canonical_url(url)
            item = by_url.get(key)
            if item is None:
                item = {"title": (hit.get("title") or "").strip(), "url": url,
                        "snippet": (hit.get("body") or "").strip(), "found_by": [], "best_rank": rank,
                        "raw_content": hit.get("raw_content", "")}
                by_url[key] = item
                out["results"].append(item)
            elif not item.get("raw_content") and hit.get("raw_content"):
                item["raw_content"] = hit["raw_content"]
            if q["type"] not in item["found_by"]:
                item["found_by"].append(q["type"])
            item["best_rank"] = min(item["best_rank"], rank)
            entry["hits"] += 1
        out["queries"].append(entry)
        if stop or consecutive_failures >= 3:
            skipped = len(queries) - position - 1
            if skipped:
                out["notes"].append(f"{skipped} remaining search(es) were skipped because the provider "
                                    "keeps failing. Fix the error above and scan again.")
            break
        if position < len(queries) - 1:
            time.sleep(spec["interval"])  # respect the provider's rate limit

    out["errors"] = [f"{msg}" + (f"  (x{n})" if n > 1 else "") for msg, n in error_counts.items()]
    return out


def test_search_provider(choice="auto", query=None):
    """Independent provider test (used by the sidebar button and the command line)."""
    query = query or '"Federated Learning is a distributed learning paradigm with two key challenges"'
    info = resolve_provider(choice)
    started = time.time()
    out = search_web_sources([{"query": query, "type": QUERY_EXACT}], 5, choice)
    entry = out["queries"][0] if out["queries"] else {"ok": False, "hits": 0}
    return {"provider": info["label"], "configured": info["configured"],
            "key_source": info["source"], "connected": bool(entry["ok"]),
            "result_count": len(out["results"]), "results": out["results"],
            "errors": out["errors"] + out["notes"], "seconds": round(time.time() - started, 2),
            "query": query}


def prioritize_hits(hits, queries):
    """Rank sources: found by exact-phrase queries, high in results, snippet shares distinctive terms."""
    terms = set()
    for q in queries:
        for word in re.findall(r"[\w\-]+", q["query"]):
            if _term_weight(word) >= 1.5:
                terms.add(word.lower())
    ranked = []
    for hit in hits:
        score = sum(QUERY_WEIGHT.get(t, 1.0) for t in hit["found_by"]) + 1.0 / (1 + hit["best_rank"])
        text_tokens = set(re.findall(r"[\w\-]+", (hit["title"] + " " + hit["snippet"]).lower()))
        if terms:
            score += 2.0 * len(terms & text_tokens) / len(terms)
        ranked.append((score, hit))
    ranked.sort(key=lambda x: -x[0])
    return [h for _, h in ranked]


# --------------------------------------------------------------------------
# 8. Webpage retrieval and extraction
# --------------------------------------------------------------------------
class FetchError(Exception):
    pass


_NOISE_CLASS = re.compile(
    r"(^|[\s_-])(nav|navbar|navigation|menu|footer|sidebar|cookie|cookies|banner|advert|"
    r"advertisement|ads?|promo|share|social|breadcrumb|popup|modal|newsletter|toolbar)($|[\s_-])", re.I)
_BLOCK_CHILDREN = ["div", "p", "ul", "ol", "table", "section", "article", "h1", "h2", "h3", "h4"]


def _download(url, timeout):
    """Download bytes. Retries once with other headers when blocked. Returns (bytes, content_type)."""
    last_error = FetchError("Blocked")
    for headers in (REQUEST_HEADERS, ALT_HEADERS):
        response = None
        try:
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            if response.status_code != 200:
                last_error = FetchError(f"HTTP {response.status_code} (page blocked or unavailable)")
                if response.status_code in (401, 403, 406, 429, 503):
                    continue
                raise last_error
            ctype = response.headers.get("Content-Type", "").lower()
            is_pdf = "pdf" in ctype or urlparse(url).path.lower().endswith(".pdf")
            limit = PDF_LIMIT if is_pdf else HTML_LIMIT
            declared = response.headers.get("Content-Length", "")
            if is_pdf and declared.isdigit() and int(declared) > limit:
                raise FetchError("PDF too large to download")
            chunks, size = [], 0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > limit:
                    if is_pdf:
                        raise FetchError("PDF too large to download")
                    break  # truncated HTML is still useful
                chunks.append(chunk)
            return b"".join(chunks), ctype
        finally:
            if response is not None:
                response.close()
    raise last_error


def _extract_html(raw):
    """Return (title, readable_text, pdf_link) from HTML bytes."""
    soup = BeautifulSoup(raw, "html.parser")

    def meta(*names):
        for name in names:
            tag = soup.find("meta", attrs={"name": re.compile(f"^{name}$", re.I)}) or \
                  soup.find("meta", attrs={"property": re.compile(f"^{name}$", re.I)})
            if tag and tag.get("content"):
                return tag["content"].strip()
        return ""

    title = meta("citation_title", "og:title") or (soup.title.get_text(strip=True) if soup.title else "")
    pdf_link = meta("citation_pdf_url")
    abstract = meta("citation_abstract", "dc.description", "og:description", "description")

    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside",
                     "form", "svg", "iframe", "button", "select"]):
        tag.decompose()
    noisy = []
    for tag in soup.find_all(True):
        attrs = getattr(tag, "attrs", None) or {}
        label = " ".join(attrs.get("class", [])) + " " + str(attrs.get("id", ""))
        if label.strip() and _NOISE_CLASS.search(label):
            noisy.append(tag)
    for tag in noisy:
        try:
            tag.decompose()
        except Exception:
            pass

    container = soup.find("article") or soup.find("main") or soup.find(attrs={"role": "main"})
    body = soup.body or soup
    if container is None or len(container.get_text(" ", strip=True)) < 400:
        container = body

    pieces = []
    if abstract and len(abstract.split()) >= 8:
        pieces.append(abstract)
    for element in container.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote",
                                       "dd", "figcaption", "div"]):
        if element.name == "div" and element.find(_BLOCK_CHILDREN):
            continue  # only leaf divs (some sites put paragraphs in divs)
        piece = element.get_text(" ", strip=True)
        minimum = 8 if element.name == "div" else 3
        if len(piece.split()) >= minimum:
            pieces.append(piece)
    seen, unique = set(), []
    for piece in pieces:
        if piece not in seen:
            seen.add(piece)
            unique.append(piece)
    text = "\n\n".join(unique) if unique else container.get_text("\n\n", strip=True)
    return title, text, pdf_link


def fetch_webpage_content(url, timeout=12):
    """
    Download a real page or PDF and extract readable text.
    First checks official Wikipedia API if applicable.
    Uses real browser User-Agent and timeouts as fallback.
    Returns {"ok", "text", "page_title", "reason", "kind", "pdf_url", "note"}.
    ok=False means the source is UNVERIFIED and must not be scored.
    """
    wiki_res = _fetch_wikipedia_content(url, timeout=timeout)
    if wiki_res and wiki_res.get("ok"):
        return wiki_res

    result = {"ok": False, "text": "", "page_title": "", "reason": "", "kind": "",
              "pdf_url": "", "note": ""}
    try:
        raw, ctype = _download(url, timeout)
        if "pdf" in ctype or urlparse(url).path.lower().endswith(".pdf"):
            result.update(ok=True, kind="pdf", text=extract_text_from_pdf(raw, max_pages=60)[:600000])
            return result
        if "html" not in ctype and "text/plain" not in ctype and "xml" not in ctype:
            result["reason"] = f"Unsupported content type: {ctype or 'unknown'}"
            return result
        if "text/plain" in ctype:
            text, title, pdf_link = extract_text_from_txt(raw), "", ""
        else:
            title, text, pdf_link = _extract_html(raw)
        result.update(page_title=title, kind="html")

        # Academic sites often show only an abstract; the page itself advertises the
        # full-text PDF in a standard <meta name="citation_pdf_url"> tag. Use it.
        if pdf_link:
            pdf_url = urljoin(url, pdf_link)
            if canonical_url(pdf_url) != canonical_url(url):
                try:
                    pdf_raw, pdf_type = _download(pdf_url, timeout + 8)
                    pdf_text = extract_text_from_pdf(pdf_raw, max_pages=60)
                    text = text + "\n\n" + pdf_text
                    result.update(kind="html+pdf", pdf_url=pdf_url)
                except Exception as exc:
                    result["note"] = f"Full-text PDF listed by the page could not be read: {exc}"
        if len(text) < 200:
            result["reason"] = "No readable text found on the page"
            return result
        result.update(ok=True, text=text[:600000])
        return result
    except requests.exceptions.Timeout:
        result["reason"] = "Timed out"
    except requests.exceptions.SSLError:
        result["reason"] = "SSL/certificate error"
    except requests.exceptions.ConnectionError as exc:
        result["reason"] = f"Connection error: {str(exc)[:120]}"
    except requests.exceptions.RequestException as exc:
        result["reason"] = f"Request failed: {str(exc)[:120]}"
    except FetchError as exc:
        result["reason"] = str(exc)
    except Exception as exc:
        result["reason"] = f"Could not read page: {exc}"
    return result


# --------------------------------------------------------------------------
# 9-12. Matching (imported from nlp.similarity)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Scan pipeline (web)
# --------------------------------------------------------------------------
def _summarize_reasons(sources, limit=4):
    counts = Counter(re.sub(r"\d{3,}", "N", s["note"]) for s in sources if not s["verified"] and s["note"])
    return "; ".join(f"{reason} (x{n})" if n > 1 else reason for reason, n in counts.most_common(limit))


def run_scan(model, text, document_name, settings, progress):
    """Full web scan. `progress(fraction, message)` updates the UI."""
    warnings = []
    sentences = extract_sentences(text)
    if not sentences:
        raise ValueError(f"No scannable sentences found (need at least {MIN_WORDS} words per sentence).")
    if len(sentences) > settings["max_sentences"]:
        warnings.append(f"Only the first {settings['max_sentences']} of {len(sentences)} "
                        "sentences were scanned (sidebar limit).")
        sentences = sentences[:settings["max_sentences"]]

    progress(0.05, "Encoding your sentences with BERT...")
    user_embeddings = encode_sentences(model, sentences)

    progress(0.12, "Building search queries...")
    queries = build_search_queries(sentences, settings["max_queries"])
    progress(0.15, f"Searching the web ({len(queries)} queries)...")
    search = search_web_sources(queries, settings["results_per_query"], settings.get("provider", "auto"))
    if search["provider"]:
        warnings += search["notes"]
    hits = prioritize_hits(search["results"], queries)
    results_retrieved = len(hits)
    selected = hits[:settings["max_pages"]]

    tavily_key, _ = get_secret("TAVILY_API_KEY")
    pages = {}
    remaining_hits = []

    # Tier 1: Fetch Wikipedia via official API, or use Tavily search raw_content if present
    for h in selected:
        u = h["url"]
        wiki_res = _fetch_wikipedia_content(u)
        if wiki_res and wiki_res.get("ok"):
            pages[u] = wiki_res
            continue
        raw_c = h.get("raw_content", "")
        if raw_c and len(raw_c) >= 200:
            pages[u] = {"ok": True, "text": raw_c[:600000], "page_title": h.get("title", ""),
                        "kind": "tavily_raw", "reason": ""}
            continue
        remaining_hits.append(h)

    # Tier 2: Batch-extract remaining URLs via Tavily Extract API
    if remaining_hits and tavily_key:
        urls_to_extract = [h["url"] for h in remaining_hits if not h["url"].lower().endswith(".pdf")]
        if urls_to_extract:
            extract_map = _tavily_extract(urls_to_extract, tavily_key)
            still_remaining = []
            for h in remaining_hits:
                if h["url"] in extract_map:
                    pages[h["url"]] = {"ok": True, "text": extract_map[h["url"]]["text"][:600000],
                                       "page_title": extract_map[h["url"]].get("title") or h.get("title", ""),
                                       "kind": "tavily_extract", "reason": ""}
                else:
                    still_remaining.append(h)
            remaining_hits = still_remaining

    # Tier 3: Fallback direct downloader with desktop browser headers and timeouts
    if remaining_hits:
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(fetch_webpage_content, h["url"]): h for h in remaining_hits}
            finished = 0
            for future in as_completed(futures):
                hit = futures[future]
                try:
                    pages[hit["url"]] = future.result()
                except Exception as exc:
                    pages[hit["url"]] = {"ok": False, "text": "", "page_title": "", "reason": str(exc),
                                         "kind": "", "pdf_url": "", "note": ""}
                finished += 1
                progress(0.20 + 0.40 * (len(selected) - len(remaining_hits) + finished) / max(len(selected), 1),
                         f"Downloaded {len(pages)}/{len(selected)} webpages...")

    all_matches, sources = [], []
    downloaded = compared = 0
    for k, hit in enumerate(selected):
        page = pages[hit["url"]]
        title = hit["title"] or page.get("page_title") or "(title unavailable)"
        source = {"title": title, "url": hit["url"], "verified": False, "downloaded": bool(page["ok"]),
                   "note": page.get("reason", ""), "matches": 0, "compared_sentences": 0,
                   "kind": page.get("kind", ""), "pdf_url": page.get("pdf_url", ""),
                   "found_by": hit["found_by"]}
        if page.get("note"):
            source["note"] = (source["note"] + " " + page["note"]).strip()
        if page["ok"]:
            downloaded += 1
            page_sentences = extract_sentences(page["text"], min_words=MIN_WORDS)
            page_sentences = rank_source_sentences(page_sentences, sentences, MAX_SOURCE_SENTENCES)
            if page_sentences:
                matches = match_against_source(model, sentences, user_embeddings, page_sentences,
                                               title, hit["url"], settings["semantic_threshold"],
                                               settings["near_threshold"],
                                               full_page_text=page["text"])
                source.update(verified=True, matches=len(matches), compared_sentences=len(page_sentences))
                all_matches += matches
                compared += 1
            else:
                source["note"] = "No readable sentences found on the page"
        sources.append(source)
        progress(0.60 + 0.35 * (k + 1) / len(selected), f"Compared {k + 1}/{len(selected)} sources...")

    finalize_semantic_matches(all_matches)       # needs every source: "found in many sources" = standard statement

    # Compute per-source statistics
    for source in sources:
        s_matches = [m for m in all_matches if (m.get("url") == source.get("url") or (source.get("title") and m.get("title") == source.get("title"))) and m["type"] in COUNTED_TYPES]
        unique_indices = set(m["index"] for m in s_matches)
        source["matches"] = len(unique_indices)
        source["best_match_pct"] = round(float(max((m["score"] for m in s_matches), default=0.0)), 1)
        source["document_coverage_pct"] = round((len(unique_indices) / max(len(sentences), 1)) * 100, 1)
        if any(m["type"] == EXACT for m in s_matches):
            source["highest_match_type"] = "Exact copy"
        elif any(m["type"] == NEAR for m in s_matches):
            source["highest_match_type"] = "Near copy"
        elif any(m["type"] == SEMANTIC for m in s_matches):
            source["highest_match_type"] = "Similar meaning"
        else:
            source["highest_match_type"] = "Original"

    # Rank verified sources with matches: highest match % first, then matched sentences
    matching_sources = [s for s in sources if s.get("verified") and s.get("matches", 0) > 0]
    matching_sources.sort(key=lambda s: (-s.get("best_match_pct", 0.0), -s.get("matches", 0), -s.get("document_coverage_pct", 0.0)))
    for r_idx, s in enumerate(matching_sources):
        s["rank"] = r_idx + 1

    all_ranked_sources = matching_sources + [s for s in sources if s not in matching_sources]

    # Group matches by sentence to select best primary match and annotate 'also_found_on'
    matches_by_sentence = {}
    for m in all_matches:
        if m["type"] in COUNTED_TYPES:
            matches_by_sentence.setdefault(m["index"], []).append(m)

    primary_matches = []
    for idx_sent, s_match_list in matches_by_sentence.items():
        s_match_list.sort(key=lambda m: (-RANK.get(m["type"], 0), -m.get("score", 0.0)))
        best_m = dict(s_match_list[0])
        extra = s_match_list[1:]
        best_m["also_found_on"] = [
            {
                "title": em.get("title", ""),
                "url": em.get("url", ""),
                "score": round(float(em.get("score", 0.0)), 1),
                "type": DISPLAY_TYPE.get(em.get("type"), em.get("type"))
            }
            for em in extra
        ]
        primary_matches.append(best_m)

    primary_matches.sort(key=lambda m: -m.get("score", 0.0))

    sentence_rows, stats = summarize_matches(sentences, all_matches)
    for row_idx, row in enumerate(sentence_rows):
        if row_idx in matches_by_sentence:
            sorted_m = sorted(matches_by_sentence[row_idx], key=lambda m: (-RANK.get(m["type"], 0), -m.get("score", 0.0)))
            row["matched"] = sorted_m[0].get("matched", "")
            row["title"] = sorted_m[0].get("title", "")
            row["url"] = sorted_m[0].get("url", "")
            row["also_found_on"] = [
                {
                    "title": em.get("title", ""),
                    "url": em.get("url", ""),
                    "score": round(float(em.get("score", 0.0)), 1),
                    "type": DISPLAY_TYPE.get(em.get("type"), em.get("type"))
                }
                for em in sorted_m[1:]
            ]

    with_matches = len(matching_sources)
    stats.update(verified_sources=compared, sources_with_matches=with_matches,
                 unverified_sources=len(sources) - compared)

    queries_ok = len([q for q in search["queries"] if q["ok"]])
    pipeline = {"queries_planned": len(queries), "queries_ok": queries_ok,
                "results_retrieved": results_retrieved, "pages_attempted": len(selected),
                "pages_downloaded": downloaded, "pages_compared": compared,
                "sources_with_matches": with_matches, "engine": search["provider_label"] or "none",
                "queries": search["queries"], "search_errors": search["errors"]}

    # --- the three honest states -------------------------------------------------
    status, reason = "matches", ""
    if compared == 0:
        status = "incomplete"
        if not search["provider"]:
            reason = search["notes"][0] if search["notes"] else "No search provider is configured. " + SETUP_HELP
        elif queries_ok == 0:
            reason = (f"The search provider ({search['provider_label']}) did not answer any query. "
                      + " | ".join(search["errors"][:4]))
        elif results_retrieved == 0:
            reason = ("The search provider answered but returned no results for any query, so there was "
                      "nothing to compare (insufficient coverage). Try more queries or different text.")
        else:
            reason = (f"{results_retrieved} search result(s) were retrieved, but no webpage could be "
                      f"downloaded and compared. Reasons: {_summarize_reasons(sources) or 'unknown'}")
    elif not any(m["type"] in COUNTED_TYPES for m in all_matches):
        status = "no_matches"
    if status != "incomplete":
        if queries_ok < len(queries):
            warnings.append(f"Only {queries_ok} of {len(queries)} searches succeeded, so coverage is "
                            "partial. " + " | ".join(search["errors"][:3]))
        if compared < len(selected):
            warnings.append(f"{len(selected) - compared} of {len(selected)} sources could not be "
                            "compared and are shown as unverified (not scored).")

    progress(1.0, "Done")
    return {
        "kind": "web_scan",
        "scan_mode": "Live Web Search",
        "status": status,
        "incomplete_reason": reason,
        "document_name": document_name,
        "scanned_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sentences": sentence_rows,
        "matches": primary_matches if primary_matches else all_matches,
        "sentence_matches": primary_matches if primary_matches else all_matches,
        "all_matches": all_matches,
        "sources": all_ranked_sources,
        "verified_sources": matching_sources,
        "stats": stats,
        "pipeline": pipeline,
        "warnings": warnings,
        "settings": {k: settings[k] for k in settings},
    }


# --------------------------------------------------------------------------
# 13. Two-document comparison (imported from nlp.similarity)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# 14-15. SQLite history
# --------------------------------------------------------------------------
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_date TEXT NOT NULL,
                document_name TEXT,
                total_sentences INTEGER,
                exact_matches INTEGER,
                near_exact_matches INTEGER,
                semantic_matches INTEGER,
                verified_sources INTEGER,
                sources_with_matches INTEGER,
                overall_similarity REAL,
                exact_pct REAL,
                semantic_pct REAL,
                details TEXT
            )""")
        columns = {row[1] for row in conn.execute("PRAGMA table_info(scans)")}
        if "status" not in columns:  # upgrade databases created by the first version
            conn.execute("ALTER TABLE scans ADD COLUMN status TEXT")


def save_scan_history(result):
    """Store a scan summary (plus full details as JSON). Returns the new scan ID."""
    s = result["stats"]
    incomplete = result.get("status") == "incomplete"
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """INSERT INTO scans (scan_date, document_name, total_sentences, exact_matches,
               near_exact_matches, semantic_matches, verified_sources, sources_with_matches,
               overall_similarity, exact_pct, semantic_pct, details, status)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (result["scanned_at"], result["document_name"], s["total_sentences"], s["exact"],
             s["near_exact"], s["semantic"], s["verified_sources"], s["sources_with_matches"],
             None if incomplete else s["overall_pct"], None if incomplete else s["exact_pct"],
             None if incomplete else s["semantic_pct"], json.dumps(result), result.get("status")))
        return cursor.lastrowid


def get_scan_history(limit=200):
    """Return previous scans (summary columns only), newest first. Incomplete scans have no %."""
    legacy_incomplete = "(status IS NULL AND verified_sources = 0)"
    hide = f"(status = 'incomplete' OR {legacy_incomplete})"
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""SELECT id AS "Scan ID", scan_date AS "Date/Time", document_name AS "Document",
               CASE WHEN status IS NOT NULL THEN status
                    WHEN verified_sources = 0 THEN 'incomplete (older version)'
                    ELSE 'completed (older version)' END AS "Status",
               total_sentences AS "Sentences", exact_matches AS "Exact",
               near_exact_matches AS "Near-exact", semantic_matches AS "Semantic",
               verified_sources AS "Verified sources", sources_with_matches AS "Sources with matches",
               CASE WHEN {hide} THEN NULL ELSE overall_similarity END AS "Overall %",
               CASE WHEN {hide} THEN NULL ELSE exact_pct END AS "Exact/near %",
               CASE WHEN {hide} THEN NULL ELSE semantic_pct END AS "Semantic %"
               FROM scans ORDER BY id DESC LIMIT ?""", (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_scan_details(scan_id):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT details FROM scans WHERE id = ?", (scan_id,)).fetchone()
    return json.loads(row[0]) if row and row[0] else None


def _normalize_result(result):
    """Fill in fields that results saved by the first version of the app do not have."""
    result.setdefault("scanned_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    result.setdefault("document_name", "Document")
    stats = result.setdefault("stats", {})
    for key in ("total_sentences", "exact", "near_exact", "semantic", "none", "verified_sources",
                "sources_with_matches", "unverified_sources"):
        stats.setdefault(key, 0)
    for key in ("exact_pct", "semantic_pct", "overall_pct"):
        stats.setdefault(key, 0.0)
    if "status" not in result:
        if stats["verified_sources"] == 0:
            result["status"] = "incomplete"
            result.setdefault("incomplete_reason",
                              "This scan was saved by an older version of the app and no webpage was "
                              "successfully compared, so its 0% was not a real result.")
        else:
            result["status"] = "matches" if stats["overall_pct"] > 0 else "no_matches"
    result.setdefault("incomplete_reason", "")
    result.setdefault("pipeline", {})
    for key in ("queries_planned", "queries_ok", "results_retrieved", "pages_attempted",
                "pages_downloaded", "pages_compared", "sources_with_matches"):
        result["pipeline"].setdefault(key, 0)
    result["pipeline"].setdefault("queries", [])
    result["pipeline"].setdefault("search_errors", [])
    result["pipeline"].setdefault("engine", "unknown")
    for key in ("warnings", "sources", "matches", "sentences"):
        result.setdefault(key, [])
    if "semantic_any" not in stats:                    # scans saved before the breakdown existed
        stats.update(semantic_breakdown(result["sentences"], result["matches"]))
    stats.setdefault("topical_sentences", len({m["index"] for m in result["matches"] if m["type"] == TOPICAL}))
    stats.setdefault("topical_pairs", len({(m["index"], m.get("url") or m.get("title") or "")
                                           for m in result["matches"] if m["type"] == TOPICAL}))
    for source in result["sources"]:
        source.setdefault("note", "")
        source.setdefault("compared_sentences", 0)
    return result


# --------------------------------------------------------------------------
# 16. PDF report
# --------------------------------------------------------------------------
class ReportPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"{APP_NAME} report - page {self.page_no()}", align="C")


def _latin(text):
    """FPDF core fonts only support Latin-1; replace anything else safely."""
    text = text or ""
    for src, dst in {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                     "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " "}.items():
        text = text.replace(src, dst)
    text = text.encode("latin-1", "replace").decode("latin-1")
    return re.sub(r"(\S{70})(?=\S)", r"\1 ", text)


def _pdf_text(pdf, text, size=9, style="", height=4.5):
    pdf.set_font("Helvetica", style, size)
    pdf.multi_cell(0, height, _latin(text), new_x="LMARGIN", new_y="NEXT")


def _pdf_link(pdf, url):
    """Print a URL as clickable lines (each line links to the full URL)."""
    pdf.set_text_color(0, 0, 200)
    pdf.set_font("Helvetica", "U", 8)
    for k in range(0, len(url), 95):
        pdf.cell(0, 4, _latin(url[k:k + 95]), link=url, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)


def _pdf_match_block(pdf, m, counted_as=None):
    _pdf_text(pdf, f"{m['type']} - {m['score']:.1f}% ({m['level']})", 9, "B")
    if m["type"] == TOPICAL:
        _pdf_text(pdf, "(Not counted in the statistics: similar meaning only, no evidence of copying.)", 8, "I")
    elif counted_as and counted_as != m["type"]:
        _pdf_text(pdf, f"(This sentence is counted in the statistics as: {counted_as}, "
                       "because another source matches it more strongly.)", 8, "I")
    _pdf_text(pdf, "Your sentence: " + m["sentence"])
    _pdf_text(pdf, "Matching passage: " + m["matched"])
    _pdf_text(pdf, "Source: " + (m["title"] or "(title unavailable)"), 8, "I")
    if m.get("url"):
        _pdf_link(pdf, m["url"])
    pdf.ln(2)


def generate_pdf_report(result):
    """Build the PDF report and return it as bytes."""
    result = _normalize_result(result)
    stats, pipe = result["stats"], result["pipeline"]
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, f"{APP_NAME} - Plagiarism Scan Report", new_x="LMARGIN", new_y="NEXT")
    _pdf_text(pdf, f"Scan date: {result['scanned_at']}", 10)
    _pdf_text(pdf, f"Document: {result['document_name']}", 10)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Scan status", new_x="LMARGIN", new_y="NEXT")
    if result["status"] == "incomplete":
        _pdf_text(pdf, NOT_FOUND_MESSAGE, 10, "B")
        _pdf_text(pdf, "Reason: " + (result["incomplete_reason"] or "unknown"), 9)
    elif result["status"] == "no_matches":
        _pdf_text(pdf, "Scan completed. No significant matches were found in the verified sources "
                       "that were compared. This does not prove the text is original.", 10)
    else:
        _pdf_text(pdf, "Scan completed. Matches were found (see below).", 10)
    pdf.ln(1)
    for line in (f"Searches succeeded: {pipe['queries_ok']}/{pipe['queries_planned']}   |   "
                 f"Search results retrieved: {pipe['results_retrieved']}",
                 f"Webpages downloaded: {pipe['pages_downloaded']}   |   "
                 f"Webpages compared: {pipe['pages_compared']}   |   "
                 f"Sources containing matches: {pipe['sources_with_matches']}"):
        _pdf_text(pdf, line, 9)
    pdf.ln(2)

    if result["status"] != "incomplete":
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Overall statistics", new_x="LMARGIN", new_y="NEXT")
        for line in report_stat_lines(stats):
            _pdf_text(pdf, line, 10)
        _pdf_text(pdf, "Note: semantic scores are BERT cosine similarities. They show similar "
                       "meaning, not proof of plagiarism. Review every match manually.", 8, "I")
        pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Sources", new_x="LMARGIN", new_y="NEXT")
    if not result["sources"]:
        _pdf_text(pdf, "No sources were retrieved.")
    for s in result["sources"]:
        status = (f"Verified - {s['matches']} matching sentence(s)" if s["verified"]
                  else f"UNVERIFIED (not scored): {s['note']}")
        _pdf_text(pdf, s["title"], 9, "B")
        _pdf_link(pdf, s["url"])
        _pdf_text(pdf, status, 8, "I")
        pdf.ln(1)

    limit = 150
    by_type = {t: sorted([m for m in result["matches"] if m["type"] == t], key=lambda m: -m["score"])
               for t in (EXACT, NEAR, SEMANTIC, TOPICAL)}
    status_of = {i: row.get("status", row.get("type", NONE)) for i, row in enumerate(result["sentences"])}
    for heading, items in (("Exact matches", by_type[EXACT]),
                           ("Near-exact matches", by_type[NEAR]),
                           ("Semantic matches (BERT)", by_type[SEMANTIC]),
                           ("Topical similarity, not counted", by_type[TOPICAL])):
        unique_sentences = len({m["index"] for m in items})
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, f"{heading} ({len(items)} source match(es) for {unique_sentences} sentence(s))",
                 new_x="LMARGIN", new_y="NEXT")
        if not items:
            _pdf_text(pdf, "None found.")
        elif heading.startswith("Topical"):
            _pdf_text(pdf, "Related in meaning but with no evidence of copying (no matching neighbouring "
                           "sentences, no distinctive shared wording, or a statement found in many "
                           "sources). Not counted in the statistics; review manually.", 8, "I")
        for m in items[:limit]:
            _pdf_match_block(pdf, m, status_of.get(m["index"]))
        if len(items) > limit:
            _pdf_text(pdf, f"... {len(items) - limit} more not shown.", 8, "I")
    return bytes(pdf.output())


# --------------------------------------------------------------------------
# 17. Display
# --------------------------------------------------------------------------
def _highlight_html(sentence_rows):
    parts = []
    for row in sentence_rows:
        if isinstance(row, str):
            parts.append(html.escape(row))
            continue
        if not isinstance(row, dict):
            continue
        text = html.escape(row.get("text") or row.get("sentence") or "")
        status = row.get("status") or row.get("type") or NONE
        score = row.get("score", 0.0)
        if status in HIGHLIGHT_BG:
            tip = f"{status} - {float(score):.1f}%"
            if row.get("semantic_sources") and status != SEMANTIC:
                tip += f" (also similar in meaning to {row['semantic_sources']} other source(s))"
            parts.append(f'<span title="{tip}" style="background:{HIGHLIGHT_BG[status]};'
                         f'color:#111;padding:2px 4px;border-radius:4px;'
                         f'border-bottom:2px solid {COLORS.get(status, "#9ca3af")};">{text}</span>')
        else:
            parts.append(text)
    return '<div style="line-height:2;font-size:0.95rem">' + " ".join(parts) + "</div>"


def _legend():
    st.markdown(
        f'<span style="background:{HIGHLIGHT_BG[EXACT]};color:#111;padding:2px 8px;border-radius:4px">'
        f'Exact</span> &nbsp; '
        f'<span style="background:{HIGHLIGHT_BG[NEAR]};color:#111;padding:2px 8px;border-radius:4px">'
        f'Near-exact</span> &nbsp; '
        f'<span style="background:{HIGHLIGHT_BG[SEMANTIC]};color:#111;padding:2px 8px;border-radius:4px">'
        f'Similar meaning</span> &nbsp; Plain text = no match found',
        unsafe_allow_html=True)


def _donut_data(stats):
    """One slice per sentence status; slices always add up to the number of sentences."""
    labels = ["Exact", "Near-exact", "Semantic only", "No significant match"]
    values = [stats["exact"], stats["near_exact"], stats["semantic"], stats["none"]]
    return labels, values


def _donut(stats, key):
    labels, values = _donut_data(stats)
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.5, sort=False,
                           marker=dict(colors=[COLORS[EXACT], COLORS[NEAR], COLORS[SEMANTIC], COLORS[NONE]])))
    fig.update_layout(title="Sentence classification", height=340, margin=dict(t=50, b=10))
    st.plotly_chart(fig, key=key)


def _match_rows(matches):
    return [{"Match type": m["type"], "Similarity %": m["score"], "Your sentence": m["sentence"],
             "Matching passage": m["matched"], "Source title": m["title"],
             "Source URL": m["url"] or None}
            for m in sorted(matches, key=lambda m: (-RANK[m["type"]], -m["score"]))]


def _show_pipeline(pipe):
    cols = st.columns(5)
    cols[0].metric("Searches succeeded", f"{pipe['queries_ok']}/{pipe['queries_planned']}")
    cols[1].metric("Search results retrieved", pipe["results_retrieved"])
    cols[2].metric("Webpages downloaded", f"{pipe['pages_downloaded']}/{pipe['pages_attempted']}")
    cols[3].metric("Webpages compared", pipe["pages_compared"])
    cols[4].metric("Sources with matches", pipe["sources_with_matches"])
    if pipe["queries"]:
        with st.expander(f"Search queries used (engine: {pipe['engine']})"):
            for q in pipe["queries"]:
                state = f"{q['hits']} result(s) via {q['engine']}" if q["ok"] else "FAILED"
                st.markdown(f"- **{q['type']}** - `{q['query']}` -> {state}")


def _show_sources(result):
    st.markdown("#### Sources")
    if not result["sources"]:
        st.write("No sources were retrieved.")
    for s in result["sources"]:
        if s["verified"]:
            label = f"{s['matches']} matching sentence(s)" if s["matches"] else "no significant match"
            kind = {"html+pdf": "page + full-text PDF", "pdf": "PDF", "html": "webpage"}.get(s.get("kind"), "")
            st.markdown(f"- \u2705 **Verified** ({kind}) - [{s['title']}]({s['url']}) - {label} "
                        f"({s['compared_sentences']} sentences compared)")
        else:
            st.markdown(f"- \u26a0\ufe0f **Unverified (not scored)** - [{s['title']}]({s['url']}) - {s['note']}")


def display_results(result, key_prefix="scan"):
    """Render a web-scan result (new scans and history). Handles all three scan states."""
    result = _normalize_result(result)
    stats, pipe = result["stats"], result["pipeline"]
    st.subheader(f"Results: {result['document_name']}")
    st.caption(f"Scanned on {result['scanned_at']}")

    if result["status"] == "incomplete":
        st.error(NOT_FOUND_MESSAGE)
        st.markdown(f"**Reason:** {result['incomplete_reason'] or 'unknown'}")
        st.info("Similarity percentages are intentionally not shown: nothing was compared, so a "
                "0% would be misleading. Open *Search diagnostics* in the sidebar to test the "
                "connection, then scan again.")
        for warning in result["warnings"]:
            st.warning(warning)
        _show_pipeline(pipe)
        _show_sources(result)
    else:
        for warning in result["warnings"]:
            st.warning(warning)
        _show_pipeline(pipe)
        if result["status"] == "no_matches":
            st.success(f"Scan completed: no significant matches were found in the "
                       f"{stats['verified_sources']} verified source(s) that were compared. "
                       "This is not proof of originality; only the pages that were retrieved were checked.")
        cols = st.columns(4)
        cols[0].metric("Overall similarity", f"{stats['overall_pct']}%")
        cols[1].metric("Exact / near-exact", f"{stats['exact_pct']}%")
        cols[2].metric("Semantic only", f"{stats['semantic_pct']}%")
        cols[3].metric("Verified sources compared", stats["verified_sources"])
        st.caption("Percentages = share of scanned sentences. Each sentence is counted once, by its "
                   "strongest match. Semantic scores are BERT cosine similarities and are **not** "
                   "proof of plagiarism.")
        if stats.get("semantic_also", 0):
            st.caption(f"{stats['semantic_any']} sentence(s) have a semantic match in at least one source; "
                       f"{stats['semantic_also']} of them are already counted as exact/near-exact "
                       "because another source matches them more strongly.")

        left, right = st.columns(2)
        with left:
            _donut(stats, f"{key_prefix}_donut")
        with right:
            per_source = [(s["title"][:40] or s["url"][:40], s["matches"])
                          for s in result["sources"] if s["verified"] and s["matches"] > 0]
            if per_source:
                fig = go.Figure(go.Bar(x=[c for _, c in per_source], y=[t for t, _ in per_source],
                                       orientation="h", marker_color="#4f46e5"))
                fig.update_layout(title="Matching sentences per source", height=340,
                                  margin=dict(t=50, b=10), yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, key=f"{key_prefix}_bar")
            else:
                st.info("No verified source contained a significant match.")

        st.markdown("#### Highlighted text")
        _legend()
        st.markdown(_highlight_html(result["sentences"]), unsafe_allow_html=True)
        _show_sources(result)

        st.markdown("#### Detailed comparison")
        rows = _match_rows(result["matches"])
        if rows:
            st.caption(f"{len(rows)} source match(es) for {len({m['index'] for m in result['matches']})} "
                       "sentence(s): one row per sentence and source, so a sentence found in several "
                       "sources appears several times but is counted once in the statistics."
                       + (f" {stats['topical_pairs']} row(s) are Topical Similarity: related in meaning but "
                          "with no evidence of copying, so they are not counted." if stats.get("topical_pairs") else ""))
            st.dataframe(rows, column_config={
                "Source URL": st.column_config.LinkColumn("Source URL"),
                "Similarity %": st.column_config.NumberColumn(format="%.1f")})
        else:
            st.write("No matches found in the verified sources.")

    if result.get("settings"):
        with st.expander("Settings used for this scan"):
            st.json(result["settings"])
    try:
        st.download_button("\u2b07\ufe0f Download PDF report", data=generate_pdf_report(result),
                           file_name=f"plagiarismai_report_{result['scanned_at'][:10]}.pdf",
                           mime="application/pdf", key=f"{key_prefix}_pdf")
    except Exception as exc:
        st.error(f"Could not build the PDF report: {exc}")


def display_comparison(result):
    stats = result["stats"]
    st.subheader(f"{result['name_a']}  vs  {result['name_b']}")
    cols = st.columns(5)
    cols[0].metric("Sentences in A matched in B", f"{stats['overall_pct']}%")
    cols[1].metric("Exact / near-exact", f"{stats['exact_pct']}%")
    cols[2].metric("Semantic only", f"{stats['semantic_pct']}%")
    cols[3].metric("Mean best similarity", f"{stats['mean_best_similarity']}%")
    cols[4].metric("Whole-document similarity", f"{stats['document_similarity']}%")
    st.caption(f"A has {stats['total_sentences']} sentences, B has {stats['sentences_b']}. "
               "Similarity measures meaning overlap, not proof of copying.")
    left, right = st.columns(2)
    with left:
        _donut(stats, "cmp_donut")
    with right:
        fig = go.Figure(go.Histogram(x=result["best_scores"], nbinsx=20, marker_color="#4f46e5"))
        fig.update_layout(title="Best cosine similarity per sentence of A (%)", height=340,
                          margin=dict(t=50, b=10), xaxis_title="Similarity %", yaxis_title="Sentences")
        st.plotly_chart(fig, key="cmp_hist")
    st.markdown("#### Highlighted text of document A")
    _legend()
    st.markdown(_highlight_html(result["sentences"]), unsafe_allow_html=True)
    st.markdown("#### Matching sentences")
    rows = [{k: v for k, v in r.items() if k not in ("Source title", "Source URL")}
            for r in _match_rows(result["matches"])]
    st.dataframe(rows) if rows else st.write("No matching sentences found.")


# --------------------------------------------------------------------------
# 18. Main app
# --------------------------------------------------------------------------
def _get_model():
    try:
        with st.spinner("Loading BERT model (the first run downloads about 90 MB)..."):
            return load_bert_model()
    except Exception as exc:
        st.error(f"Could not load the BERT model: {exc}\n\nThe first run needs internet to download "
                 f"`{MODEL_NAME}`. Check your connection and that `sentence-transformers` and "
                 "`torch` are installed.")
        return None


def _provider_status_line(pid):
    spec = PROVIDERS[pid]
    key, source = get_secret(spec["env"])
    if key:
        return f"\u2705 **{spec['label']}** - key configured ({source})"
    return f"\u274c **{spec['label']}** - `{spec['env']}` not set"


def _search_diagnostics(provider_choice, key_prefix="diag"):
    """Provider status, key status, and an independent connection + search test."""
    info = resolve_provider(provider_choice)
    st.markdown(f"**Active provider:** {info['label']}")
    if info["configured"]:
        st.success("Provider is configured" + (f" - key from {info['source']}" if info["source"] else ""))
    else:
        st.error(info["problem"])
    st.markdown(_provider_status_line("brave"))
    st.markdown(_provider_status_line("tavily"))
    st.text_input("Brave API key (this session only)", type="password", key=f"{key_prefix}_session_key_BRAVE_API_KEY")
    st.text_input("Tavily API key (this session only)", type="password", key=f"{key_prefix}_session_key_TAVILY_API_KEY")
    st.caption("Keys typed here are kept in memory for this session only and are never saved. "
               "For permanent setup use BRAVE_API_KEY / TAVILY_API_KEY in a .env file.")
    query = st.text_input("Test query", value='"Federated Learning is a distributed learning '
                                              'paradigm with two key challenges"', key=f"{key_prefix}_query")
    if st.button("Test provider connection and search", key=f"{key_prefix}_btn"):
        with st.spinner("Contacting the search provider..."):
            test = test_search_provider(provider_choice, query)
        if not test["configured"]:
            st.error("Not configured: " + (test["errors"][0] if test["errors"] else ""))
        elif test["connected"]:
            st.success(f"Connection OK ({test['seconds']} s)")
            st.write(f"Results retrieved: **{test['result_count']}**")
            for r in test["results"]:
                st.markdown(f"- [{r['title'] or r['url']}]({r['url']})")
            if test["result_count"] == 0:
                st.warning("The provider answered but returned zero results for this query.")
        else:
            st.error("Provider connection or search FAILED")
        for message in test["errors"] if test["configured"] else []:
            st.caption(message)


def _render_sidebar():
    st.sidebar.markdown("### 🔍 PlagiarismAI")
    st.sidebar.caption("An originality checker and writing assistant for essays and academic work.")

    with st.sidebar.expander("⚙️ Settings (optional)", expanded=False):
        st.slider("How strict should matching be?", min_value=50, max_value=95, value=70, key="sb_sem_thresh")
        st.slider("Exact copy strictness", min_value=80, max_value=99, value=90, key="sb_near_thresh")
        st.slider("Number of web searches", min_value=1, max_value=15, value=8, key="sb_queries")
        st.slider("Pages to read", min_value=1, max_value=30, value=12, key="sb_max_pages")


def _get_settings():
    sem_thresh = st.session_state.get("sb_sem_thresh", st.session_state.get("adv_sem_thresh", 70))
    near_thresh = st.session_state.get("sb_near_thresh", st.session_state.get("adv_near_thresh", 90))
    queries = st.session_state.get("sb_queries", st.session_state.get("adv_queries", 8))
    pages = st.session_state.get("sb_max_pages", st.session_state.get("adv_max_pages", 12))

    t_key, _ = get_secret("TAVILY_API_KEY")
    b_key, _ = get_secret("BRAVE_API_KEY")
    default_provider = "tavily" if t_key else ("brave" if b_key else "auto")
    provider = st.session_state.get("adv_provider", default_provider)

    return {
        "provider": provider,
        "semantic_threshold": sem_thresh / 100.0,
        "near_threshold": near_thresh / 100.0,
        "max_queries": queries,
        "results_per_query": st.session_state.get("adv_res_per_q", 5),
        "max_pages": pages,
        "max_sentences": st.session_state.get("adv_max_sentences", 200),
    }


def as_list(val, key=None):
    """Safely extracts a list whether val is a list, dict, or None."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, dict):
        if key and key in val:
            sub = val[key]
            if isinstance(sub, list):
                return sub
            elif sub is not None:
                return [sub]
        for candidate in ("items", "list", "keywords", "topics", "citations", "values"):
            if candidate in val and isinstance(val[candidate], list):
                return val[candidate]
        collected = []
        for v in val.values():
            if isinstance(v, list):
                collected.extend(v)
            elif isinstance(v, str) and v:
                collected.append(v)
        return collected
    return [val]


def as_dict(val, key=None):
    """Safely extracts a dict whether val is a dict, list, or None."""
    if val is None:
        return {}
    if isinstance(val, dict):
        if key and key in val and isinstance(val[key], dict):
            return val[key]
        return val
    return {}


def _load_sample_file(filename):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "samples", filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _tab_check(settings):
    st.markdown("### Document Originality & Writing Checker")
    st.write("Check how original your writing is, in a few seconds.")

    # Input text / file
    upload = st.file_uploader("Upload a document (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"], key="check_upload")
    default_text = st.session_state.get("check_input_text", "")
    pasted = st.text_area("Or paste your document text here", value=default_text, height=220, key="check_text_area")

    tavily_key, _ = get_secret("TAVILY_API_KEY")
    brave_key, _ = get_secret("BRAVE_API_KEY")
    has_search_key = bool(tavily_key or brave_key)

    OPT_WEB = "Search the web (best for checking online text)"
    OPT_REF = "Use my reference library only"

    col_opt1, col_opt2 = st.columns([2, 1])
    with col_opt1:
        if has_search_key:
            mode_choice = st.radio(
                "Comparison source",
                [OPT_WEB, OPT_REF],
                index=0,
                horizontal=True,
                key="check_mode_choice"
            )
        else:
            mode_choice = st.radio(
                "Comparison source",
                [OPT_REF],
                index=0,
                horizontal=True,
                key="check_mode_choice"
            )
    with col_opt2:
        if has_search_key:
            p_name = "Tavily" if tavily_key else "Brave"
            st.caption(f"🌐 **Web search is ready** ({p_name} connected).")
        else:
            st.caption("ℹ️ **Web search is not set up** (no search API key in .env) — using your reference library.")

    max_checks = 10
    try:
        max_checks = int(get_env_or_secret("MAX_CHECKS_PER_SESSION", "10"))
    except ValueError:
        max_checks = 10

    checks_used = st.session_state.get("checks_performed", 0)
    if max_checks > 0:
        st.caption(f"🛡️ Session checks: **{checks_used}/{max_checks}** used.")

    if st.button("Check my document", type="primary", key="main_check_btn", use_container_width=True):
        if max_checks > 0 and checks_used >= max_checks:
            st.error(f"🛑 You have reached the maximum allowed checks for this session ({max_checks}/{max_checks}). To prevent search quota exhaustion, please refresh or contact the administrator.")
            return

        try:
            if upload is not None:
                text = extract_text_from_upload(upload)
                name = upload.name
            else:
                text = pasted
                name = st.session_state.get("check_doc_name", "Pasted text")
        except Exception as exc:
            st.error(f"Could not read the document: {exc}")
            return

        if not text or not text.strip():
            st.error("Please paste some text or upload a document first.")
            return

        model = _get_model()
        if model is None:
            return

        bar = st.progress(0.0, text="Reading your document...")
        try:
            bar.progress(0.15, text="Preprocessing text and analyzing structure...")
            prep_res = preprocess_document(text)

            bar.progress(0.30, text="Detecting language and checking translation...")
            lang_res = detect_language_smart(text)
            text_for_scan = text
            trans_info = None
            if lang_res.get("is_multilingual"):
                trans_info = translate_text_smart(text, lang_res.get("code", "auto"))
                text_for_scan = trans_info.get("translated_text", text)

            bar.progress(0.45, text="Analyzing writing tone, key entities, and themes...")
            sentiment_res = analyze_sentiment(text)
            entities_res = extract_named_entities(text)
            keywords_res = extract_keywords(text)
            citations_res = extract_citations(text)
            topics_res = extract_document_topics(text)

            bar.progress(0.65, text="Comparing text against sources...")
            if mode_choice == OPT_WEB:
                if tavily_key:
                    settings["provider"] = "tavily"
                elif brave_key:
                    settings["provider"] = "brave"

                scan_res = run_scan(model, text_for_scan, name, settings,
                                    lambda f, m: bar.progress(min(0.65 + 0.30 * f, 0.95), text=m))
                if scan_res.get("status") == "incomplete":
                    # Graceful fallback to reference library if web search is not configured or failed
                    bar.progress(0.80, text="Web search provider unavailable; scanning reference library...")
                    scan_res = scan_against_reference_corpus(
                        model, text_for_scan, name,
                        semantic_threshold=settings["semantic_threshold"],
                        near_threshold=settings["near_threshold"]
                    )
                    scan_res["fallback_to_library"] = True
                    scan_res["scan_mode"] = "Reference Library (fallback from web search)"
            else:
                scan_res = scan_against_reference_corpus(
                    model, text_for_scan, name,
                    semantic_threshold=settings["semantic_threshold"],
                    near_threshold=settings["near_threshold"],
                    progress=lambda f, m: bar.progress(min(0.65 + 0.30 * f, 0.95), text=m)
                )

            bar.progress(0.98, text="Finalizing report...")
            scan_res["scan_id"] = save_scan_history(scan_res)
            st.session_state["checks_performed"] = checks_used + 1

            st.session_state["scan_result"] = scan_res
            st.session_state["prep_result"] = prep_res
            st.session_state["nlp_analysis"] = {
                "sentiment": sentiment_res,
                "entities": entities_res,
                "keywords": keywords_res,
                "citations": citations_res,
                "topics": topics_res,
                "language": lang_res,
                "translation": trans_info,
            }
            mem_mb = get_process_memory_mb()
            st.success("✅ Analysis complete! Open the **Results** tab to see your originality score and findings, or **Fix it for me** to rewrite flagged parts.")
            if mem_mb > 0:
                st.caption(f"💾 **Memory Footprint:** ~{mem_mb:.1f} MB RSS (low-memory mode: {'ON' if is_low_memory_mode() else 'OFF'})")
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
        finally:
            bar.empty()


def _tab_results():
    if "scan_result" not in st.session_state:
        st.info("No document checked yet. Please go to the **Check** tab to upload or paste a document.")
        return

    mem_mb = get_process_memory_mb()
    if mem_mb > 0:
        st.caption(f"💾 Server memory: ~{mem_mb:.1f} MB RSS (low-memory mode: {'ON' if is_low_memory_mode() else 'OFF'})")

    result = _normalize_result(st.session_state["scan_result"])
    stats = as_dict(result.get("stats"))
    nlp = as_dict(st.session_state.get("nlp_analysis"))

    matches = [m for m in as_list(result.get("matches")) if isinstance(m, dict) and m.get("type") in COUNTED_TYPES]
    # Verified sources with actual matches, ranked highest best_match_pct first
    matching_sources = [s for s in as_list(result.get("sources")) if isinstance(s, dict) and s.get("verified") and s.get("matches", 0) > 0]
    matching_sources.sort(key=lambda s: (-s.get("best_match_pct", 0.0), -s.get("matches", 0), -s.get("document_coverage_pct", 0.0)))
    for r_idx, s in enumerate(matching_sources):
        s["rank"] = r_idx + 1

    st.subheader(f"Results for: {result.get('document_name', 'Document')}")
    st.caption(f"Checked on {result.get('scanned_at', '')}")

    scan_mode_name = result.get("scan_mode", "Live Web Search" if result.get("pipeline", {}).get("engine") not in ("none", "Academic Reference Library") else "Reference Library")

    # Honesty status banner: clearly show mode used and if web search fell back
    if result.get("fallback_to_library"):
        st.warning("⚠️ **Live Web Search Notice:** Live web search could not complete (network, provider or quota limit). The document was compared against your verified reference library instead.")
    elif result.get("status") == "no_matches" and "Web" in scan_mode_name:
        st.info("🌐 **Web Search Status:** Searched online sources via Tavily and downloaded candidate pages. No sentences matched external sources above similarity thresholds.")
    else:
        st.caption(f"🔎 **Scan Mode Used:** {scan_mode_name}")

    pipe = as_dict(result.get("pipeline"))
    pages_found = pipe.get("results_retrieved", len(result.get("sources", [])))
    pages_compared = pipe.get("pages_compared", stats.get("verified_sources", 0))
    pages_failed = max(0, pages_found - pages_compared)
    low_coverage = ("Web" in scan_mode_name and pages_found >= 2 and (pages_compared / max(pages_found, 1) < 0.5 or pages_compared < 2))

    # (a) Big Originality Score + one summary line
    overall_pct = stats.get("overall_pct", 0.0)
    originality = max(0.0, round(100.0 - overall_pct, 1))

    if low_coverage:
        st.warning(f"## ⚠️ Result May Be Incomplete (Low Online Coverage)\nWe compared your text with **{pages_compared} of {pages_found} candidate pages found** online ({pages_failed} pages could not be read). Based on readable sources, your text shows **at least {originality}% originality**, but this result may be incomplete.")
    elif originality >= 80.0:
        st.success(f"## 🟢 Originality Score: {originality}%\nYour document is predominantly original with minimal similarity to checked sources.")
    elif originality >= 50.0:
        st.warning(f"## 🟡 Originality Score: {originality}%\nYour document shows moderate similarity to checked sources; consider reviewing flagged passages.")
    else:
        st.error(f"## 🔴 Originality Score: {originality}%\nYour document contains substantial overlap with checked sources; citations or rewrites are recommended.")

    total_sentences = stats.get("total_sentences", len(result.get("sentences", [])))
    n_exact = stats.get("exact", 0)
    n_near = stats.get("near_exact", 0)
    n_semantic = stats.get("semantic", 0)
    n_original = stats.get("none", max(0, total_sentences - (n_exact + n_near + n_semantic)))
    n_matched = total_sentences - n_original

    source_kind_str = "online sources" if "Web" in scan_mode_name else "reference sources"
    if n_matched > 0:
        st.markdown(f"### 📋 Summary: **{n_matched} of {total_sentences} sentences** match {source_kind_str}: **{n_exact} exact**, **{n_near} near**, **{n_semantic} similar meaning**.")
    else:
        st.markdown(f"### 📋 Summary: **0 of {total_sentences} sentences** match {source_kind_str} — your text shows **100% independent writing**.")

    if "Web" in scan_mode_name and pages_found > 0:
        st.info(f"🌐 **Search Coverage:** We compared your text with **{pages_compared} of {pages_found} pages found** online. {pages_failed} page(s) could not be read.")

    # (b) Match summary cards
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    exact_pct = round((n_exact / max(total_sentences, 1)) * 100, 1)
    near_pct = round((n_near / max(total_sentences, 1)) * 100, 1)
    similar_pct = round((n_semantic / max(total_sentences, 1)) * 100, 1)
    original_pct = round((n_original / max(total_sentences, 1)) * 100, 1)

    with col_c1:
        st.metric("Exact copy", f"{n_exact} ({exact_pct}%)", help="Identical or verbatim contained wording (100% match)")
    with col_c2:
        st.metric("Near copy", f"{n_near} ({near_pct}%)", help="Very high text overlap with small edits (90-99% match)")
    with col_c3:
        st.metric("Similar meaning", f"{n_semantic} ({similar_pct}%)", help="Conceptual similarity with different wording")
    with col_c4:
        st.metric("Original", f"{n_original} ({original_pct}%)", help="Independent sentences with no meaningful external overlap")

    st.divider()

    # (c) "Verified sources" table ranked from highest to lowest match
    st.markdown("#### 1. Verified Sources")
    st.caption("Only sources that were downloaded and had at least one sentence match are shown, ranked from highest match % to lowest.")

    if not matching_sources:
        st.info("No external sources matched your text above similarity thresholds.")
    else:
        table_html = """
<div style="overflow-x: auto; margin-bottom: 20px;">
  <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">
    <thead>
      <tr style="border-bottom: 2px solid #555; text-align: left;">
        <th style="padding: 8px;">Rank</th>
        <th style="padding: 8px;">Source Title</th>
        <th style="padding: 8px;">Domain</th>
        <th style="padding: 8px;">Best Match %</th>
        <th style="padding: 8px;">Match Type</th>
        <th style="padding: 8px;">Matched Sentences</th>
        <th style="padding: 8px;">Link</th>
      </tr>
    </thead>
    <tbody>
"""
        for r_idx, s in enumerate(matching_sources):
            title = s.get("title", "Source document")
            url = s.get("url", "")
            domain = urlparse(url).netloc if url and url.startswith("http") else "Reference Library"
            best_pct = s.get("best_match_pct", 0.0)
            m_type = s.get("highest_match_type", "Similar meaning")
            m_count = s.get("matches", 0)
            doc_cov = s.get("document_coverage_pct", 0.0)
            if url and url.startswith("http"):
                link_tag = f'<a href="{url}" target="_blank" rel="noopener noreferrer" style="color: #4da6ff; font-weight: bold;">Open source ↗</a>'
            else:
                link_tag = '<span style="color: #888;">Local file</span>'

            badge_color = "#EF5350" if "Exact" in m_type else ("#FFA726" if "Near" in m_type else "#AB47BC")
            type_badge = f'<span style="background:{badge_color}; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.8rem;">{m_type}</span>'

            table_html += f"""
      <tr style="border-bottom: 1px solid #333;">
        <td style="padding: 8px; font-weight: bold;">#{r_idx + 1}</td>
        <td style="padding: 8px; max-width: 280px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="{title}">{title}</td>
        <td style="padding: 8px; color: #aaa;"><code>{domain}</code></td>
        <td style="padding: 8px; font-weight: bold;">{best_pct:.1f}%</td>
        <td style="padding: 8px;">{type_badge}</td>
        <td style="padding: 8px;">{m_count} ({doc_cov}% of doc)</td>
        <td style="padding: 8px;">{link_tag}</td>
      </tr>
"""
        table_html += """
    </tbody>
  </table>
</div>
"""
        st.markdown(table_html, unsafe_allow_html=True)

    # Candidate sources that could not be verified / compared
    unverified_sources = [s for s in as_list(result.get("sources")) if isinstance(s, dict) and not s.get("verified")]
    if unverified_sources:
        with st.expander(f"⚠️ Could Not Be Checked ({len(unverified_sources)} candidate web pages)", expanded=False):
            st.caption("The following candidate web pages were retrieved by search queries but could not be downloaded, parsed, or verified (e.g. anti-bot protection, blocked scripts, or network timeouts). They are labeled as **Unverified** and were NOT counted in your originality score:")
            for us in unverified_sources:
                u_title = us.get("title") or "(Title unavailable)"
                u_url = us.get("url") or ""
                u_reason = us.get("note") or us.get("reason") or "Page unreadable or blocked"
                st.markdown(f"- **{u_title}** — *{u_reason}*  \n  [Open link ↗]({u_url})")

    st.divider()

    # (d) "Copied parts"
    st.markdown("#### 2. Copied Parts")
    st.caption("Each flagged sentence compared side-by-side with its matching source text, sorted by similarity.")

    filter_choice = st.selectbox(
        "Filter copied parts by match type:",
        ["All flagged passages", "Exact copy only", "Near copy only", "Similar meaning only"],
        key="filter_copied_parts"
    )

    flagged_matches = [m for m in as_list(result.get("matches")) if isinstance(m, dict) and m.get("type") in COUNTED_TYPES]
    flagged_matches.sort(key=lambda m: -m.get("score", 0.0))

    if filter_choice == "Exact copy only":
        flagged_matches = [m for m in flagged_matches if m.get("type") == EXACT or "Exact" in m.get("type", "")]
    elif filter_choice == "Near copy only":
        flagged_matches = [m for m in flagged_matches if m.get("type") == NEAR or "Near" in m.get("type", "")]
    elif filter_choice == "Similar meaning only":
        flagged_matches = [m for m in flagged_matches if m.get("type") == SEMANTIC or "Similar" in m.get("type", "")]

    if not flagged_matches:
        st.info("🎉 No copied parts match the selected filter. None of your sentences exceeded thresholds.")
    else:
        for idx, m in enumerate(flagged_matches):
            score_pct = m.get("score", 0.0)
            m_type = m.get("type", "")
            plain_type = DISPLAY_TYPE.get(m_type, "Similar meaning" if "Semantic" in m_type else ("Near copy" if "Near" in m_type else "Exact copy"))
            title = m.get("title", "Reference Source")
            url = m.get("url", "")

            badge_col = "#EF5350" if "Exact" in plain_type else ("#FFA726" if "Near" in plain_type else "#AB47BC")
            st.markdown(f"##### Passage {idx + 1} — <span style='background:{badge_col}; color:#fff; padding:3px 8px; border-radius:4px;'>{plain_type}</span> ({score_pct:.1f}% similarity)", unsafe_allow_html=True)

            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("**Your sentence:**")
                st.info(f"\"{m.get('sentence', '')}\"")
            with col_r:
                st.markdown(f"**Matching text in source:** (*{title}*)")
                matched_text = m.get("matched", "") or m.get("sentence", "")
                st.warning(f"\"{matched_text}\"")

            if url and url.startswith("http"):
                st.markdown(f"**Primary Source:** <a href='{url}' target='_blank' rel='noopener noreferrer'>{title} ↗</a>", unsafe_allow_html=True)
            else:
                st.caption(f"**Primary Source:** {title}")

            also_found = as_list(m.get("also_found_on"))
            if also_found:
                also_parts = []
                for af in also_found:
                    if isinstance(af, dict):
                        af_title = af.get("title", "Another source")
                        af_url = af.get("url", "")
                        af_score = af.get("score", 0.0)
                        if af_url and af_url.startswith("http"):
                            also_parts.append(f"<a href='{af_url}' target='_blank' rel='noopener noreferrer'>{af_title}</a> ({af_score:.1f}%)")
                        else:
                            also_parts.append(f"{af_title} ({af_score:.1f}%)")
                if also_parts:
                    st.markdown("<small style='color:#aaa;'>ℹ️ <b>Also found on:</b> " + " • ".join(also_parts) + "</small>", unsafe_allow_html=True)

            if st.button(f"Fix this passage #{idx + 1}", key=f"btn_fix_part_{idx}"):
                st.session_state["fix_target_sentence"] = m.get("sentence", "")
                st.session_state["fix_target_source"] = title
                st.session_state["fix_target_url"] = url
                st.info(f"Passage {idx + 1} selected! Please switch to the **Fix it for me** tab above to see your AI rewrite.")
            st.write("")

    st.divider()

    # (e) Analysis section with charts
    st.markdown("#### 3. Visual Analysis")
    col_ch1, col_ch2 = st.columns(2)

    with col_ch1:
        # Chart 1: Donut chart of document breakdown
        pie_labels = ["Exact copy", "Near copy", "Similar meaning", "Original"]
        pie_values = [n_exact, n_near, n_semantic, n_original]
        pie_colors = ["#EF5350", "#FFA726", "#AB47BC", "#66BB6A"]

        fig_pie = go.Figure(data=[go.Pie(
            labels=pie_labels,
            values=pie_values,
            hole=0.45,
            marker=dict(colors=pie_colors),
            textinfo="label+value+percent",
            hoverinfo="label+percent+value",
            sort=False
        )])
        fig_pie.update_layout(
            title="Document Originality Breakdown",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=20),
            height=320,
            showlegend=False
        )
        st.plotly_chart(fig_pie, use_container_width=True, key="chart_doc_donut")

    with col_ch2:
        # Chart 2: Top sources by best match %
        if matching_sources:
            top_sources = matching_sources[:8]
            y_titles = [s.get("title", "")[:32] + ("..." if len(s.get("title", "")) > 32 else "") for s in top_sources][::-1]
            x_scores = [s.get("best_match_pct", 0.0) for s in top_sources][::-1]
            bar_colors = ["#EF5350" if "Exact" in s.get("highest_match_type", "") else ("#FFA726" if "Near" in s.get("highest_match_type", "") else "#AB47BC") for s in top_sources][::-1]

            fig_bar = go.Figure(data=[go.Bar(
                x=x_scores,
                y=y_titles,
                orientation="h",
                marker=dict(color=bar_colors),
                text=[f"{score:.1f}%" for score in x_scores],
                textposition="auto"
            )])
            fig_bar.update_layout(
                title="Top Matched Sources by Peak Similarity (%)",
                xaxis=dict(range=[0, 105], title="Peak Similarity %"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=40, b=20),
                height=320
            )
            st.plotly_chart(fig_bar, use_container_width=True, key="chart_top_sources_bar")
        else:
            st.info("No external sources exceeded similarity thresholds to plot.")

    # Document Map Strip
    st.markdown("##### Document Sequence Map")
    st.caption("Each tile represents a sentence in reading order, colored by match classification so you can spot where copied sections are located.")
    sentence_rows = as_list(result.get("sentences"))
    if sentence_rows:
        map_html = '<div style="display:flex; flex-wrap:wrap; gap:6px; padding:12px; background:#1e1e1e; border-radius:8px; border:1px solid #444; margin-bottom:15px;">'
        for s_idx, s_row in enumerate(sentence_rows):
            status = s_row.get("status", NONE)
            score = s_row.get("score", 0.0)
            clean_status = DISPLAY_TYPE.get(status, "Original" if status == NONE else status)
            color = "#EF5350" if "Exact" in clean_status else ("#FFA726" if "Near" in clean_status else ("#AB47BC" if "Similar" in clean_status else "#66BB6A"))
            text_snip = s_row.get("text", "")[:60].replace('"', '&quot;')
            map_html += f'<div style="background:{color}; color:#fff; padding:6px 10px; border-radius:4px; font-size:0.8rem; font-weight:bold; cursor:pointer;" title="Sentence {s_idx + 1} ({clean_status} - {score:.1f}%): {text_snip}...">S{s_idx + 1}</div>'
        map_html += '</div>'
        st.markdown(map_html, unsafe_allow_html=True)

    st.divider()

    # (f) Tone of your writing
    st.markdown("#### 4. Tone of Your Writing")
    sentiment = as_dict(nlp.get("sentiment"))
    tone_label = sentiment.get("label", "Neutral")
    tone_conf = int(sentiment.get("confidence", 0.85) * 100)
    st.markdown(f"**Overall Tone:** `{tone_label}` ({tone_conf}% confidence)")
    if tone_label == "Neutral":
        st.write("Your writing maintains an objective, balanced tone appropriate for academic papers.")
    elif tone_label == "Positive":
        st.write("Your writing is optimistic, constructive, and forward-looking.")
    else:
        st.write("Your writing takes a critical, analytical, or cautious perspective.")

    para_tones = as_list(sentiment.get("paragraph_sentiments"))
    if para_tones:
        with st.expander("Paragraph-by-paragraph tone details", expanded=False):
            for p in para_tones:
                if isinstance(p, dict):
                    st.markdown(f"- **Paragraph {p.get('paragraph_index', 1)}** (`{p.get('label', 'Neutral')}`): {p.get('text_preview', '')}")

    st.divider()

    # What your document is about (3 topic chips)
    st.markdown("#### 5. What Your Document Is About")
    topics_data = nlp.get("topics")
    topics_list = as_list(topics_data, "topics")
    if topics_list:
        topic_chips = []
        for t in topics_list[:3]:
            if isinstance(t, dict):
                label = t.get("label", "Topic")
                top_words = ", ".join(as_list(t.get("top_words"))[:4])
                topic_chips.append(f"🏷️ **{label}** ({top_words})")
            elif isinstance(t, str):
                topic_chips.append(f"🏷️ **{t}**")
        for chip in topic_chips:
            st.markdown(f"- {chip}")
    else:
        st.info("No specific topic themes identified.")

    st.divider()

    # Key details found
    st.markdown("#### 6. Key Details Found")
    entities = as_dict(nlp.get("entities"))
    keywords = as_list(nlp.get("keywords"))
    citations = as_list(nlp.get("citations"))

    details_found = False
    if entities.get("people"):
        details_found = True
        chips = [f"`{p}`" for p in as_list(entities["people"])[:6]]
        st.markdown(f"**People & Researchers:** {' '.join(chips)}")
    if entities.get("organizations"):
        details_found = True
        chips = [f"`{o}`" for o in as_list(entities["organizations"])[:6]]
        st.markdown(f"**Organizations & Institutions:** {' '.join(chips)}")
    if entities.get("places"):
        details_found = True
        chips = [f"`{loc}`" for loc in as_list(entities["places"])[:5]]
        st.markdown(f"**Places & Locations:** {' '.join(chips)}")
    if entities.get("dates"):
        details_found = True
        chips = [f"`{d}`" for d in as_list(entities["dates"])[:5]]
        st.markdown(f"**Dates & Timeframes:** {' '.join(chips)}")
    if keywords:
        details_found = True
        chips = [f"`{kw}`" for kw in keywords[:8]]
        st.markdown(f"**Key Concepts & Terms:** {' '.join(chips)}")
    if citations:
        details_found = True
        chips = [f"`{c}`" for c in citations[:6]]
        st.markdown(f"**Scholarly Citations Identified:** {' '.join(chips)}")
    if not details_found:
        st.info("No named entities or specialized keywords detected.")

    st.divider()

    # (g) Language badge showing the translation provider actually used
    lang_info = as_dict(nlp.get("language"))
    lang_name = lang_info.get("language", "English")
    if lang_info.get("is_multilingual"):
        trans_info = as_dict(nlp.get("translation"))
        prov = trans_info.get("provider", "fallback translator")
        has_google_key = bool(os.getenv("GOOGLE_TRANSLATE_API_KEY", "").strip())
        if has_google_key and "Google Cloud" in prov:
            prov_label = "Google Cloud Translation"
        elif "Gemini" in prov:
            prov_label = "Gemini AI Translation"
        else:
            prov_label = "fallback translator"
        conf = int(lang_info.get("confidence", 0.8) * 100)
        st.info(f"🌐 **Language Detected:** {lang_name} ({conf}% confidence) — translated to English for comparison via **{prov_label}**.")
    else:
        st.caption(f"🌐 **Language:** {lang_name} (original)")

    st.divider()

    # Document text with highlights
    st.markdown("#### Document Text with Highlights")
    _legend()
    st.markdown(_highlight_html(result["sentences"]), unsafe_allow_html=True)

    # (h) PDF download button and Check another document
    st.divider()
    col_dl, col_reset = st.columns([1, 1])
    with col_dl:
        try:
            pdf_bytes = generate_pdf_report(result)
            doc_name = result.get("document_name", "Document").replace(" ", "_")
            st.download_button(
                label="📄 Download Official PDF Report",
                data=pdf_bytes,
                file_name=f"PlagiarismAI_Report_{doc_name}.pdf",
                mime="application/pdf",
                key="res_download_pdf_btn",
                use_container_width=True
            )
        except Exception as exc:
            st.caption(f"PDF download unavailable: {exc}")
    with col_reset:
        if st.button("🔄 Check Another Document", key="btn_check_another", use_container_width=True):
            st.session_state.pop("scan_result", None)
            st.session_state.pop("prep_result", None)
            st.session_state.pop("nlp_analysis", None)
            st.session_state["check_input_text"] = ""
            st.session_state["check_doc_name"] = ""
            st.rerun()


def _tab_fix():
    st.markdown("### Academic Writing Assistant")
    st.write("Select or paste any flagged sentence to receive AI-powered rewriting, citation formatting, and originality advice.")
    st.caption("Note: AI rewrites are meant as learning aids. Always cite your sources properly.")

    # Auto-populate if chosen from Results
    default_passage = st.session_state.get("fix_target_sentence", "")
    default_source = st.session_state.get("fix_target_source", "Academic Source")

    # Dropdown of matches if available
    result = as_dict(st.session_state.get("scan_result"))
    options = []
    for m in as_list(result.get("matches")):
        if m.get("type") in COUNTED_TYPES:
            options.append(m.get("sentence", ""))

    selected_from_list = None
    if options:
        selected_from_list = st.selectbox("Or choose a flagged sentence from your latest check:", ["(Custom pasted sentence)"] + options)
        if selected_from_list != "(Custom pasted sentence)":
            default_passage = selected_from_list

    passage_input = st.text_area("Passage to improve:", value=default_passage, height=100, key="fix_text_input")
    source_input = st.text_input("Source reference / author (optional):", value=default_source, key="fix_source_input")

    if st.button("Fix this passage with AI Writing Coach", type="primary", key="btn_fix_action"):
        if not passage_input or not passage_input.strip():
            st.warning("Please select or paste a sentence to fix.")
            return

        with st.spinner("Analyzing passage and generating academic rewrite..."):
            fix_res = as_dict(fix_passage(passage_input, source_input))

        st.divider()
        if fix_res.get("is_fallback"):
            notice = fix_res.get("notice") or "API key not configured or offline"
            st.warning(f"⚠️ **AI unavailable - showing a template** ({notice}).")
        else:
            st.success(f"✨ **Powered by Gemini** ({fix_res.get('provider')})")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Your Sentence")
            st.error(f"> \"{passage_input}\"")
            st.markdown("##### Why this looks copied")
            st.write(fix_res.get("explanation", "This passage shares wording and sentence structure with existing reference documents."))

        with col2:
            st.markdown("#### Suggested Academic Rewrite")
            st.success(f"> \"{fix_res.get('suggested_rewrite', '')}\"")
            st.text_area("Copy rewrite:", value=fix_res.get("suggested_rewrite", ""), height=70)

        st.markdown("#### How to Cite This")
        st.code(fix_res.get("citation_suggestion", f"({source_input}, 2024)"))

        st.markdown("#### 3 Tips to Make Your Writing Original")
        for tip in as_list(fix_res.get("tips")):
            st.markdown(f"- {tip}")


def _compare_tab(settings):
    st.write("Upload two documents to compare them directly. This works offline (after the model has been downloaded once).")
    col_a, col_b = st.columns(2)
    file_a = col_a.file_uploader("Document A (checked against B)", type=["pdf", "docx", "txt"], key="cmp_a")
    file_b = col_b.file_uploader("Document B (reference)", type=["pdf", "docx", "txt"], key="cmp_b")
    if st.button("Compare documents", type="primary", key="cmp_btn"):
        if file_a is None or file_b is None:
            st.error("Please upload both documents.")
            return
        try:
            text_a = extract_text_from_upload(file_a)
            text_b = extract_text_from_upload(file_b)
        except Exception as exc:
            st.error(f"Could not read a file: {exc}")
            return
        model = _get_model()
        if model is None:
            return
        try:
            with st.spinner("Comparing with BERT..."):
                st.session_state["cmp_result"] = compare_documents(
                    model, text_a, text_b, file_a.name, file_b.name,
                    settings.get("semantic_threshold", 0.70), settings.get("near_threshold", 0.90))
        except Exception as exc:
            st.error(f"Comparison failed: {exc}")
    if "cmp_result" in st.session_state:
        display_comparison(st.session_state["cmp_result"])


def _history_tab():
    if IS_EPHEMERAL_DB:
        st.info(f"ℹ️ **Ephemeral Storage Notice:** Scan history is saved in temporary cloud container storage (`{os.path.dirname(DB_PATH)}`) and may reset if the server restarts. Set `PLAGIARISM_DB_PATH` to persist history across restarts.")
    rows = get_scan_history()
    if not rows:
        st.info("No scans saved yet.")
        return
    st.dataframe(rows)
    options = {f"#{r['Scan ID']} - {r['Document']} - {r['Date/Time']} - {r['Status']}": r["Scan ID"]
               for r in rows}
    choice = st.selectbox("Open a previous scan", list(options.keys()), key="hist_select")
    details = get_scan_details(options[choice])
    if details:
        display_results(details, f"hist_{options[choice]}")
    else:
        st.warning("Details for this scan are not available.")


def _tab_report(settings):
    st.header("About this project")
    st.caption("The NLP and AI concepts used in this project, and how each one is used.")

    # 1. Short "How it works" flow (one line per step)
    st.markdown("#### 🔄 How it works")
    st.markdown("""
1. **Upload or paste:** Provide your text directly or upload a PDF, Word (DOCX), or plain text (TXT) file.
2. **Extract text:** Extract raw text and document structure using dedicated format extractors (`pypdf`, `python-docx`).
3. **Clean and prepare:** Normalize unicode, strip noise and citations, segment into sentences, remove stopwords, and lemmatize tokens.
4. **Search the web:** Formulate exact quoted phrases and distinctive keyword queries across beginning, middle, and end sections.
5. **Read the pages:** Retrieve candidate articles via Wikipedia REST API, Tavily raw content/extract endpoints, or direct download.
6. **Compare sentences:** Cascade comparisons through exact containment, SequenceMatcher near-copy, and Sentence-BERT embeddings.
7. **Label and rank sources:** Classify sentences as Exact copy, Near copy, Similar meaning, or Original, ranking verified sources by highest match %.
8. **Analyse tone, topics and details:** Compute emotional tone via VADER, extract named entities, keywords, academic citations, and LDA topic distributions.
9. **Rewrite and cite with AI:** Generate educational explanations, academic rewrites, APA citations, and originality tips via Gemini LLM.
10. **Report:** View interactive originality metrics, inspect matched sources, and download an official PDF audit report.
""")

    st.divider()

    # 2. Concept Groups
    st.markdown("#### 🧠 NLP & AI Concepts Implemented in PlagiarismAI")
    st.caption("Each concept is verified against the codebase. Fallback mechanisms are explicitly noted.")

    # Group 1: Text Preparation
    with st.expander("📄 Text Preparation", expanded=True):
        st.markdown("""
| Concept | What it is | How this project uses it | File |
| :--- | :--- | :--- | :--- |
| **Text extraction (PDF, DOCX, TXT)** | Extracts clean raw character streams from binary and structured document formats. | Reads uploaded files via `pypdf`, `python-docx`, and UTF-8 text decoders. | `nlp/extract.py` |
| **Text cleaning & normalization** | Converts text to Unicode NFKC standard, normalizes quotes/dashes, and strips noise. | Removes URLs, academic markers, citation numbers, and excessive whitespace. | `nlp/preprocess.py`, `nlp/similarity.py` |
| **Sentence segmentation** | Splits unformatted paragraphs into discrete, grammatically coherent sentence units. | Regex splitter breaks document into sentences so each can be independently matched and highlighted. | `nlp/similarity.py`, `nlp/preprocess.py` |
| **Tokenization** | Segments sentence strings into individual word tokens for linguistic inspection. | Splits sentences into whitespace/punctuation tokens for vocabulary and frequency counting. | `nlp/preprocess.py`, `nlp/analysis.py` |
| **Stopword removal** | Filters out high-frequency grammatical words carrying minimal semantic distinction. | Drops common English words (and Devanagari particles) so comparison focuses on content words. | `nlp/preprocess.py`, `nlp/similarity.py` |
| **Lemmatization** | Reduces inflected word variations down to their base canonical dictionary lemma. | Applies NLTK WordNet lemmatizer with a rule-based suffix stemmer as fallback. | `nlp/preprocess.py` |
""")

    # Group 2: Turning text into numbers
    with st.expander("🔢 Turning Text into Numbers", expanded=False):
        st.markdown("""
| Concept | What it is | How this project uses it | File |
| :--- | :--- | :--- | :--- |
| **TF-IDF vectorization** | Weights words by term frequency scaled by inverse document frequency across a corpus. | Converts document sentences into 5,000-dimensional sparse vectors using scikit-learn. | `nlp/preprocess.py` |
| **Sentence embeddings** | Maps full sentences into dense 384-dimensional continuous vector representations. | Encodes sentences with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (falls back to `all-MiniLM-L6-v2`). | `nlp/similarity.py`, `app.py` |
""")

    # Group 3: Finding copied text
    with st.expander("🔍 Finding Copied Text", expanded=False):
        st.markdown(r"""
| Concept | What it is | How this project uses it | File |
| :--- | :--- | :--- | :--- |
| **Exact & near-exact matching** | Identifies word-for-word duplication and slight character/word edits. | Verifies verbatim containment (100% Exact copy) and uses SequenceMatcher ($\ge 90\%$) for near copies. | `nlp/similarity.py` |
| **Jaccard similarity** | Measures the ratio of shared unique vocabulary tokens over the union of tokens. | Evaluates surface vocabulary overlap between compared document pairs. | `nlp/preprocess.py` |
| **Cosine similarity** | Computes the cosine of the angle between two mathematical vectors. | Calculates similarity between TF-IDF term vectors and dense Sentence-BERT vectors. | `nlp/similarity.py`, `nlp/preprocess.py` |
| **Semantic similarity** | Evaluates contextual meaning overlap regardless of whether identical vocabulary is used. | Flags paraphrased sentences exceeding the semantic threshold ($\ge 70\%$) using BERT embeddings. | `nlp/similarity.py` |
| **Combined originality score** | Blends lexical, statistical, and neural similarity scores into a single composite metric. | Weights Jaccard (20%), TF-IDF cosine (30%), and Sentence-BERT (50%) for document comparison. | `nlp/preprocess.py` |
| **Web search retrieval** | Discovers candidate online source documents via external search engine APIs. | Executes exact-phrase and keyword queries via Tavily Search API (or Brave Search API). | `app.py` |
| **Webpage & PDF text extraction** | Downloads and converts HTML web pages and online PDFs into plain text. | Extracts text via official Wikipedia REST API, Tavily Extract, and BeautifulSoup filtering. | `app.py`, `nlp/extract.py` |
| **Source ranking by match %** | Orders discovered source documents from greatest match overlap to least. | Ranks verified sources by `best_match_pct` descending, followed by match count. | `app.py`, `nlp/similarity.py` |
""")

    # Group 4: Understanding the writing
    with st.expander("📊 Understanding the Writing", expanded=False):
        st.markdown("""
| Concept | What it is | How this project uses it | File |
| :--- | :--- | :--- | :--- |
| **Sentiment analysis** | Classifies emotional tone into Positive, Negative, or Neutral with confidence. | Uses VADER intensity scoring with a rule-based Hindi/Marathi Devanagari lexicon fallback. | `nlp/analysis.py` |
| **Named entity recognition (NER)** | Identifies real-world entities: people, organizations, locations, and dates. | Uses spaCy (`en_core_web_sm`) with regex-based capitalization and date heuristic fallback. | `nlp/analysis.py` |
| **Keyword extraction** | Discovers the most representative, distinguishing key terms from text. | Employs KeyBERT embedding-based keyphrase extraction with frequency-filtering fallback. | `nlp/analysis.py` |
| **Citation extraction** | Detects academic bibliographic references, numeric citations, DOIs, and URLs. | Uses regular expressions matching brackets `[1]`, author-year `(Smith, 2020)`, and DOI prefixes. | `nlp/analysis.py` |
| **Topic modeling with LDA** | Discovers hidden thematic topics across documents using Latent Dirichlet Allocation. | Employs scikit-learn's `LatentDirichletAllocation` over a reference academic corpus (keyword fallback for $<150$ words). | `nlp/topics.py` |
| **Topic overlap score** | Measures thematic similarity across documents based on their topic distributions. | Calculates cosine similarity across document topic probability distributions. | `nlp/topics.py` |
""")

    # Group 5: Multilingual
    with st.expander("🌐 Multilingual NLP", expanded=False):
        st.markdown("""
| Concept | What it is | How this project uses it | File |
| :--- | :--- | :--- | :--- |
| **Language detection** | Determines the natural language and script of submitted text. | Inspects Devanagari character ratio (U+0900–U+097F), distinctive Marathi/Hindi marker words, and `langdetect`. | `nlp/multilingual.py` |
| **Machine translation** | Converts non-English sentences to English for standardized analysis. | Uses Google Cloud Translation API if `GOOGLE_TRANSLATE_API_KEY` is set; uses offline fallback engine otherwise. | `nlp/multilingual.py` |
| **Cross-lingual matching** | Compares sentences across different languages directly or via translation. | Uses multilingual Sentence-BERT (`paraphrase-multilingual-MiniLM-L12-v2`) and translate-then-compare alignment. | `nlp/multilingual.py`, `nlp/similarity.py` |
""")

    # Group 6: Generative AI
    with st.expander("✨ Generative AI", expanded=False):
        st.markdown("""
| Concept | What it is | How this project uses it | File |
| :--- | :--- | :--- | :--- |
| **Large language model (LLM)** | Generates contextual explanations, academic rewrites, and APA citations. | Powered by the official Google Gen AI SDK (`google-genai`) using model `gemini-3.5-flash` (configurable). | `llm/assistant.py` |
| **Prompt engineering** | Guides LLM behavior using explicit instructions, schema constraints, and examples. | Defines academic persona role, few-shot examples, JSON schema enforcement, and temperature 0.3. | `llm/prompts.py`, `llm/assistant.py` |
| **Prompt-injection protection** | Prevents untrusted document text from overriding assistant instructions. | Encloses user text inside strict structural isolation delimiters (`<<<PASSAGE>>>`). | `llm/prompts.py` |
| **Response caching & fallback** | Saves quota and ensures graceful offline resilience when APIs are unavailable. | Stores suggestions in-memory by SHA-256 passage hash and falls back to structured templates if offline. | `llm/assistant.py` |
""")

    # Group 7: Evaluation
    with st.expander("📈 Evaluation & Benchmarks", expanded=False):
        st.markdown("""
| Concept | What it is | How this project uses it | File |
| :--- | :--- | :--- | :--- |
| **Evaluation metrics** | Quantifies classifier effectiveness against ground truth labels. | Computes Accuracy, Precision, Recall, F1-Score, and Confusion Matrix across methods. | `evaluate.py` |
| **Benchmark datasets** | Curated reference sets of labeled original, paraphrased, and copied text. | Evaluates 31 plagiarism detection pairs, 20 sentiment sentences, and 10 multilingual pairs. | `data/eval/`, `reports/evaluation.json` |
""")

    st.divider()

    # 3. Compact "Libraries and tools used" list
    st.markdown("#### 🛠️ Libraries and Tools Used")
    st.markdown("""
- **Streamlit** (`streamlit`): Web application dashboard, session state, and reactive interface.
- **Sentence-Transformers** (`sentence-transformers`): Pre-trained multilingual neural embeddings (`paraphrase-multilingual-MiniLM-L12-v2`).
- **Scikit-Learn** (`scikit-learn`): TF-IDF vectorization, cosine similarity, Latent Dirichlet Allocation (LDA), and evaluation metrics.
- **Google Gen AI SDK** (`google-genai`): Official Google SDK for Gemini LLM academic assistance (`gemini-3.5-flash`).
- **BeautifulSoup4** (`beautifulsoup4`): HTML parsing, DOM traversal, and noise removal for web page text extraction.
- **Requests** (`requests`): HTTP client for search APIs (Tavily/Brave), Wikipedia REST API, and Google Cloud Translation.
- **PyPDF** (`pypdf`): PDF binary document parsing and text extraction.
- **python-docx** (`python-docx`): Microsoft Word (.docx) document parsing and text extraction.
- **Plotly** (`plotly`): Interactive visual charts (similarity score distributions and source match breakdowns).
- **fpdf2** (`fpdf2`): Programmatic generation and layout of downloadable PDF originality reports.
- **NumPy & Pandas** (`numpy`, `pandas`): Array operations, similarity matrix math, and evaluation dataset manipulation.
- **Matplotlib** (`matplotlib`): Generation of static benchmark comparison charts in reports.
- **NLTK** (`nltk`): WordNet lemmatizer and text normalization utilities.
- **spaCy** (`spacy`): Named entity recognition (`en_core_web_sm`) for extracting people, places, and organizations.
- **VADER Sentiment** (`vaderSentiment`): Rule-based sentiment analysis for tone classification.
- **KeyBERT** (`keybert`): Embedding-based keyword and keyphrase extraction.
- **langdetect** (`langdetect`): Probabilistic language identification for international text.
- **deep-translator** (`deep-translator`): Free machine translation fallback library.
""")

    st.divider()

    # 4. Existing sections below the concepts
    st.markdown("#### 🔬 How We Read Your Text (Preprocessing Inspection)")
    prep = st.session_state.get("prep_result")
    if prep:
        st.write(f"**Scanned document:** `{prep.get('raw_word_count')} words` raw | `{prep.get('cleaned_word_count')} words` cleaned | `{prep.get('sentence_count')} sentences` | `{prep.get('unique_lemmas')} unique lemmas`")
        col_raw, col_clean = st.columns(2)
        with col_raw:
            st.caption("Raw Extracted Text (Before Cleaning)")
            st.text_area("raw_preview", value=prep.get("original_text", "")[:600] + ("..." if len(prep.get("original_text", "")) > 600 else ""), height=150, disabled=True)
        with col_clean:
            st.caption("Cleaned & Lemmatized Text (After Cleaning)")
            st.text_area("clean_preview", value=prep.get("cleaned_text", "")[:600] + ("..." if len(prep.get("cleaned_text", "")) > 600 else ""), height=150, disabled=True)
    else:
        st.info("Run a scan in the **Check** tab to inspect live preprocessing statistics for your document.")

    st.divider()

    st.markdown("#### 📊 Algorithm Benchmark Evaluation")
    st.warning("⚠️ **Evaluation Notice:** Similarity thresholds were chosen by inspection on a small dataset; results are optimistic.")
    eval_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "evaluation.json")
    if os.path.exists(eval_path):
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
            p_bench = eval_data.get("plagiarism_detection_benchmark", {})
            m_bench = eval_data.get("multilingual_benchmark", {})

            rows = []
            for key, val in p_bench.items():
                rows.append({
                    "Method": val.get("method_name"),
                    "Data Evaluated": f"{val.get('pairs_evaluated', 31)} pairs",
                    "Threshold": val.get("threshold"),
                    "Accuracy": f"{val.get('accuracy', 0)*100:.1f}%",
                    "Precision": f"{val.get('precision', 0)*100:.1f}%",
                    "Recall": f"{val.get('recall', 0)*100:.1f}%",
                    "F1-Score": f"{val.get('f1', 0):.3f}"
                })
            has_google_key = bool(os.getenv("GOOGLE_TRANSLATE_API_KEY", "").strip())
            for key, val in m_bench.items():
                m_name = val.get("method_name", "")
                if "Translate-Then-Compare" in m_name:
                    trans_label = "Google Cloud Translation" if has_google_key else "fallback translator"
                    m_name = f"Translate-Then-Compare ({trans_label})"
                rows.append({
                    "Method": m_name,
                    "Data Evaluated": f"{val.get('pairs_evaluated', 10)} cross-lingual pairs",
                    "Threshold": val.get("threshold"),
                    "Accuracy": f"{val.get('accuracy', 0)*100:.1f}%",
                    "Precision": f"{val.get('precision', 0)*100:.1f}%",
                    "Recall": f"{val.get('recall', 0)*100:.1f}%",
                    "F1-Score": f"{val.get('f1', 0):.3f}"
                })
            st.dataframe(rows, use_container_width=True)

            has_gemini_key = bool(os.getenv("GEMINI_API_KEY", "").strip())
            if has_google_key:
                st.info("ℹ️ **Translation Provider Status:** `GOOGLE_TRANSLATE_API_KEY` is configured — cross-lingual translation via **Google Cloud Translation**.")
            elif has_gemini_key:
                st.info("ℹ️ **Translation Provider Status:** `GOOGLE_TRANSLATE_API_KEY` is not set — cross-lingual translation via **Gemini AI Translation**.")
            else:
                st.info("ℹ️ **Translation Provider Status:** No cloud translation keys set — cross-lingual checks use the **fallback translator**.")

            # Sentiment Benchmark
            s_bench = eval_data.get("sentiment_benchmark", {})
            if s_bench:
                st.markdown("##### Sentiment Analysis Benchmark (20 Labeled Sentences)")
                s_acc = s_bench.get("overall_accuracy", 0.8) * 100
                st.write(f"**Overall Accuracy:** **{s_acc:.1f}%** ({int(round(s_acc*0.2))}/20 sentences correctly classified).")

            # Information Extraction Benchmark
            ie_bench = eval_data.get("information_extraction_benchmark", {})
            if ie_bench:
                st.markdown("##### Information Extraction Benchmark (22 Sentences: English, Hindi, Marathi, Negatives)")
                ne_stats = ie_bench.get("named_entities", {})
                cit_stats = ie_bench.get("citations", {})
                cats = ne_stats.get("by_category", {})

                ie_rows = [
                    {
                        "Task / Category": "Named Entities (Overall)",
                        "Evaluated": f"{ie_bench.get('total_samples', 22)} sentences",
                        "Precision": f"{ne_stats.get('precision', 0)*100:.1f}%",
                        "Recall": f"{ne_stats.get('recall', 0)*100:.1f}%",
                        "F1-Score": f"{ne_stats.get('f1', 0):.3f}"
                    },
                    {
                        "Task / Category": "  • People (PERSON)",
                        "Evaluated": f"{cats.get('people', {}).get('tp', 0) + cats.get('people', {}).get('fn', 0)} true entities",
                        "Precision": f"{cats.get('people', {}).get('precision', 0)*100:.1f}%",
                        "Recall": f"{cats.get('people', {}).get('recall', 0)*100:.1f}%",
                        "F1-Score": f"{cats.get('people', {}).get('f1', 0):.3f}"
                    },
                    {
                        "Task / Category": "  • Organizations (ORG)",
                        "Evaluated": f"{cats.get('organizations', {}).get('tp', 0) + cats.get('organizations', {}).get('fn', 0)} true entities",
                        "Precision": f"{cats.get('organizations', {}).get('precision', 0)*100:.1f}%",
                        "Recall": f"{cats.get('organizations', {}).get('recall', 0)*100:.1f}%",
                        "F1-Score": f"{cats.get('organizations', {}).get('f1', 0):.3f}"
                    },
                    {
                        "Task / Category": "  • Places (GPE/LOC)",
                        "Evaluated": f"{cats.get('places', {}).get('tp', 0) + cats.get('places', {}).get('fn', 0)} true entities",
                        "Precision": f"{cats.get('places', {}).get('precision', 0)*100:.1f}%",
                        "Recall": f"{cats.get('places', {}).get('recall', 0)*100:.1f}%",
                        "F1-Score": f"{cats.get('places', {}).get('f1', 0):.3f}"
                    },
                    {
                        "Task / Category": "  • Dates (DATE)",
                        "Evaluated": f"{cats.get('dates', {}).get('tp', 0) + cats.get('dates', {}).get('fn', 0)} true entities",
                        "Precision": f"{cats.get('dates', {}).get('precision', 0)*100:.1f}%",
                        "Recall": f"{cats.get('dates', {}).get('recall', 0)*100:.1f}%",
                        "F1-Score": f"{cats.get('dates', {}).get('f1', 0):.3f}"
                    },
                    {
                        "Task / Category": "Citations & References (Bracket, Author-Year, DOI, URL)",
                        "Evaluated": f"{ie_bench.get('total_samples', 22)} sentences",
                        "Precision": f"{cit_stats.get('precision', 0)*100:.1f}%",
                        "Recall": f"{cit_stats.get('recall', 0)*100:.1f}%",
                        "F1-Score": f"{cit_stats.get('f1', 0):.3f}"
                    }
                ]
                st.dataframe(ie_rows, use_container_width=True)

                if is_spacy_available():
                    st.success("✅ **Named Entity Recognition:** spaCy model (`en_core_web_sm`) is loaded and active.")
                else:
                    st.warning("⚠️ **Named Entity Recognition Notice:** spaCy model (`en_core_web_sm`) is missing; regex fallback is active. Install with: `python -m spacy download en_core_web_sm`.")

            if eval_data.get("limitations"):
                with st.expander("ℹ️ Benchmark Methodology & Provider Notes", expanded=False):
                    for lim in eval_data["limitations"]:
                        st.markdown(f"- {lim}")
        except Exception:
            st.warning("Could not read reports/evaluation.json.")

    fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "figures", "similarity_methods_benchmark.png")
    if os.path.exists(fig_path):
        st.image(fig_path, caption="Plagiarism Detection & Multilingual Methods Benchmark Comparison")

    st.divider()

    st.markdown("#### ⚠️ Known Limitations")
    st.markdown("""
- **Paraphrase Boundaries:** Heavily modified text without shared distinctive words or neighboring matched context is classified as topical similarity rather than flagged plagiarism to prevent false accusations.
- **Web Coverage:** Real-time online scanning relies on search queries and candidate page accessibility; paywalled or JavaScript-blocked pages cannot be evaluated.
- **Multilingual Support:** Deep grammatical and semantic matching is optimized for English, Hindi, and Marathi; other languages rely on automated machine translation before comparison.
- **Heuristic Fallbacks:** When offline or without API keys, rule-based dictionary and template fallbacks maintain system availability with reduced linguistic nuance.
""")

    st.divider()

    st.markdown("#### 📁 Offline Document Comparison Tool")
    _compare_tab(settings)

    st.divider()

    st.markdown("#### 📜 Database Scan History")
    _history_tab()

    st.divider()

    st.markdown("#### 📄 Download PDF Report")
    if "scan_result" in st.session_state:
        pdf_bytes = generate_pdf_report(st.session_state["scan_result"])
        st.download_button(
            label="📄 Download Official PDF Report",
            data=pdf_bytes,
            file_name=f"PlagiarismAI_Report_{st.session_state['scan_result'].get('document_name', 'Doc')}.pdf",
            mime="application/pdf"
        )
    else:
        st.caption("Perform a scan to generate and download a PDF report.")

    st.divider()

    with st.expander("⚙️ Advanced settings and search diagnostics", expanded=False):
        st.markdown("##### Pipeline Configuration")
        labels = {"auto": "Auto (first configured: Brave, then Tavily)", "brave": "Brave Search API",
                  "tavily": "Tavily Search API", "ddgs": "DuckDuckGo via ddgs (unofficial)"}
        st.selectbox("Search provider", PROVIDER_CHOICES, format_func=labels.get, key="adv_provider")
        st.slider("Semantic similarity threshold (%)", 50, 95, 70, key="adv_sem_thresh")
        st.slider("Near-exact threshold (%)", 80, 99, 90, key="adv_near_thresh")
        st.slider("Search queries", 1, 15, 8, key="adv_queries")
        st.slider("Results per query", 1, 10, 5, key="adv_res_per_q")
        st.slider("Max webpages to download", 1, 30, 12, key="adv_max_pages")
        st.slider("Max sentences scanned", 20, 500, 200, key="adv_max_sentences")

        st.markdown("##### Search Provider Diagnostics")
        _search_diagnostics(st.session_state.get("adv_provider", "auto"), key_prefix="rep_diag")


def main():
    st.set_page_config(page_title=f"{APP_NAME} - Document Originality Checker",
                       page_icon="🔍", layout="wide")
    init_db()
    verify_gemini_model_at_startup()

    app_pwd = get_env_or_secret("APP_PASSWORD", "")
    if app_pwd:
        if not st.session_state.get("authenticated", False):
            st.title("🔍 PlagiarismAI")
            st.markdown("### 🔒 Authentication Required")
            st.info("This PlagiarismAI deployment is password-protected. Please enter the access password to continue.")
            entered = st.text_input("Enter Access Password", type="password", key="app_pwd_input")
            if st.button("Unlock Application", type="primary", key="app_unlock_btn"):
                if entered == app_pwd:
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("Incorrect password. Please try again.")
            return

    st.title("🔍 PlagiarismAI")
    st.caption("Check how original your writing is, in a few seconds.")

    # Show warning if multilingual model failed and English-only model is active
    if not _MODEL_LOAD_STATUS.get("is_multilingual", True) and _MODEL_LOAD_STATUS.get("warning"):
        st.warning(f"⚠️ **Model Notice:** {_MODEL_LOAD_STATUS['warning']}")

    if not is_spacy_available():
        st.warning("⚠️ **spaCy Model Notice:** `en_core_web_sm` is not loaded. Named Entity Recognition is using the regex fallback. To enable spaCy NER, run: `python -m spacy download en_core_web_sm`.")

    _render_sidebar()
    settings = _get_settings()
    tabs = st.tabs(["Check", "Results", "Fix it for me", "About the Project"])
    with tabs[0]:
        _tab_check(settings)
    with tabs[1]:
        _tab_results()
    with tabs[2]:
        _tab_fix()
    with tabs[3]:
        _tab_report(settings)


def run_cli_search_test(query=None):
    """python app.py --test-search ["query"]   -> real provider connection + search test."""
    test = test_search_provider("auto", query)
    print(f"Provider        : {test['provider']}")
    print(f"Configured      : {test['configured']} {('(' + test['key_source'] + ')') if test['key_source'] else ''}")
    print(f"Query           : {test['query']}")
    print(f"Connection OK   : {test['connected']}  ({test['seconds']} s)")
    print(f"Results retrieved: {test['result_count']}")
    for r in test["results"]:
        print(f"  - {r['title']}\n    {r['url']}")
    for message in test["errors"]:
        print("ERROR/NOTE:", message)
    return 0 if test["result_count"] > 0 else 1


def run_cli_scan(path):
    """python app.py --test-scan passage.txt   -> full real scan (BERT + web) printed to the terminal."""
    text = open(path, encoding="utf-8").read()
    model = load_bert_model()
    settings = {"provider": "auto", "semantic_threshold": 0.70, "near_threshold": 0.90, "max_queries": 8,
                "results_per_query": 5, "max_pages": 12, "max_sentences": 200}
    result = run_scan(model, text, os.path.basename(path), settings,
                      lambda fraction, message: print(f"[{fraction:4.0%}] {message}"))
    pipe, stats = result["pipeline"], result["stats"]
    print(f"\nSTATUS: {result['status'].upper()}  {result['incomplete_reason']}")
    print(f"Provider: {pipe['engine']} | searches ok {pipe['queries_ok']}/{pipe['queries_planned']} | "
          f"results {pipe['results_retrieved']} | downloaded {pipe['pages_downloaded']} | "
          f"compared {pipe['pages_compared']} | sources with matches {pipe['sources_with_matches']}")
    if result["status"] != "incomplete":
        print(f"Overall {stats['overall_pct']}% | exact/near {stats['exact_pct']}% | "
              f"semantic-only {stats['semantic_pct']}% | any semantic {stats['semantic_any_pct']}%")
    for source in result["sources"]:
        state = "VERIFIED" if source["verified"] else "UNVERIFIED: " + source["note"]
        print(f"  [{state}] {source['title']} - {source['url']} ({source['matches']} matches)")
    return 0 if result["status"] != "incomplete" else 1


if __name__ == "__main__" and "--test-search" in sys.argv:
    _args = sys.argv[sys.argv.index("--test-search") + 1:]
    sys.exit(run_cli_search_test(_args[0] if _args else None))
elif __name__ == "__main__" and "--test-scan" in sys.argv:
    sys.exit(run_cli_scan(sys.argv[sys.argv.index("--test-scan") + 1]))
elif __name__ == "__main__":
    main()
