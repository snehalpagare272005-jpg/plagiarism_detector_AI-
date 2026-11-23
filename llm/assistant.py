"""
LLM Assistant Module for PlagiarismAI.
Integrates Gemini API using the official Google Gen AI SDK (`google-genai`):
- Configurable model via GEMINI_MODEL env var (defaults to gemini-2.5-flash).
- Temperature setting (0.3) for academic precision.
- Safe JSON output parsing with markdown code-fence stripping and retry.
- In-memory session caching by passage hash to save API quota.
- Graceful template fallbacks when API key is missing, network is offline, or rate limits occur.
"""

import hashlib
import json
import os
import re
import sys
from typing import Any, Dict, Optional

from llm.prompts import SYSTEM_ROLE_INSTRUCTION, build_fix_prompt


def get_env_or_secret(name: str, default: str = "") -> str:
    """Return environment variable or Streamlit secret without requiring .env."""
    # Check .env first to dynamically pick up newly saved keys
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_file = os.path.join(root_dir, ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    if k.strip() == name:
                        clean_v = v.strip().strip('"').strip("'")
                        if clean_v:
                            os.environ[name] = clean_v
                            return clean_v
        except OSError:
            pass

    val = os.getenv(name, "").strip()
    if val:
        return val
    try:
        import streamlit as st
        if hasattr(st, "secrets") and name in st.secrets:
            s_val = str(st.secrets[name]).strip()
            if s_val:
                return s_val
    except Exception:
        pass
    return default


DEFAULT_MODEL_NAME = get_env_or_secret("GEMINI_MODEL", "gemini-3.5-flash")

# In-memory session cache for instant repeat clicks
_FIX_CACHE: Dict[str, Dict[str, Any]] = {}


def verify_gemini_model_at_startup() -> Dict[str, Any]:
    """
    Startup diagnostic check: validates GEMINI_MODEL setting and checks
    if the Google Gen AI SDK and API key are available.
    Prints a clear message if the model name is rejected or missing.
    """
    active_model = get_env_or_secret("GEMINI_MODEL", DEFAULT_MODEL_NAME).strip()
    api_key = get_env_or_secret("GEMINI_API_KEY", "").strip()

    status = {
        "model": active_model,
        "sdk": "google-genai",
        "key_configured": bool(api_key),
        "valid": True,
        "message": ""
    }

    # Known current valid Gemini models on Google Gen AI SDK
    known_models = {
        "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.7-flash",
        "gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite",
        "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-pro-exp"
    }

    if active_model not in known_models and not active_model.startswith("gemini-"):
        print(f"[!] WARNING: Configured GEMINI_MODEL='{active_model}' may not be recognized by the Google Gen AI SDK.", file=sys.stderr)
        print(f"    Supported models include: {', '.join(sorted(known_models))}", file=sys.stderr)
        status["valid"] = False
        status["message"] = f"Unrecognized model '{active_model}'. Use a known model like gemini-2.5-flash."

    if not api_key:
        print(f"[*] Note: GEMINI_API_KEY not configured. GenAI assistant will use offline academic template fallback.")
    else:
        print(f"[*] GenAI Assistant configured: Model='{active_model}' using official google-genai SDK.")

    return status


def _get_cache_key(passage: str, source_title: str) -> str:
    key_str = f"{passage.strip()}_{source_title.strip()}"
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()


def _clean_json_response(raw_text: str) -> str:
    """Strips markdown code fences (```json ... ```) from model response."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _get_fallback_fix(passage: str, source_title: str = "", source_url: str = "") -> Dict[str, Any]:
    """
    Template-based fallback used when API key is missing or offline, ensuring demo never crashes.
    """
    source_ref = source_title or source_url or "the original author"
    words = passage.split()

    if len(words) > 4:
        rewrite_candidate = f"According to findings in {source_ref}, research indicates that {passage.lower().rstrip('.')}."
    else:
        rewrite_candidate = f"As noted by {source_ref}: {passage}."

    return {
        "explanation": f"This statement matches phrasing found in {source_ref}. Paraphrasing into your own words will demonstrate deeper comprehension.",
        "rewrite": rewrite_candidate,
        "suggested_rewrite": rewrite_candidate,
        "citation": f"Adapted from {source_ref}. Retrieved from: {source_url or 'Reference document'}.",
        "citation_suggestion": f"Adapted from {source_ref}. Retrieved from: {source_url or 'Reference document'}.",
        "tips": [
            "Express the underlying idea rather than modifying individual vocabulary words.",
            "Synthesize points from multiple scholarly papers to establish originality.",
            "Always include formal attribution even when paraphrasing substantially."
        ],
        "provider": "Template Fallback Engine (Offline)",
        "is_fallback": True
    }


def call_gemini_api(prompt: str, model_name: str, api_key: str, temperature: float = 0.3) -> str:
    """
    Invokes Gemini API via the official Google Gen AI SDK (`google-genai`).
    """
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=SYSTEM_ROLE_INSTRUCTION
            )
        )
        return response.text
    except Exception as e:
        raise RuntimeError(f"google-genai API error ({type(e).__name__}): {e}")


def fix_passage(
    passage: str,
    source_title: str = "",
    source_url: str = "",
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main function returning educational explanation, rewrite, citation, and writing tips.
    Checks cache first, then Gemini API, with safe fallback on failure.
    """
    if not passage or not passage.strip():
        return {
            "explanation": "No text provided to analyze.",
            "rewrite": "",
            "suggested_rewrite": "",
            "citation": "",
            "citation_suggestion": "",
            "tips": [],
            "provider": "None",
            "is_fallback": True
        }

    cache_key = _get_cache_key(passage, source_title)
    if cache_key in _FIX_CACHE:
        res = dict(_FIX_CACHE[cache_key])
        res["cached"] = True
        return res

    api_key = get_env_or_secret("GEMINI_API_KEY", "").strip()
    active_model = model_name or get_env_or_secret("GEMINI_MODEL", DEFAULT_MODEL_NAME).strip()

    if not api_key:
        fallback = _get_fallback_fix(passage, source_title, source_url)
        _FIX_CACHE[cache_key] = fallback
        return fallback

    prompt = build_fix_prompt(passage, source_title, source_url)

    # Candidate models: active model first, followed by stable alternatives if 503/404 occurs
    models_to_try = [active_model]
    for alt in ["gemini-3.5-flash-lite", "gemini-3.8-flash"]:
        if alt not in models_to_try:
            models_to_try.append(alt)

    for current_model in models_to_try:
        raw_response = ""
        for attempt in range(2):
            try:
                raw_response = call_gemini_api(prompt, current_model, api_key, temperature=0.3)
                cleaned = _clean_json_response(raw_response)
                parsed = json.loads(cleaned)

                explanation = parsed.get("explanation", "")
                rewrite = parsed.get("rewrite", "")
                citation = parsed.get("citation", "")
                tips = parsed.get("tips", [])

                result = {
                    "explanation": explanation,
                    "rewrite": rewrite,
                    "suggested_rewrite": rewrite,
                    "citation": citation,
                    "citation_suggestion": citation,
                    "tips": tips if isinstance(tips, list) else [str(tips)],
                    "provider": f"Gemini API ({current_model})",
                    "is_fallback": False,
                    "cached": False
                }
                _FIX_CACHE[cache_key] = result
                return result

            except json.JSONDecodeError:
                if attempt == 0:
                    prompt += "\nREMINDER: Return ONLY a valid JSON object. Do not wrap in markdown or include extraneous commentary."
                    continue
            except Exception as e:
                err_msg = str(e)
                # If high demand (503) or model unavailable/not found (404), fail over to next model
                if ("503" in err_msg or "unavailable" in err_msg.lower() or "404" in err_msg or "not found" in err_msg.lower()) and current_model != models_to_try[-1]:
                    break

                if "429" in err_msg or "quota" in err_msg.lower():
                    user_msg = "Gemini API rate limit reached. Returning educational writing template."
                elif "not found" in err_msg.lower() or "404" in err_msg:
                    user_msg = f"Model '{current_model}' was rejected or not found. Check GEMINI_MODEL setting."
                    print(f"[!] ERROR: Gemini API rejected model name '{current_model}': {err_msg}", file=sys.stderr)
                elif "network" in err_msg.lower() or "connection" in err_msg.lower() or "11001" in err_msg:
                    user_msg = "Network connection unavailable. Using offline guidance template."
                else:
                    user_msg = f"API notice ({type(e).__name__}). Using standard template."

                fallback = _get_fallback_fix(passage, source_title, source_url)
                fallback["notice"] = user_msg
                _FIX_CACHE[cache_key] = fallback
                return fallback

    fallback = _get_fallback_fix(passage, source_title, source_url)
    _FIX_CACHE[cache_key] = fallback
    return fallback
