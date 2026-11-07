import os
import re
import time
from collections import Counter
from typing import Dict, List, Optional, Set

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from google.genai.errors import APIError
import pandas as pd
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="De-Jargonify Core Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

# Hierarchical model fallback list
MODELS = [
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]


# --- Baseline Dataset Ingestion & Indexing Engine ---

class DatasetJargonBaseline:
    def __init__(self, dataset_path: str = "dataset_full.csv"):
        self.dataset_path = dataset_path
        self.jargon_phrases: Set[str] = set()
        self.domain_lexicons: Dict[str, Set[str]] = {}
        self.baseline_stats: Dict[str, float] = {
            "mean_readability": 14.5,
            "median_readability": 16.1,
            "sample_count": 0,
        }
        self.few_shot_examples: List[Dict[str, str]] = []
        self._load_baseline()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z-]{3,}\b", text.lower())

    def _extract_ngrams(self, tokens: List[str], n: int) -> List[str]:
        return [
            " ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
        ]

    def _load_baseline(self):
        if not os.path.exists(self.dataset_path):
            print(f"Warning: Baseline dataset '{self.dataset_path}' not found.")
            return

        try:
            df = pd.read_csv(self.dataset_path)
            self.baseline_stats["sample_count"] = len(df)
            if "readability_score" in df.columns:
                self.baseline_stats["mean_readability"] = round(
                    float(df["readability_score"].mean()), 2
                )
                self.baseline_stats["median_readability"] = round(
                    float(df["readability_score"].median()), 2
                )

            high_quality_df = (
                df[df["quality_flag"] == "High"]
                if "quality_flag" in df.columns
                else df
            )
            for _, row in high_quality_df.head(4).iterrows():
                src = str(row.get("source", "")).strip()
                tgt = str(row.get("target", "")).lstrip(": ").strip()
                if src and tgt:
                    self.few_shot_examples.append({"source": src, "target": tgt})

            phrase_counter = Counter()
            stop_phrases = {
                "because of",
                "capable of",
                "improved with",
                "despite the",
                "the fact",
                "considering the",
            }

            for _, row in df.iterrows():
                text = str(row.get("source", ""))
                domain = str(row.get("domain_context", "General"))
                tokens = self._tokenize(text)

                if domain not in self.domain_lexicons:
                    self.domain_lexicons[domain] = set()

                for n in (2, 3):
                    ngrams = self._extract_ngrams(tokens, n)
                    phrase_counter.update(ngrams)
                    for ng in ngrams:
                        if ng not in stop_phrases:
                            self.domain_lexicons[domain].add(ng)

            self.jargon_phrases = {
                phrase
                for phrase, count in phrase_counter.items()
                if count >= 5 and phrase not in stop_phrases
            }

        except Exception as e:
            print(f"Error loading {self.dataset_path}: {e}")

    def find_baseline_matches(self, text: str) -> List[str]:
        text_lower = text.lower()
        matches = [
            phrase
            for phrase in self.jargon_phrases
            if re.search(r"\b" + re.escape(phrase) + r"\b", text_lower)
        ]
        return sorted(matches, key=len, reverse=True)


baseline_engine = DatasetJargonBaseline("dataset_full.csv")


# --- Linguistic & Readability Scoring Engine ---

def count_syllables(word: str) -> int:
    word = word.lower().strip()
    if len(word) <= 3:
        return 1
    word = re.sub(r"(?:[^laeiouy]|ed|es|e)$", "", word)
    word = re.sub(r"^y", "", word)
    matches = re.findall(r"[aeiouy]{1,2}", word)
    return max(1, len(matches))


def calculate_metrics(text: str) -> dict:
    words = re.findall(r"\b[A-Za-z0-9'-]+\b", text)
    sentences = re.findall(r"[^.!?]+[.!?]?", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    total_words = len(words)
    total_sentences = max(1, len(sentences))

    if total_words == 0:
        return {"flesch_reading_ease": 0.0, "gunning_fog": 0.0}

    total_syllables = sum(count_syllables(w) for w in words)
    complex_words = sum(1 for w in words if count_syllables(w) >= 3)

    flesch = (
        206.835
        - 1.015 * (total_words / total_sentences)
        - 84.6 * (total_syllables / total_words)
    )
    flesch_score = round(max(0.0, min(100.0, flesch)), 1)

    fog = 0.4 * (
        (total_words / total_sentences) + 100 * (complex_words / total_words)
    )
    fog_score = round(fog, 1)

    return {
        "flesch_reading_ease": flesch_score,
        "gunning_fog": fog_score,
    }


# --- Pydantic Schema for Structured LLM Output ---

class AnalysisResponse(BaseModel):
    plain_english_translation: str = Field(
        description="Direct, simplified, plain-English translation of the input text."
    )
    suggested_intent: str = Field(
        description=(
            "A tentative, suggestive hypothesis of the sender's underlying intent, motivation, or goal. "
            "Frame this as an interpretation rather than a concrete certainty (e.g., using phrasing like "
            "'The sender may be looking to...', 'This likely signals an attempt to...', or 'Possible intent:')."
        )
    )
    flagged_jargon: list[str] = Field(
        description="List of specific buzzwords or obfuscated phrases identified in the original text."
    )
    actionable_tasks: list[str] = Field(
        description="Bullet points of concrete tasks, deadlines, or expectations extracted from the message."
    )


class TextPayload(BaseModel):
    text: str


# --- API Endpoint ---

@app.post("/analyze")
async def analyze_text(payload: TextPayload):
    raw_text = payload.text.strip()
    if not raw_text:
        raise HTTPException(
            status_code=400,
            detail="Input text cannot be empty. Please provide corporate text to analyze.",
        )

    # --- Test Trigger Checkers ---
    trigger_lower = raw_text.lower()
    if trigger_lower in ("mock:overloaded", "test:overloaded", "trigger:overload"):
        raise HTTPException(
            status_code=429,
            detail="The AI engine is currently experiencing high demand and capacity limits. Please wait a moment and retry.",
        )

    if trigger_lower in ("mock:down", "test:down", "trigger:down"):
        raise HTTPException(
            status_code=503,
            detail="Unable to establish a connection with the AI translation service. The server may be unreachable or undergoing maintenance.",
        )

    if not client:
        raise HTTPException(
            status_code=500,
            detail="Gemini API is not configured. Please verify that GEMINI_API_KEY is present in your environment variables.",
        )

    input_metrics = calculate_metrics(raw_text)
    baseline_jargon = baseline_engine.find_baseline_matches(raw_text)

    baseline_context = (
        f"Baseline Jargon Terms Detected from Benchmark Corpus: {', '.join(baseline_jargon)}\n"
        if baseline_jargon
        else ""
    )

    prompt = f"""
    Analyze the following communication. Translate all obfuscated business buzzwords, technical jargon,
    euphemisms, and passive-aggressive framing into clear, direct, and actionable plain English.

    When analyzing the suggested intent:
    - Do NOT state the intent as an absolute or concrete certainty.
    - Frame the output as a plausible suggestion, possibility, or hypothesis about what the author might actually mean or want (e.g., "The author may be attempting to...", "This suggests a desire to...", "Likely seeking:").

    {baseline_context}
    Input Text:
    \"\"\"{raw_text}\"\"\"
    """

    parsed_result = None
    last_error_code = None
    last_error_message = None

    for model_name in MODELS:
        print(f"Trying model: {model_name}")
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=AnalysisResponse,
                        temperature=0.2,
                    ),
                )
                parsed_result = AnalysisResponse.model_validate_json(
                    response.text
                )
                print(f"Successfully processed with {model_name}")
                break
            except APIError as e:
                last_error_code = getattr(e, "code", 500)
                last_error_message = str(e)
                if last_error_code in (429, 503):
                    time.sleep(1.0 * (attempt + 1))
                    continue
                else:
                    break
            except Exception as e:
                last_error_message = str(e)
                break

        if parsed_result:
            break

    if not parsed_result:
        if last_error_code == 429:
            raise HTTPException(
                status_code=429,
                detail="All translation engine models are currently overloaded. Please wait 10 seconds and try again.",
            )
        elif last_error_code in (500, 502, 503, 504):
            raise HTTPException(
                status_code=503,
                detail="The AI service is temporarily unavailable. Please retry shortly.",
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Translation engine encountered an unhandled issue: {last_error_message or 'Unknown error'}",
            )

    combined_jargon = list(
        dict.fromkeys(parsed_result.flagged_jargon + baseline_jargon)
    )
    analysis_data = parsed_result.model_dump()
    analysis_data["flagged_jargon"] = combined_jargon

    output_metrics = calculate_metrics(parsed_result.plain_english_translation)
    gunning_fog_drop = round(
        input_metrics["gunning_fog"] - output_metrics["gunning_fog"], 1
    )

    return {
        "original_text": raw_text,
        "metrics": {
            "before": input_metrics,
            "after": output_metrics,
            "flesch_lift": round(
                output_metrics["flesch_reading_ease"]
                - input_metrics["flesch_reading_ease"],
                1,
            ),
            "gunning_fog_drop": gunning_fog_drop,
            "corpus_baseline": {
                "dataset_mean_readability": baseline_engine.baseline_stats[
                    "mean_readability"
                ],
                "dataset_samples": baseline_engine.baseline_stats[
                    "sample_count"
                ],
            },
        },
        "analysis": analysis_data,
        "baseline_hits": baseline_jargon,
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
