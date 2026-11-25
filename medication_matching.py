"""
Fuzzy medication name matching with RapidFuzz.

OCR text is noisy: letters get swapped, words are missing, packaging may show
brand names instead of generics. RapidFuzz scores candidates so the user can
confirm before any openFDA lookup. Low-confidence / ambiguous results are
rejected instead of guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from rapidfuzz import fuzz, process

from utils import extract_candidate_tokens, extract_strengths, normalize_text


# Starter vocabulary for demos / unit tests. Expand this list (or load from CSV)
# as you collect packaging for the research dataset. Include both brand and
# generic names that appear on labels.
DEFAULT_MEDICATION_VOCABULARY: List[str] = [
    "acetaminophen",
    "ibuprofen",
    "aspirin",
    "naproxen",
    "diphenhydramine",
    "loratadine",
    "cetirizine",
    "fexofenadine",
    "pseudoephedrine",
    "guaifenesin",
    "dextromethorphan",
    "omeprazole",
    "famotidine",
    "calcium carbonate",
    "magnesium hydroxide",
    "loperamide",
    "bismuth subsalicylate",
    "melatonin",
    "vitamin d",
    "vitamin c",
    "multivitamin",
    "amoxicillin",
    "azithromycin",
    "metformin",
    "atorvastatin",
    "lisinopril",
    "amlodipine",
    "metoprolol",
    "losartan",
    "sertraline",
    "fluoxetine",
    "levothyroxine",
    "albuterol",
    "prednisone",
    "hydrochlorothiazide",
    "tylenol",
    "advil",
    "motrin",
    "aleve",
    "benadryl",
    "claritin",
    "zyrtec",
    "allegra",
    "mucinex",
    "robitussin",
    "prilosec",
    "pepcid",
    "tums",
    "imodium",
    "pepto bismol",
]


@dataclass
class MatchCandidate:
    """One fuzzy-matched medication suggestion."""

    name: str
    score: float  # 0–100
    matched_query: str


@dataclass
class MatchResult:
    """Matching outcome with explicit accept / reject status."""

    candidates: List[MatchCandidate]
    accepted: bool
    reason: str
    extracted_strengths: List[str]
    query_text: str

    @property
    def top(self) -> Optional[MatchCandidate]:
        return self.candidates[0] if self.candidates else None


def load_vocabulary(extra_names: Optional[Iterable[str]] = None) -> List[str]:
    """Combine the default vocabulary with any extra names from ground truth."""
    names = {normalize_text(n): n for n in DEFAULT_MEDICATION_VOCABULARY}
    if extra_names:
        for name in extra_names:
            if not name:
                continue
            key = normalize_text(name)
            if key and key not in names:
                # Preserve a readable display form (title case for multi-word).
                names[key] = name.strip()
    # Return display names sorted for stable UI order.
    return sorted(names.values(), key=lambda s: s.lower())


def _build_queries(ocr_text: str) -> List[str]:
    """
    Build several query strings from OCR text.

    Matching the full string AND individual tokens helps when OCR only
    recovers part of the label.
    """
    normalized = normalize_text(ocr_text)
    queries = []
    if normalized:
        queries.append(normalized)

    tokens = extract_candidate_tokens(ocr_text)
    # Also try bigrams: "pepto bismol", "calcium carbonate", etc.
    for i in range(len(tokens) - 1):
        queries.append(f"{tokens[i]} {tokens[i + 1]}")
    queries.extend(tokens)

    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique


def match_medications(
    ocr_text: str,
    vocabulary: Optional[Sequence[str]] = None,
    top_k: int = 5,
    min_score: float = 70.0,
    ambiguity_gap: float = 5.0,
) -> MatchResult:
    """
    Fuzzy-match OCR text against a medication vocabulary.

    Rejection rules (prefer 'I don't know' over a wrong guess):
    - No candidate reaches `min_score`.
    - Top two candidates are within `ambiguity_gap` points (ambiguous).
    """
    vocab = list(vocabulary) if vocabulary is not None else load_vocabulary()
    strengths = extract_strengths(ocr_text)
    queries = _build_queries(ocr_text)

    if not queries or not vocab:
        return MatchResult(
            candidates=[],
            accepted=False,
            reason="No OCR text or empty vocabulary.",
            extracted_strengths=strengths,
            query_text=normalize_text(ocr_text),
        )

    # Score every vocabulary entry by the best query against it.
    best_by_name = {}
    for query in queries:
        # token_set_ratio tolerates extra words and reordered tokens.
        matches = process.extract(
            query,
            vocab,
            scorer=fuzz.token_set_ratio,
            limit=top_k,
        )
        for name, score, _ in matches:
            prev = best_by_name.get(name)
            if prev is None or score > prev[0]:
                best_by_name[name] = (score, query)

    ranked = sorted(best_by_name.items(), key=lambda kv: kv[1][0], reverse=True)
    candidates = [
        MatchCandidate(name=name, score=float(score), matched_query=query)
        for name, (score, query) in ranked[:top_k]
    ]

    if not candidates or candidates[0].score < min_score:
        return MatchResult(
            candidates=candidates,
            accepted=False,
            reason=(
                f"Top score below threshold ({min_score}). "
                "Rejecting low-confidence match."
            ),
            extracted_strengths=strengths,
            query_text=normalize_text(ocr_text),
        )

    if len(candidates) >= 2:
        gap = candidates[0].score - candidates[1].score
        if gap < ambiguity_gap and candidates[1].score >= min_score:
            return MatchResult(
                candidates=candidates,
                accepted=False,
                reason=(
                    f"Ambiguous result: top scores are within {ambiguity_gap} "
                    "points. Please confirm manually."
                ),
                extracted_strengths=strengths,
                query_text=normalize_text(ocr_text),
            )

    return MatchResult(
        candidates=candidates,
        accepted=True,
        reason="Top candidate exceeds threshold with a clear margin.",
        extracted_strengths=strengths,
        query_text=normalize_text(ocr_text),
    )


def names_match(predicted: str, expected: str, threshold: float = 85.0) -> bool:
    """Soft equality used by evaluation metrics for medication names."""
    if not predicted or not expected:
        return False
    score = fuzz.token_set_ratio(normalize_text(predicted), normalize_text(expected))
    return score >= threshold


def strengths_match(predicted: Sequence[str], expected: str) -> bool:
    """True if any extracted strength matches the expected strength string."""
    if not expected:
        return False
    expected_norm = normalize_text(expected)
    for strength in predicted:
        if normalize_text(strength) == expected_norm:
            return True
        # Also allow fuzzy near-matches for OCR digit noise (e.g. 2O0 mg).
        if fuzz.ratio(normalize_text(strength), expected_norm) >= 90:
            return True
    return False
