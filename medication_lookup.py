"""
openFDA Drug Label API client.

Endpoint: https://api.fda.gov/drug/label.json

This module only retrieves publicly available label sections. It does NOT
provide personalized medical advice or dosage recommendations. Missing fields
are reported as unavailable rather than invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import requests

from utils import first_nonempty, truncate


OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
DEFAULT_TIMEOUT_SECONDS = 12

DISCLAIMER = (
    "MedScan is a computer vision class project, not a medical device. "
    "Always verify medication identity against the original packaging and "
    "consult a pharmacist or clinician for medical decisions. Information "
    "below comes from the public openFDA drug label API and may be incomplete."
)


@dataclass
class MedicationInfo:
    """Subset of openFDA label fields shown in the demo UI."""

    query: str
    found: bool
    brand_name: Optional[str] = None
    generic_name: Optional[str] = None
    active_ingredients: Optional[str] = None
    indications_and_usage: Optional[str] = None
    warnings: Optional[str] = None
    adverse_reactions: Optional[str] = None  # "side effects" when present
    manufacturer: Optional[str] = None
    drug_interactions: Optional[str] = None
    source: str = "openFDA Drug Label API (https://api.fda.gov/drug/label.json)"
    disclaimer: str = DISCLAIMER
    error: Optional[str] = None
    raw_openfda: Dict[str, Any] = field(default_factory=dict)

    def as_display_dict(self) -> Dict[str, str]:
        """Flatten for Streamlit tables / markdown."""

        def show(value: Optional[str]) -> str:
            return value if value else "Not available in openFDA response"

        return {
            "Brand name": show(self.brand_name),
            "Generic name": show(self.generic_name),
            "Active ingredients": show(self.active_ingredients),
            "Indications and usage": show(self.indications_and_usage),
            "Warnings": show(self.warnings),
            "Side effects / adverse reactions": show(self.adverse_reactions),
            "Manufacturer": show(self.manufacturer),
            "Drug interactions": show(self.drug_interactions),
            "Source": self.source,
        }


def _join_field(value: Any, max_len: int = 1200) -> Optional[str]:
    """openFDA fields are often lists of long strings."""
    if value is None:
        return None
    if isinstance(value, list):
        text = " ".join(str(v) for v in value if v)
    else:
        text = str(value)
    text = text.strip()
    if not text:
        return None
    return truncate(text, max_len)


def _extract_from_result(result: Dict[str, Any]) -> Dict[str, Optional[str]]:
    openfda = result.get("openfda") or {}
    return {
        "brand_name": _join_field(openfda.get("brand_name"), max_len=200),
        "generic_name": _join_field(openfda.get("generic_name"), max_len=200),
        "active_ingredients": _join_field(
            result.get("active_ingredient") or openfda.get("substance_name"),
            max_len=400,
        ),
        "indications_and_usage": _join_field(result.get("indications_and_usage")),
        "warnings": _join_field(
            result.get("warnings") or result.get("warnings_and_cautions")
        ),
        "adverse_reactions": _join_field(result.get("adverse_reactions")),
        "manufacturer": _join_field(
            openfda.get("manufacturer_name"), max_len=200
        ),
        "drug_interactions": _join_field(result.get("drug_interactions")),
        "raw_openfda": openfda,
    }


def _search(query: str, search_field: str, limit: int = 1) -> Dict[str, Any]:
    """Perform one openFDA search. Raises requests exceptions to caller."""
    # quote keeps spaces/special chars valid in the search expression.
    search = f'{search_field}:"{query}"'
    params = {"search": search, "limit": limit}
    response = requests.get(
        OPENFDA_LABEL_URL,
        params=params,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        headers={"User-Agent": "MedScan-CS415-StudentProject/1.0"},
    )
    # openFDA returns 404 JSON when no matches exist — treat as not found.
    if response.status_code == 404:
        return {"results": []}
    response.raise_for_status()
    return response.json()


def lookup_medication(name: str) -> MedicationInfo:
    """
    Look up a medication by brand or generic name.

    Search order:
    1. openfda.brand_name
    2. openfda.generic_name
    3. openfda.substance_name
    """
    query = (name or "").strip()
    if not query:
        return MedicationInfo(
            query=query,
            found=False,
            error="Empty medication name.",
        )

    fields_to_try = [
        "openfda.brand_name",
        "openfda.generic_name",
        "openfda.substance_name",
    ]

    last_error = None
    for field in fields_to_try:
        try:
            payload = _search(query, field)
        except requests.Timeout:
            return MedicationInfo(
                query=query,
                found=False,
                error="openFDA request timed out. Try again later.",
            )
        except requests.RequestException as exc:
            last_error = str(exc)
            continue

        results = payload.get("results") or []
        if not results:
            continue

        extracted = _extract_from_result(results[0])
        return MedicationInfo(
            query=query,
            found=True,
            brand_name=extracted["brand_name"],
            generic_name=extracted["generic_name"],
            active_ingredients=extracted["active_ingredients"],
            indications_and_usage=extracted["indications_and_usage"],
            warnings=extracted["warnings"],
            adverse_reactions=extracted["adverse_reactions"],
            manufacturer=extracted["manufacturer"],
            drug_interactions=extracted["drug_interactions"],
            raw_openfda=extracted["raw_openfda"] or {},
        )

    return MedicationInfo(
        query=query,
        found=False,
        error=first_nonempty(
            [
                f"No openFDA label found for '{query}'.",
                last_error,
            ]
        ),
    )
