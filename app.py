from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import tempfile
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


SCHEMA_VERSION = "1.0"
EXTRACTOR_VERSION = "rule-based-heading-chunker-v1"

ERROR_CODES = {
    "validation_failure": "E001",
    "empty_corpus": "E002",
    "acl_denied": "E003",
    "index_build_failure": "E004",
    "search_failure": "E005",
    "artifact_generation_failure": "E006",
    "pipeline_failure": "E007",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class RawDocument(StrictModel):
    doc_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = ""
    acl: list[str] = Field(default_factory=lambda: ["public"])
    source_uri: str | None = None

    @field_validator("acl")
    @classmethod
    def _clean_acl(cls, v: Iterable[str]) -> list[str]:
        cleaned = [str(item).strip().lower() for item in v if str(item).strip()]
        return cleaned or ["public"]


class IngestionRequest(StrictModel):
    schema_version: str = SCHEMA_VERSION
    documents: list[RawDocument] = Field(default_factory=list)


class Chunk(StrictModel):
    schema_version: str = SCHEMA_VERSION
    chunk_id: str
    doc_id: str
    title: str
    text: str
    heading: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    acl: list[str]
    source_sha256: str
    extractor_version: str
    confidence: float = Field(ge=0.0, le=1.0)
    ingested_at: str
    char_count: int = Field(ge=0)

    @field_validator("text")
    @classmethod
    def _non_empty_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("chunk text cannot be empty")
        return v

    @model_validator(mode="after")
    def _validate_offsets(self) -> "Chunk":
        if self.end_offset < self.start_offset:
            raise ValueError("end_offset must be >= start_offset")
        return self


class SearchRequest(StrictModel):
    schema_version: str = SCHEMA_VERSION
    query: str = ""
    top_k: int = Field(default=5, ge=1, le=100)
    role: str = "public"
    lexical_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    dense_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1, le=1000)

    @model_validator(mode="after")
    def _sanitize(self) -> "SearchRequest":
        self.query = self.query.strip()
        self.role = self.role.strip().lower() or "public"
        if self.lexical_weight == 0.0 and self.dense_weight == 0.0:
            self.dense_weight = 1.0
        return self


class SearchResult(StrictModel):
    schema_version: str = SCHEMA_VERSION
    rank: int = Field(ge=1)
    chunk_id: str
    doc_id: str
    title: str
    heading: str
    snippet: str
    lexical_score: float = Field(ge=0.0)
    dense_score: float = Field(ge=0.0)
    fused_score: float = Field(ge=0.0)
    lexical_rank: int = Field(ge=0)
    dense_rank: int = Field(ge=0)
    acl: list[str]
    source_sha256: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    extractor_version: str
    confidence: float = Field(ge=0.0, le=1.0)
    ingested_at: str


class AppSettings(StrictModel):
    seed: int = 7
    max_chunk_chars: int = Field(default=700, ge=50, le=5000)
    overlap_chars: int = Field(default=120, ge=0, le=2000)
    top_k_default: int = Field(default=5, ge=1, le=20)
    lexical_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    dense_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    demo_role: str = "staff"
    demo_query: str = "How does hybrid retrieval improve evidence ranking?"
    auto_animation: bool = True

    @model_validator(mode="after")
    def _sanitize_weights(self) -> "AppSettings":
        if self.lexical_weight == 0.0 and self.dense_weight == 0.0:
            self.dense_weight = 1.0
        return self


DEMO_DOCUMENTS: list[RawDocument] = [
    RawDocument(
        doc_id="policy-remote-work",
        title="Remote Work Policy",
        acl=["public", "staff"],
        source_uri="synthetic://policy-remote-work",
        text="""# Purpose
This policy defines how teams use remote work, secure equipment, and protected data.
Employees may work from a café, co-working space, or home office if the connection is trusted.

# Eligibility
Roles that handle customer personal data must complete security training before remote access.
Naïve copy-paste of credentials into unmanaged tools is prohibited.

# Equipment
Approved laptops must have disk encryption, endpoint detection, and automatic patching.
If a device is lost, the employee must report it within four hours.

# Security
Remote sessions must use the corporate VPN or zero-trust gateway.
Public Wi-Fi is allowed only when the VPN is active.
""",
    ),
    RawDocument(
        doc_id="faq-product-alpha",
        title="Product Alpha FAQ",
        acl=["public"],
        source_uri="synthetic://faq-product-alpha",
        text="""Section 1 General
Q: What is Product Alpha?
A: Product Alpha is a workflow automation tool for small teams.

Q: Does Product Alpha support offline mode?
A: Basic editing is cached locally, but sync requires network access.

Section 2 Billing
Q: How is billing calculated?
A: Billing is monthly per active workspace member.
Unused seats are prorated at the end of the billing period.

Q: Do you offer discounts?
A: Annual plans receive a discount compared with monthly billing.
""",
    ),
    RawDocument(
        doc_id="incident-2026-04-latency",
        title="Incident Report: API Latency Spike",
        acl=["staff"],
        source_uri="synthetic://incident-2026-04-latency",
        text="""# Summary
On 2026-04-18, the public API experienced elevated p95 latency for 42 minutes.
The affected service was the search gateway.

# Timeline
09:12 UTC: Alert triggered for p95 latency above threshold.
09:18 UTC: On-call engineer mitigated by rolling back the index refresh job.
09:54 UTC: Latency returned to baseline.

# Root cause
A bad embedding batch caused repeated cache misses and increased CPU pressure.
The rollback restored the previous index snapshot.

# Corrective actions
Add canary evaluation before promoting new embeddings.
Add latency budget alarms per retrieval stage.
""",
    ),
    RawDocument(
        doc_id="contract-master-services-agreement",
        title="Master Services Agreement Extract",
        acl=["admin"],
        source_uri="synthetic://contract-master-services-agreement",
        text="""Clause 7 Limitation of Liability
Except for breach of confidentiality or willful misconduct, neither party shall be liable
for indirect, incidental, or consequential damages.

Clause 8 Data Protection
Supplier shall process customer data only on documented instructions and shall maintain
appropriate technical and organizational measures.

Clause 9 Audit Rights
Customer may audit compliance once per calendar year with thirty days notice.
""",
    ),
    RawDocument(
        doc_id="research-note-vector-search",
        title="Research Note: Hybrid Vector Search",
        acl=["public", "staff"],
        source_uri="synthetic://research-note-vector-search",
        text="""# Abstract
Hybrid retrieval combines lexical signals with dense embedding signals.
In small corpora, lexical matching remains strong for exact terms, while dense retrieval
helps with paraphrases and conceptual overlap.

# Method
We compare BM25-like lexical scoring, TF-IDF cosine similarity, truncated SVD dense vectors,
and reciprocal rank fusion. Filters are applied before ranking to enforce access control.

# Results
On a synthetic mini-benchmark, hybrid retrieval improved recall and nDCG@10 over lexical-only
and dense-only baselines. Error analysis showed relation-type sensitivity and the need for
citation faithfulness checks.

# Operational notes
CPU-only deployment is practical for small and medium document libraries.
Embedding outputs are candidates, not facts, until validated and reviewed.
""",
    ),
]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _clamp_int(value: int, low: int, high: int, default: int) -> int:
    try:
        iv = int(value)
    except Exception:
        iv = default
    return max(low, min(high, iv))


def _clamp_float(value: float, low: float, high: float, default: float) -> float:
    try:
        fv = float(value)
    except Exception:
        fv = default
    return max(low, min(high, fv))


def config_hash(settings: AppSettings, include: tuple[str, ...] | None = None) -> str:
    data = settings.model_dump(mode="json")
    if include is not None:
        data = {k: data[k] for k in include if k in data}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def log_event(stage: str, **kwargs: Any) -> None:
    payload: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "schema_version": SCHEMA_VERSION,
        **kwargs,
    }
    logging.getLogger("ssrp").info(json.dumps(payload, default=str))
    st.session_state.setdefault("logs", []).append(payload)
    if len(st.session_state["logs"]) > 500:
        st.session_state["logs"] = st.session_state["logs"][-500:]


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_HEADING_PREFIX_RE = re.compile(
    r"^(section|chapter|article|clause|appendix)\s+[0-9A-Za-z.\-]+",
    re.IGNORECASE,
)


def _is_heading_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return True
    return bool(_HEADING_PREFIX_RE.match(stripped)) and len(stripped) <= 180


def _split_heading_segments(text: str) -> list[tuple[str, int, int, str]]:
    if not text:
        return []

    lines = text.splitlines(keepends=True)
    segments: list[tuple[str, int, int, str]] = []

    current_heading = "Untitled"
    current_start = 0
    current_parts: list[str] = []
    pos = 0

    for line in lines:
        line_start = pos
        line_end = pos + len(line)

        if _is_heading_line(line):
            if current_parts and "".join(current_parts).strip():
                segments.append(
                    (
                        current_heading,
                        current_start,
                        line_start,
                        "".join(current_parts),
                    )
                )

            current_heading = line.strip().lstrip("#").strip() or line.strip()
            current_start = line_start
            current_parts = [line]
        else:
            current_parts.append(line)

        pos = line_end

    if current_parts and "".join(current_parts).strip():
        segments.append((current_heading, current_start, pos, "".join(current_parts)))

    return segments


def _token_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _make_chunks_from_tokens(
    text: str,
    tokens: list[tuple[int, int]],
    base_offset: int,
    max_chars: int,
    overlap_chars: int,
) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    i = 0
    n = len(tokens)

    while i < n:
        j = i

        while j + 1 < n:
            next_end = tokens[j + 1][1]
            slice_len = next_end - tokens[i][0]
            if slice_len > max_chars:
                break
            j += 1

        chunk_start_rel = tokens[i][0]
        chunk_end_rel = tokens[j][1]
        chunk_text = text[chunk_start_rel:chunk_end_rel]

        overlap_count = 0
        if overlap_chars > 0:
            for k in range(j, i - 1, -1):
                suffix_len = chunk_end_rel - tokens[k][0]
                if suffix_len <= overlap_chars:
                    overlap_count = j - k + 1
                else:
                    break

        next_i = j - overlap_count + 1
        if next_i <= i:
            next_i = i + 1

        out.append(
            (
                chunk_text,
                base_offset + chunk_start_rel,
                base_offset + chunk_end_rel,
            )
        )
        i = next_i

    return out


def chunk_document(doc: RawDocument, settings: AppSettings) -> list[Chunk]:
    source_sha = sha256_text(doc.text)
    segments = _split_heading_segments(doc.text)
    chunks: list[Chunk] = []
    counter = 0
    ingested_at = datetime.now(timezone.utc).isoformat()

    for heading, seg_start, _seg_end, seg_text in segments:
        tokens = _token_spans(seg_text)
        if not tokens:
            continue

        for chunk_text, start, end in _make_chunks_from_tokens(
            seg_text,
            tokens,
            seg_start,
            settings.max_chunk_chars,
            settings.overlap_chars,
        ):
            if not chunk_text.strip():
                continue

            counter += 1
            chunks.append(
                Chunk(
                    schema_version=SCHEMA_VERSION,
                    chunk_id=f"{doc.doc_id}::c{counter:04d}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=chunk_text,
                    heading=heading,
                    start_offset=start,
                    end_offset=end,
                    acl=doc.acl,
                    source_sha256=source_sha,
                    extractor_version=EXTRACTOR_VERSION,
                    confidence=1.0,
                    ingested_at=ingested_at,
                    char_count=len(chunk_text.strip()),
                )
            )

    return chunks


def _is_allowed(acl: Any, role: str) -> bool:
    acl_set = {str(x).strip().lower() for x in acl}
    role = role.strip().lower()

    if role == "admin":
        return True
    if role == "staff":
        return bool(acl_set.intersection({"staff", "public"}))
    return "public" in acl_set


def highlight_snippet(text: str, query: str, max_chars: int = 280) -> str:
    if not text:
        return ""

    terms = [t for t in re.findall(r"[A-Za-z0-9_\-']{2,}", query.lower()) if t]
    unique_terms = list(dict.fromkeys(terms))

    match = None
    for term in unique_terms:
        m = re.search(rf"(?i)\b{re.escape(term)}\b", text)
        if m and (match is None or m.start() < match.start()):
            match = m

    if match:
        start = max(0, match.start() - 90)
        end = min(len(text), match.end() + 140)
    else:
        start = 0
        end = min(len(text), max_chars)

    snippet = text[start:end].strip()
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"

    for term in unique_terms:
        snippet = re.sub(rf"(?i)\b{re.escape(term)}\b", f"**{term}**", snippet)

    return snippet


class SearchIndex:
    def __init__(self, chunks: list[Chunk], settings: AppSettings) -> None:
        self.settings = settings
        self.chunks = chunks
        self.df = (
            pd.DataFrame([c.model_dump(mode="json") for c in chunks])
            if chunks
            else pd.DataFrame(
                columns=[
                    "schema_version",
                    "chunk_id",
                    "doc_id",
                    "title",
                    "text",
                    "heading",
                    "start_offset",
                    "end_offset",
                    "acl",
                    "source_sha256",
                    "extractor_version",
                    "confidence",
                    "ingested_at",
                    "char_count",
                ]
            )
        )
        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None
        self.tfidf_matrix: Any = None
        self.dense_matrix: Any = None
        self._build()

    def _build(self) -> None:
        if self.df.empty:
            return

        texts = self.df["text"].astype(str).tolist()
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

        n_samples, n_features = self.tfidf_matrix.shape
        if n_features <= 1 or n_samples <= 1:
            self.dense_matrix = normalize(self.tfidf_matrix, norm="l2", axis=1)
            return

        n_components = min(32, n_features - 1, n_samples - 1)
        if n_components < 1:
            self.dense_matrix = normalize(self.tfidf_matrix, norm="l2", axis=1)
            return

        try:
            self.svd = TruncatedSVD(
                n_components=n_components,
                random_state=self.settings.seed,
            )
            dense = self.svd.fit_transform(self.tfidf_matrix)
            self.dense_matrix = normalize(dense, norm="l2", axis=1)
        except Exception:
            self.svd = None
            self.dense_matrix = normalize(self.tfidf_matrix, norm="l2", axis=1)

    def search(self, request: SearchRequest) -> list[SearchResult]:
        if self.df.empty or not request.query:
            return []

        if self.vectorizer is None or self.tfidf_matrix is None:
            return []

        allowed = np.array(
            [_is_allowed(acl, request.role) for acl in self.df["acl"]],
            dtype=bool,
        )
        allowed_indices = np.where(allowed)[0]
        if allowed_indices.size == 0:
            return []

        q_vec = self.vectorizer.transform([request.query])
        lexical = np.asarray(
            cosine_similarity(q_vec, self.tfidf_matrix)[0],
            dtype=float,
        ).reshape(-1)
        lexical = np.nan_to_num(lexical, nan=0.0, posinf=0.0, neginf=0.0)

        if self.svd is not None and self.dense_matrix is not None:
            q_dense = normalize(self.svd.transform(q_vec), norm="l2", axis=1)
            dense = np.asarray(
                cosine_similarity(q_dense, self.dense_matrix)[0],
                dtype=float,
            ).reshape(-1)
            dense = np.nan_to_num(dense, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            dense = lexical.copy()

        if (
            float(np.max(lexical[allowed_indices])) == 0.0
            and float(np.max(dense[allowed_indices])) == 0.0
        ):
            return []

        lexical[~allowed] = -np.inf
        dense[~allowed] = -np.inf

        n = len(self.df)
        lex_rank = np.zeros(n, dtype=int)
        dense_rank = np.zeros(n, dtype=int)

        for rank, idx in enumerate(np.argsort(-lexical, kind="stable"), start=1):
            if not np.isfinite(lexical[idx]):
                break
            lex_rank[idx] = rank

        for rank, idx in enumerate(np.argsort(-dense, kind="stable"), start=1):
            if not np.isfinite(dense[idx]):
                break
            dense_rank[idx] = rank

        fused = np.zeros(n, dtype=float)
        k = request.rrf_k

        for idx in allowed_indices:
            lr = int(lex_rank[idx])
            dr = int(dense_rank[idx])
            if lr > 0:
                fused[idx] += request.lexical_weight / float(k + lr)
            if dr > 0:
                fused[idx] += request.dense_weight / float(k + dr)

        candidate_idx = allowed_indices[np.argsort(-fused[allowed_indices], kind="stable")]
        top = candidate_idx[: request.top_k]

        results: list[SearchResult] = []
        for rank, idx in enumerate(top, start=1):
            row = self.df.iloc[int(idx)]
            lex_score = 0.0 if not np.isfinite(lexical[idx]) else float(max(0.0, lexical[idx]))
            dense_score = 0.0 if not np.isfinite(dense[idx]) else float(max(0.0, dense[idx]))
            fused_score = float(max(0.0, fused[idx]))

            results.append(
                SearchResult(
                    schema_version=SCHEMA_VERSION,
                    rank=rank,
                    chunk_id=str(row["chunk_id"]),
                    doc_id=str(row["doc_id"]),
                    title=str(row["title"]),
                    heading=str(row["heading"]),
                    snippet=highlight_snippet(str(row["text"]), request.query),
                    lexical_score=lex_score,
                    dense_score=dense_score,
                    fused_score=fused_score,
                    lexical_rank=int(lex_rank[idx]),
                    dense_rank=int(dense_rank[idx]),
                    acl=list(row["acl"]),
                    source_sha256=str(row["source_sha256"]),
                    start_offset=int(row["start_offset"]),
                    end_offset=int(row["end_offset"]),
                    extractor_version=str(row["extractor_version"]),
                    confidence=float(row["confidence"]),
                    ingested_at=str(row["ingested_at"]),
                )
            )

        return results


def build_index_with_progress(
    docs: list[RawDocument],
    settings: AppSettings,
) -> tuple[SearchIndex, list[Chunk]]:
    random.seed(settings.seed)
    np.random.seed(settings.seed)

    progress = st.progress(0.0, text="Validating documents")
    log_event("pipeline_start", seed=settings.seed, documents=len(docs))

    req = IngestionRequest(documents=docs)
    time.sleep(0.05)

    progress.progress(0.18, text="Fingerprinting and chunking")
    chunks: list[Chunk] = []

    for doc in req.documents:
        doc_chunks = chunk_document(doc, settings)
        chunks.extend(doc_chunks)
        log_event(
            "document_chunked",
            doc_id=doc.doc_id,
            chunks=len(doc_chunks),
            sha=sha256_text(doc.text)[:12],
        )
        time.sleep(0.02)

    if not chunks:
        log_event(
            "warning",
            code=ERROR_CODES["empty_corpus"],
            message="No chunks produced from documents.",
        )

    progress.progress(0.48, text="Building lexical TF-IDF index")
    index = SearchIndex(chunks, settings)

    progress.progress(0.82, text="Building dense SVD index")
    time.sleep(0.05)

    progress.progress(1.0, text="Ready")
    log_event(
        "pipeline_complete",
        chunks=len(chunks),
        features=int(index.tfidf_matrix.shape[1]) if index.tfidf_matrix is not None else 0,
    )

    return index, chunks


def request_fingerprint(request: SearchRequest, ingestion_hash: str) -> str:
    payload = request.model_dump(mode="json")
    payload["ingestion_hash"] = ingestion_hash
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def summarize_index(
    index: SearchIndex,
    results: list[SearchResult],
    request: SearchRequest,
    latency_ms: float,
    cfg_hash: str,
) -> dict[str, Any]:
    df = index.df

    if df.empty:
        return {
            "documents": 0,
            "chunks": 0,
            "avg_chunk_chars": 0.0,
            "allowed_chunks": 0,
            "returned": 0,
            "latency_ms": round(float(latency_ms), 2),
            "config_hash": cfg_hash,
            "lexical_features": 0,
            "dense_dimensions": 0,
        }

    allowed_count = sum(_is_allowed(acl, request.role) for acl in df["acl"])
    dense_dims = (
        int(index.dense_matrix.shape[1])
        if index.svd is not None and index.dense_matrix is not None
        else 0
    )

    return {
        "documents": int(df["doc_id"].nunique()),
        "chunks": int(len(df)),
        "avg_chunk_chars": float(df["char_count"].mean()),
        "allowed_chunks": int(allowed_count),
        "returned": len(results),
        "latency_ms": round(float(latency_ms), 2),
        "config_hash": cfg_hash,
        "lexical_features": int(index.tfidf_matrix.shape[1]) if index.tfidf_matrix is not None else 0,
        "dense_dimensions": dense_dims,
    }


_FONT: ImageFont.ImageFont | None = None


def _get_font() -> ImageFont.ImageFont:
    global _FONT
    if _FONT is None:
        _FONT = ImageFont.load_default()
    return _FONT


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    max_width: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (226, 232, 240),
) -> int:
    words = str(text).split()
    if not words:
        return y

    line = ""
    line_height = getattr(font, "size", 12) + 6

    for word in words:
        test = (line + " " + word).strip()
        try:
            width = draw.textlength(test, font=font)
        except AttributeError:
            width = font.getsize(test)[0]

        if width <= max_width or not line:
            line = test
        else:
            draw.text((x, y), line, font=font, fill=fill)
            y += line_height
            line = word

    if line:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height

    return y


def _base_frame(title: str, subtitle: str = "") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    width, height = 720, 405
    img = Image.new("RGB", (width, height), (15, 23, 42))
    draw = ImageDraw.Draw(img)
    font = _get_font()

    draw.rectangle([0, 0, width, 58], fill=(30, 41, 59))
    draw.text((18, 18), title, font=font, fill=(56, 189, 248))

    if subtitle:
        _draw_wrapped(draw, subtitle, 18, 70, width - 36, font, fill=(203, 213, 225))

    return img, draw


def build_demo_clip(
    request: SearchRequest,
    results: list[SearchResult],
    stats: dict[str, Any],
    settings: AppSettings,
    cfg_hash: str,
) -> bytes:
    frames: list[Image.Image] = []
    font = _get_font()

    img, draw = _base_frame(
        "Semantic Search & Retrieval Platform",
        f"Live local demo · config {cfg_hash} · {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
    )
    _draw_wrapped(
        draw,
        "CPU-only · no API keys · deterministic ingestion · hybrid retrieval · ACL-aware ranking",
        18,
        120,
        684,
        font,
    )
    frames.append(img)

    img, draw = _base_frame(
        "1. Ingestion and fingerprinting",
        f"{stats['documents']} documents · SHA-256 raw fingerprints",
    )
    y = 110
    for doc in DEMO_DOCUMENTS[:4]:
        sha = sha256_text(doc.text)[:12]
        draw.text(
            (18, y),
            f"• {doc.title} [{', '.join(doc.acl)}] sha:{sha}",
            font=font,
            fill=(226, 232, 240),
        )
        y += 22
    frames.append(img)

    img, draw = _base_frame(
        "2. Heading-aware semantic chunking",
        f"max {settings.max_chunk_chars} chars · overlap {settings.overlap_chars} chars · chunks {stats['chunks']}",
    )
    _draw_wrapped(
        draw,
        "Chunks preserve document, heading, character offsets, ACL, extractor version, confidence, and ingestion timestamp.",
        18,
        110,
        684,
        font,
    )
    frames.append(img)

    img, draw = _base_frame(
        "3. Lexical + dense index",
        f"TF-IDF features {stats['lexical_features']} · SVD dims {stats['dense_dimensions']}",
    )
    _draw_wrapped(
        draw,
        "Lexical ranking captures exact terms. Dense SVD vectors capture paraphrase overlap. Reciprocal Rank Fusion combines both ranked lists.",
        18,
        110,
        684,
        font,
    )
    frames.append(img)

    img, draw = _base_frame(
        "4. Query and ACL filter",
        f"role={request.role} · top_k={request.top_k}",
    )
    _draw_wrapped(draw, f"Query: {request.query or '(empty)'}", 18, 110, 684, font)
    draw.text(
        (18, 180),
        f"Allowed chunks: {stats['allowed_chunks']} · returned: {stats['returned']}",
        font=font,
        fill=(34, 197, 94),
    )
    frames.append(img)

    if results:
        max_score = max((r.fused_score for r in results), default=0.0)
        if max_score <= 0:
            max_score = 1.0

        for r in results[:3]:
            img, draw = _base_frame(
                f"5. Ranked evidence #{r.rank}",
                f"{r.title} · {r.heading}",
            )
            y = 110
            y = _draw_wrapped(draw, r.snippet.replace("**", ""), 18, y, 684, font)

            bar_w = int((r.fused_score / max_score) * 684)
            draw.rectangle([18, y + 10, 18 + max(bar_w, 2), y + 26], fill=(56, 189, 248))

            draw.text(
                (18, y + 34),
                f"lexical={r.lexical_score:.3f} dense={r.dense_score:.3f} fused={r.fused_score:.5f}",
                font=font,
                fill=(203, 213, 225),
            )
            draw.text(
                (18, y + 56),
                f"offsets {r.start_offset}-{r.end_offset} · sha {r.source_sha256[:10]} · confidence {r.confidence:.2f}",
                font=font,
                fill=(148, 163, 184),
            )
            frames.append(img)
    else:
        img, draw = _base_frame("5. No permitted results", "Adjust role, query, or weights.")
        frames.append(img)

    img, draw = _base_frame(
        "6. Interactive app ready",
        "Edit query/role/weights in sidebar · download clip and provenance",
    )
    frames.append(img)

    buf = BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=900,
        loop=0,
        optimize=True,
    )
    return buf.getvalue()


def _sidebar_settings() -> tuple[AppSettings, bool, bool, bool]:
    st.sidebar.header("⚙️ Runtime controls")

    seed = st.sidebar.number_input(
        "Deterministic seed",
        min_value=0,
        max_value=999999,
        value=_clamp_int(_env_int("SSRP_SEED", 7), 0, 999999, 7),
        step=1,
    )

    max_chunk_chars = st.sidebar.slider(
        "Max chunk characters",
        min_value=120,
        max_value=2400,
        value=_clamp_int(_env_int("SSRP_MAX_CHUNK_CHARS", 700), 120, 2400, 700),
        step=40,
    )

    overlap_chars = st.sidebar.slider(
        "Overlap characters",
        min_value=0,
        max_value=600,
        value=_clamp_int(_env_int("SSRP_OVERLAP_CHARS", 120), 0, 600, 120),
        step=20,
    )

    top_k = st.sidebar.slider(
        "Top K",
        min_value=1,
        max_value=20,
        value=_clamp_int(_env_int("SSRP_TOP_K", 5), 1, 20, 5),
    )

    lexical_weight = st.sidebar.slider(
        "Lexical weight",
        min_value=0.0,
        max_value=1.0,
        value=_clamp_float(_env_float("SSRP_LEXICAL_WEIGHT", 0.5), 0.0, 1.0, 0.5),
        step=0.05,
    )

    dense_weight = st.sidebar.slider(
        "Dense weight",
        min_value=0.0,
        max_value=1.0,
        value=_clamp_float(_env_float("SSRP_DENSE_WEIGHT", 0.5), 0.0, 1.0, 0.5),
        step=0.05,
    )

    rrf_k = st.sidebar.number_input(
        "RRF k",
        min_value=1,
        max_value=1000,
        value=_clamp_int(_env_int("SSRP_RRF_K", 60), 1, 1000, 60),
        step=1,
    )

    role = st.sidebar.selectbox(
        "Viewer role",
        ["public", "staff", "admin"],
        index=1,
    )

    query = st.sidebar.text_area(
        "Query",
        value=os.getenv(
            "SSRP_DEMO_QUERY",
            "How does hybrid retrieval improve evidence ranking?",
        ),
        height=70,
    )

    auto_animation = st.sidebar.checkbox(
        "Auto-generate demo clip",
        value=_env_bool("SSRP_AUTO_ANIMATION", True),
    )

    st.sidebar.markdown("---")

    btn_search = st.sidebar.button("Run search", type="primary")
    btn_rebuild = st.sidebar.button("Rebuild index")
    btn_clip = st.sidebar.button("Regenerate demo clip")

    settings = AppSettings(
        seed=int(seed),
        max_chunk_chars=int(max_chunk_chars),
        overlap_chars=int(overlap_chars),
        top_k_default=int(top_k),
        lexical_weight=float(lexical_weight),
        dense_weight=float(dense_weight),
        rrf_k=int(rrf_k),
        demo_role=str(role),
        demo_query=str(query),
        auto_animation=bool(auto_animation),
    )

    return settings, bool(btn_search), bool(btn_rebuild), bool(btn_clip)


def main() -> None:
    st.set_page_config(
        page_title="Semantic Search & Retrieval Platform",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    if "logs" not in st.session_state:
        st.session_state.logs = []

    settings, btn_search, btn_rebuild, btn_clip = _sidebar_settings()

    random.seed(settings.seed)
    np.random.seed(settings.seed)

    full_hash = config_hash(settings)
    ingestion_hash = config_hash(
        settings,
        include=("seed", "max_chunk_chars", "overlap_chars"),
    )
    st.session_state.config_hash = full_hash

    if btn_rebuild:
        st.session_state.pop("index", None)
        st.session_state.pop("demo_clip", None)
        st.session_state.pop("demo_clip_key", None)

    if "index" not in st.session_state or st.session_state.get("ingestion_hash") != ingestion_hash:
        try:
            with st.spinner("Building local ingestion and index pipeline..."):
                index, chunks = build_index_with_progress(DEMO_DOCUMENTS, settings)

            st.session_state.index = index
            st.session_state.chunk_count = len(chunks)
            st.session_state.ingestion_hash = ingestion_hash
            st.session_state.pop("results", None)
            st.session_state.pop("result_fp", None)
            st.session_state.pop("demo_clip", None)
            st.session_state.pop("demo_clip_key", None)
        except Exception as exc:
            st.error(f"[{ERROR_CODES['pipeline_failure']}] Pipeline failed: {exc}")
            log_event("error", code=ERROR_CODES["pipeline_failure"], detail=str(exc))
            st.stop()
    else:
        index = st.session_state.index

    try:
        request = SearchRequest(
            schema_version=SCHEMA_VERSION,
            query=settings.demo_query,
            top_k=settings.top_k_default,
            role=settings.demo_role,
            lexical_weight=settings.lexical_weight,
            dense_weight=settings.dense_weight,
            rrf_k=settings.rrf_k,
        )
    except Exception as exc:
        st.error(f"[{ERROR_CODES['validation_failure']}] Invalid search input: {exc}")
        log_event("error", code=ERROR_CODES["validation_failure"], detail=str(exc))
        st.stop()

    fp = request_fingerprint(request, ingestion_hash)

    if "results" not in st.session_state or st.session_state.get("result_fp") != fp or btn_search:
        try:
            t0 = time.perf_counter()
            results = index.search(request)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            st.session_state.results = results
            st.session_state.latency_ms = latency_ms
            st.session_state.result_fp = fp
            st.session_state.search_request = request

            log_event(
                "search",
                query=request.query,
                role=request.role,
                top_k=request.top_k,
                results=len(results),
                latency_ms=round(latency_ms, 2),
            )
        except Exception as exc:
            st.error(f"[{ERROR_CODES['search_failure']}] Search failed: {exc}")
            log_event("error", code=ERROR_CODES["search_failure"], detail=str(exc))
            results = []
            latency_ms = 0.0
    else:
        results = st.session_state.results
        latency_ms = st.session_state.latency_ms

    stats = summarize_index(index, results, request, latency_ms, full_hash)

    st.title("🧭 Semantic Search and Retrieval Platform — Local Working Demo")
    st.caption(
        f"Schema `{SCHEMA_VERSION}` · Extractor `{EXTRACTOR_VERSION}` · "
        f"Config `{full_hash}` · CPU-only · No API keys"
    )
    st.markdown(
        "This prototype automatically ingests a small synthetic document library, builds a "
        "lexical TF-IDF index and a dense SVD embedding index, applies ACL filtering before "
        "ranking, and generates a downloadable animated demo clip from the current state."
    )

    clip = st.session_state.get("demo_clip")
    clip_key = st.session_state.get("demo_clip_key")

    if btn_clip or (
        settings.auto_animation and (clip is None or clip_key != fp)
    ):
        try:
            with st.spinner("Generating live demo clip..."):
                clip = build_demo_clip(request, results, stats, settings, full_hash)
            st.session_state.demo_clip = clip
            st.session_state.demo_clip_key = fp

            artifact_dir = os.getenv("SSRP_ARTIFACT_DIR")
            if artifact_dir:
                out_path = Path(artifact_dir) / f"demo_clip_{full_hash}.gif"
                atomic_write_bytes(out_path, clip)
                st.session_state.artifact_path = str(out_path)
                log_event("artifact_saved", path=str(out_path))
        except Exception as exc:
            st.warning(
                f"[{ERROR_CODES['artifact_generation_failure']}] "
                f"Could not generate demo clip: {exc}"
            )
            log_event(
                "error",
                code=ERROR_CODES["artifact_generation_failure"],
                detail=str(exc),
            )
            clip = None

    tabs = st.tabs(
        [
            "📊 Overview",
            "🔎 Search results",
            "🧾 Provenance & ACL",
            "🧪 Logs & errors",
            "🎬 Demo clip",
        ]
    )

    with tabs[0]:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Documents", stats["documents"])
        c2.metric("Chunks", stats["chunks"])
        c3.metric("Avg chunk chars", f"{stats['avg_chunk_chars']:.0f}")
        c4.metric("Allowed chunks", stats["allowed_chunks"])
        c5.metric("Returned", stats["returned"])
        c6.metric("Latency", f"{stats['latency_ms']:.1f} ms")

        st.subheader("Document library")
        doc_df = pd.DataFrame(
            [
                {
                    "Title": doc.title,
                    "Doc ID": doc.doc_id,
                    "ACL": ", ".join(doc.acl),
                    "SHA-256": sha256_text(doc.text)[:16],
                    "Source": doc.source_uri or "synthetic",
                }
                for doc in DEMO_DOCUMENTS
            ]
        )
        st.dataframe(doc_df, use_container_width=True)

        left, right = st.columns(2)

        with left:
            st.subheader("Chunks per document")
            if not index.df.empty:
                chunk_counts = (
                    index.df.groupby("title")["chunk_id"]
                    .count()
                    .reset_index(name="chunks")
                )
                st.bar_chart(chunk_counts.set_index("title"))
            else:
                st.info("No chunks available.")

        with right:
            st.subheader("ACL distribution")
            acl_rows: list[str] = []
            for acl in index.df["acl"]:
                acl_rows.extend([str(x) for x in acl])

            acl_series = pd.Series(acl_rows, dtype=str)
            if acl_series.empty:
                st.info("No ACL labels available.")
            else:
                acl_df = (
                    acl_series.value_counts()
                    .rename_axis("acl")
                    .reset_index(name="count")
                )
                st.bar_chart(acl_df.set_index("acl"))

        st.subheader("Current configuration")
        st.json(
            {
                "settings": settings.model_dump(mode="json"),
                "config_hash": full_hash,
                "ingestion_hash": ingestion_hash,
                "error_codes": ERROR_CODES,
            }
        )

    with tabs[1]:
        st.subheader("Ranked evidence")
        st.write(
            f"Query: **{request.query or '(empty)'}** · Role: `{request.role}` · "
            f"Top K: `{request.top_k}` · Lexical weight: `{request.lexical_weight}` · "
            f"Dense weight: `{request.dense_weight}`"
        )

        if results:
            score_df = pd.DataFrame(
                [
                    {
                        "Rank": r.rank,
                        "Title": r.title,
                        "Heading": r.heading,
                        "Lexical": round(r.lexical_score, 4),
                        "Dense": round(r.dense_score, 4),
                        "Fused": round(r.fused_score, 6),
                    }
                    for r in results
                ]
            )
            st.dataframe(score_df, use_container_width=True)

            st.markdown(
                "Explain-why formula: reciprocal rank fusion over lexical and dense rankings."
            )
            st.code(
                f"fused = {request.lexical_weight}/({request.rrf_k}+lexical_rank) + "
                f"{request.dense_weight}/({request.rrf_k}+dense_rank)",
                language="text",
            )

            for r in results:
                with st.expander(
                    f"#{r.rank} — {r.title} · {r.heading} · fused {r.fused_score:.5f}"
                ):
                    st.markdown(r.snippet)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Lexical rank", r.lexical_rank)
                    m2.metric("Dense rank", r.dense_rank)
                    m3.metric("Confidence", f"{r.confidence:.2f}")
                    st.caption(
                        f"Chunk `{r.chunk_id}` | offsets {r.start_offset}-{r.end_offset} | "
                        f"SHA `{r.source_sha256[:12]}` | ACL {', '.join(r.acl)}"
                    )
        else:
            st.info(
                "No permitted results for this query, role, and weight configuration. "
                "Try another role, enable more documents, or change the query."
            )

    with tabs[2]:
        st.subheader("Provenance and access control")
        if results:
            prov_df = pd.DataFrame([r.model_dump(mode="json") for r in results])
            st.dataframe(prov_df, use_container_width=True)

            st.download_button(
                label="Download provenance JSON",
                data=json.dumps(
                    [r.model_dump(mode="json") for r in results],
                    indent=2,
                    default=str,
                ).encode("utf-8"),
                file_name="provenance.json",
                mime="application/json",
            )
        else:
            st.info("No results to export.")

        st.markdown(
            "Every returned chunk carries source document, character span, extractor version, "
            "confidence, ingestion timestamp, SHA-256 fingerprint, and ACL labels."
        )

    with tabs[3]:
        st.subheader("Structured logs")
        logs = st.session_state.get("logs", [])
        if logs:
            logs_df = pd.DataFrame(logs)
            st.dataframe(logs_df, use_container_width=True)
            st.download_button(
                label="Download logs JSON",
                data=json.dumps(logs, indent=2, default=str).encode("utf-8"),
                file_name="app_logs.json",
                mime="application/json",
            )
        else:
            st.info("No logs yet.")

    with tabs[4]:
        st.subheader("Generated live demo clip")
        st.caption(
            "This clip is generated locally from the current app state. It is an animated GIF "
            "container so it can be produced without external video codecs or API services."
        )

        if clip:
            st.download_button(
                label="Download demo clip (animated GIF)",
                data=clip,
                file_name=f"semantic_search_demo_{full_hash}.gif",
                mime="image/gif",
            )

            artifact_path = st.session_state.get("artifact_path")
            if artifact_path:
                st.success(f"Artifact saved to: `{artifact_path}`")

            video_tab, image_tab = st.tabs(["Video player", "Fallback image"])

            with video_tab:
                try:
                    st.video(
                        clip,
                        format="image/gif",
                        loop=True,
                        autoplay=True,
                    )
                except Exception as exc:
                    st.warning(f"Video player fallback used: {exc}")
                    st.image(clip, use_container_width=True)

            with image_tab:
                st.image(clip, use_container_width=True)
        else:
            st.warning("Demo clip was not generated. Enable auto-generation or press Regenerate.")


if __name__ == "__main__":
    main()