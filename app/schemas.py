"""Pydantic schemas for the FastAPI endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    k: int = Field(6, ge=1, le=20)


class Citation(BaseModel):
    file_name: str
    page: int | None = None
    score: float = 0.0


class AskResponse(BaseModel):
    claim_id: str
    question: str
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[dict[str, Any]]


class ValidationResponse(BaseModel):
    claim_id: str
    case_state: str
    missing_documents: list[str]
    inconsistencies: list[dict[str, Any]]
    finding_count: int