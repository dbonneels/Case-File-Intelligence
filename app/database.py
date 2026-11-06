"""SQLite helpers for claim metadata and validation findings."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DB_PATH = Path("storage/case_intelligence.db")


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_schema() -> None:
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS claims (
                claim_id TEXT PRIMARY KEY,
                claim_type TEXT,
                data_path TEXT,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id TEXT,
                file_name TEXT,
                doc_type TEXT,
                page_count INTEGER,
                char_count INTEGER,
                metadata_json TEXT
            );
            CREATE TABLE IF NOT EXISTS document_mappings (
                claim_id TEXT,
                file_name TEXT,
                canonical_type TEXT
            );
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                claim_id TEXT,
                file_name TEXT,
                doc_type TEXT,
                page_start INTEGER,
                page_end INTEGER,
                text TEXT
            );
            CREATE TABLE IF NOT EXISTS findings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id TEXT,
                category TEXT,
                severity TEXT,
                field TEXT,
                summary TEXT,
                details_json TEXT
            );
            """
        )
        connection.commit()


def replace_documents(claim_id: str, rows: list[dict]) -> None:
    with get_connection() as connection:
        connection.execute("DELETE FROM documents WHERE claim_id = ?", (claim_id,))
        for row in rows:
            connection.execute(
                """
                INSERT INTO documents
                (claim_id, file_name, doc_type, page_count, char_count, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    claim_id,
                    row["file_name"],
                    row["doc_type"],
                    row["page_count"],
                    row["char_count"],
                    json.dumps(row.get("metadata", {}), ensure_ascii=False),
                ),
            )
        connection.commit()


def replace_mappings(claim_id: str, rows: list[dict]) -> None:
    with get_connection() as connection:
        connection.execute("DELETE FROM document_mappings WHERE claim_id = ?", (claim_id,))
        for row in rows:
            connection.execute(
                "INSERT INTO document_mappings (claim_id, file_name, canonical_type) VALUES (?, ?, ?)",
                (claim_id, row["file_name"], row["canonical_type"]),
            )
        connection.commit()


def replace_chunks(claim_id: str, rows: list[dict]) -> None:
    with get_connection() as connection:
        connection.execute("DELETE FROM chunks WHERE claim_id = ?", (claim_id,))
        for row in rows:
            connection.execute(
                """
                INSERT INTO chunks (chunk_id, claim_id, file_name, doc_type, page_start, page_end, text)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["chunk_id"],
                    claim_id,
                    row["file_name"],
                    row["doc_type"],
                    row.get("page_start"),
                    row.get("page_end"),
                    row["text"],
                ),
            )
        connection.commit()


def replace_findings(claim_id: str, rows: list[dict]) -> None:
    with get_connection() as connection:
        connection.execute("DELETE FROM findings WHERE claim_id = ?", (claim_id,))
        for row in rows:
            connection.execute(
                """
                INSERT INTO findings (claim_id, category, severity, field, summary, details_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    claim_id,
                    row["category"],
                    row["severity"],
                    row["field"],
                    row["summary"],
                    json.dumps(row.get("details", {}), ensure_ascii=False),
                ),
            )
        connection.commit()


def upsert_claim(claim_id: str, claim_type: str, data_path: str) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO claims (claim_id, claim_type, data_path, created_at)
            VALUES (?, ?, ?, datetime('now'))
            ON CONFLICT(claim_id) DO UPDATE SET
                claim_type=excluded.claim_type,
                data_path=excluded.data_path
            """,
            (claim_id, claim_type, data_path),
        )
        connection.commit()


def get_documents(claim_id: str) -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT file_name, doc_type, page_count, metadata_json FROM documents WHERE claim_id = ?",
            (claim_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_mappings(claim_id: str) -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT file_name, canonical_type FROM document_mappings WHERE claim_id = ?",
            (claim_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_findings(claim_id: str) -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT category, severity, field, summary, details_json FROM findings WHERE claim_id = ?",
            (claim_id,),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["details"] = json.loads(item.pop("details_json") or "{}")
        result.append(item)
    return result


def get_chunks(claim_id: str) -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT chunk_id, file_name, doc_type, page_start, page_end, text FROM chunks WHERE claim_id = ?",
            (claim_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_claims() -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute("SELECT claim_id, claim_type, data_path FROM claims").fetchall()
    return [dict(row) for row in rows]