"""FastAPI application for insurance claim RAG and validation."""


import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from groq import Groq
from pydantic import BaseModel, Field

from app.database import get_chunks, get_documents, get_findings, init_schema, get_claims
from app.schemas import AskRequest, AskResponse, Citation
from ingestion.classify_documents import load_manifest
from rag.retrieval import search
from validation.check_inconsistencies import find_inconsistencies
from validation.check_missing_documents import find_missing_documents

load_dotenv()
DATA_ROOT = Path("data/claims")
GROQ_MODEL = "qwen/qwen3.8-27b"
groq_client = None


def get_groq_client() -> Groq:
    global groq_client
    if groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured")
        groq_client = Groq(api_key=api_key)
    return groq_client



def _normalize_document(row: dict) -> dict:
    item = dict(row)
    raw = item.pop("metadata_json", None)
    try:
        item["metadata"] = json.loads(raw) if raw else {}
    except (TypeError, json.JSONDecodeError):
        item["metadata"] = {}
    return item


def claim_dir(claim_id: str) -> Path:
    path = DATA_ROOT / claim_id
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found")
    return path


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_schema()
    yield


app = FastAPI(
    title="Case-File Intelligence",
    description="RAG et validation de dossiers d'assurance synthétiques.",
    version="0.1.0",
    lifespan=lifespan,
)


def manifest_for(claim_id: str) -> dict:
    return load_manifest(claim_dir(claim_id))


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/case-file-intelligence-demo/ui")

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ui", response_class=HTMLResponse)
def ui() -> str:
    return """<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>Case-File Intelligence</title>
  <style>
    :root {
      --color-start: #0f172a;
      --color-mid: #1e3a8a;
      --color-end: #0f766e;
      --card-bg: rgba(255, 255, 255, 0.92);
    }
    body {
      font-family: "Segoe UI", system-ui, sans-serif;
      max-width: 1080px;
      margin: 0 auto;
      padding: 2rem 1rem 3rem;
      color: #0f172a;
      background: linear-gradient(135deg, var(--color-start) 0%, var(--color-mid) 45%, var(--color-end) 100%);
      min-height: 100vh;
    }
    h1 {
      color: #ffffff;
      text-shadow: 0 2px 12px rgba(0, 0, 0, 0.35);
      margin-bottom: 1.5rem;
    }
    textarea, select {
      width: 100%;
      max-width: 100%;
      box-sizing: border-box;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      padding: 0.75rem;
      font-family: inherit;
      font-size: 1rem;
    }
    textarea {
      width: 100%;
      min-height: 110px;
      resize: vertical;
    }
    button {
      border: none;
      border-radius: 8px;
      padding: 0.7rem 1.1rem;
      margin-right: 0.5rem;
      margin-top: 0.75rem;
      cursor: pointer;
      font-weight: 600;
      color: #ffffff;
      background: linear-gradient(135deg, #2563eb, #0891b2);
      box-shadow: 0 6px 16px rgba(15, 23, 42, 0.25);
    }
    button:hover {
      filter: brightness(1.08);
    }
    .card {
      border: none;
      border-radius: 16px;
      padding: 1.25rem;
      margin-bottom: 1rem;
      background: var(--card-bg);
      box-shadow: 0 18px 45px rgba(15, 23, 42, 0.22);
      backdrop-filter: blur(8px);
    }
    .card h2 {
      margin-top: 0;
      color: #1e3a8a;
    }
    pre {
      background: #0f172a;
      color: #e2e8f0;
      padding: 1rem;
      border-radius: 10px;
      white-space: pre-wrap;
      overflow-x: auto;
    }
    .anomaly-category {
      font-style: italic;
      color: #60a5fa;
      font-size: smaller;
    }
    .doc-status {
      display: inline-block;
      margin-left: 0.5rem;
      padding: 0.15rem 0.5rem;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 600;
    }
    .doc-status.clear { background: #dcfce7; color: #166534; }
    .doc-status.ambiguous { background: #fef3c7; color: #92400e; }
    .doc-status.unknown { background: #e5e7eb; color: #374151; }
    .doc-alternatives { color: #92400e; font-size: 0.8rem; }
  </style>
</head>
<body>
  <h1>Case-File Intelligence</h1>
  <div class="card">
    <div style="display:flex;align-items:center;gap:0.75rem;flex-wrap:wrap;margin-bottom:0.75rem;">
      <label style="flex:0 0 auto;margin:0;">Dossier
        <select id="claim" style="width:auto;min-width:200px;max-width:300px;"></select>
      </label>
      <button id="validate-btn" onclick="toggleValidation()" style="margin:0;">Vérifier le dossier</button>
    </div>
    <p><textarea id="question" placeholder="Question..."></textarea></p>
    <button onclick="ask()">Demander</button>
  </div>
  <div class="card" id="response-card" style="display:none;"><h2>Réponse</h2><pre id="answer"></pre></div>
  <div class="card" id="citations-card" style="display:none;"><h2>Citations</h2><pre id="citations"></pre></div>
  <div class="card" id="validation-card" style="display:none;"><h2>Validation</h2><div id="validation-summary" style="white-space:pre-wrap;margin-bottom:0.75rem;"></div><button id="details-btn" onclick="toggleValidationDetails()">Afficher les détails</button><pre id="validation" style="display:none;"></pre></div>
  <div class="card" id="documents-card">
    <h2>Documents du dossier</h2>
    <div id="documents" style="white-space:pre-wrap;"></div>
  </div>
  <div class="card" id="anomalies-card">
    <h2>Anomalies intentionnelles à détecter</h2>
    <p>Cette démonstration détecte et signale notamment les anomalies suivantes :</p>
    <div id="anomalies-SIN-2026-0042" class="anomaly-block" style="display:none;">
      <h3>Dossier SIN-2026-0042 (dégât des eaux)</h3>
      <ul>
        <li>Conflit de date de sinistre : 12/10/2026 dans la déclaration et la fiche plombier, 13/10/2026 dans le rapport d'expertise (<span class="anomaly-category">date_conflict</span>).</li>
        <li>Échéance de cotisation du 01/07/2026 marquée impayée dans le relevé de cotisations (<span class="anomaly-category">premium_status</span>).</li>
        <li>Écarts de montants TTC : 8 450 €, 6 300 €, 4 301 € et 6 875 € (<span class="anomaly-category">amount_conflict</span>).</li>
        <li>Pièce manquante : facture finale de réparation (final_repair_invoice) (<span class="anomaly-category">missing_documents</span>).</li>
      <li>Pièce manquante : conditions générales complètes (general_policy_conditions) (<span class="anomaly-category">missing_documents</span>).</li>
      </ul>
    </div>
    <div id="anomalies-AUTO-2026-0087" class="anomaly-block" style="display:none;">
      <h3>Dossier AUTO-2026-0087 (accident automobile)</h3>
      <ul>
        <li>Pièce manquante : constat amiable officiel signé (constat_amiable_officiel_signe) (<span class="anomaly-category">missing_documents</span>).</li>
        <li>Pièce manquante : photographies originales (photographies_originales) (<span class="anomaly-category">missing_documents</span>).</li>
        <li>Pièce manquante : conditions générales complètes (conditions_generales_CG-AUTO-2025-v2) (<span class="anomaly-category">missing_documents</span>).</li>
        <li>Pièce manquante : ordre de réparation et facture finale (ordre_reparation_et_facture_finale) (<span class="anomaly-category">missing_documents</span>).</li>
        <li>Versions divergentes des conducteurs sur la priorité et l'engagement du véhicule B (<span class="anomaly-category">statement_conflict</span>).</li>
        <li>Montants TTC divergents : 6 800 €, 6 600 € (provisoire) et 7 420 € (<span class="anomaly-category">amount_conflict</span>).</li>
        <li>Périmètres de réparation potentiellement différents : expertise provisoire et devis incluant des calibrages/contrôles complémentaires (<span class="anomaly-category">repair_scope</span>).</li>
      </ul>

    </div>
    <p id="anomaly-categories">Catégories d'anomalies détectées : <span class="anomaly-category">missing_documents</span>, <span class="anomaly-category">date_conflict</span>, <span class="anomaly-category">premium_status</span>, <span class="anomaly-category">amount_conflict</span>, <span class="anomaly-category">statement_conflict</span>, <span class="anomaly-category">repair_scope</span>.</p>
  </div>
  <script>
    async function loadClaims() {
      const response = await fetch('/claims');
      const claims = await response.json();
      const select = document.getElementById('claim');
      const claimLabels = {
        'AUTO-2026-0087': 'Sinistre auto AUTO-2026-0087',
        'SIN-2026-0042': 'Sinistre maison SIN-2026-0042'
      };
      claims.forEach(claim => {
        const option = document.createElement('option');
        option.value = claim.claim_id;
        option.textContent = claimLabels[claim.claim_id] || claim.claim_id;
        select.appendChild(option);
      });
      select.addEventListener('change', () => { updateAnomalies(); loadDocuments(); });
      updateAnomalies();
      loadDocuments();
    }
    function updateAnomalies() {
      const claimId = document.getElementById('claim').value;
      document.querySelectorAll('.anomaly-block').forEach(block => {
        block.style.display = block.id === 'anomalies-' + claimId ? 'block' : 'none';
      });
    }
    async function loadDocuments() {
      const claimId = document.getElementById('claim').value;
      const response = await fetch(`/claims/${claimId}/documents`);
      if (!response.ok) {
        document.getElementById('documents').textContent = 'Documents non indexés.';
        return;
      }
      const docs = await response.json();
      const lines = docs.map(doc => {
        const cls = (doc.metadata && doc.metadata.classification) || {};
        const status = cls.classification_status || 'clear';
        const badge = `<span class="doc-status ${status}">${status}</span>`;
        const alternatives = Array.isArray(cls.alternative_types) && cls.alternative_types.length
          ? ` <span class="doc-alternatives">(alternatives : ${cls.alternative_types.join(', ')})</span>`
          : '';
        return `<div>${doc.file_name} — <strong>${doc.doc_type}</strong>${badge}${alternatives}</div>`;
      });
      document.getElementById('documents').innerHTML = lines.join('') || 'Aucun document.';
    }
    async function ask() {
      const claimId = document.getElementById('claim').value;
      const question = document.getElementById('question').value;
      const response = await fetch(`/claims/${claimId}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, k: 6 })
      });
      const data = await response.json();
      document.getElementById('answer').textContent = data.answer || JSON.stringify(data, null, 2);
      document.getElementById('citations').textContent = JSON.stringify(data.citations || [], null, 2);
       document.getElementById('response-card').style.display = 'block';
       document.getElementById('citations-card').style.display = 'block';
    }
    function toggleValidationDetails() {
      const pre = document.getElementById('validation');
      const btn = document.getElementById('details-btn');
      if (pre.style.display === 'block') {
        pre.style.display = 'none';
        btn.textContent = 'Afficher les détails';
      } else {
        pre.style.display = 'block';
        btn.textContent = 'Masquer les détails';
      }
    }
    function toggleValidation() {
      const card = document.getElementById('validation-card');
      const btn = document.getElementById('validate-btn');
      if (card.style.display === 'block') {
        card.style.display = 'none';
        btn.textContent = 'Vérifier le dossier';
      } else {
        validate();
      }
    }
    async function validate() {
      const claimId = document.getElementById('claim').value;
      const response = await fetch(`/claims/${claimId}/validate`);
      const data = await response.json();
      const lines = [];
      lines.push('État du dossier : ' + (data.case_state || 'UNKNOWN'));
      if (Array.isArray(data.missing_documents) && data.missing_documents.length) {
        data.missing_documents.forEach(item => lines.push('Pièce manquante : ' + item));
      } else {
        lines.push('Pièce manquante : aucune');
      }
      if (Array.isArray(data.inconsistencies) && data.inconsistencies.length) {
        lines.push('Anomalies détectées :');
        data.inconsistencies.forEach(item => lines.push(' - ' + (item.summary || item.category)));
      } else {
        lines.push('Anomalies détectées : aucune');
      }
      document.getElementById('validation-summary').textContent = lines.join('\\n');
      document.getElementById('validation').textContent = JSON.stringify(data, null, 2);
       document.getElementById('validation-card').style.display = 'block';
       document.getElementById('validate-btn').textContent = 'Masquer la validation';
    }
    loadClaims();
  </script>
</body>
</html>"""


@app.get("/claims")
def list_claims() -> list[dict]:
    rows = get_claims()
    if not rows:
        # Fallback to filesystem if not ingested yet.
        rows = [{"claim_id": p.name, "claim_type": "unknown", "data_path": str(p)} for p in sorted(DATA_ROOT.iterdir()) if p.is_dir()]
    return rows


@app.get("/claims/{claim_id}")
def get_claim(claim_id: str) -> dict:
    manifest = manifest_for(claim_id)
    docs = [_normalize_document(row) for row in get_documents(claim_id)]
    return {"claim_id": claim_id, "manifest": manifest, "documents": docs}


@app.get("/claims/{claim_id}/documents")
def list_documents(claim_id: str) -> list[dict]:
    claim_dir(claim_id)
    docs = [_normalize_document(row) for row in get_documents(claim_id)]
    if not docs:
        raise HTTPException(status_code=404, detail="Documents not indexed yet")
    return docs


@app.get("/claims/{claim_id}/validate")
def validate_claim(claim_id: str) -> dict:
    directory = claim_dir(claim_id)
    manifest = manifest_for(claim_id)
    missing = find_missing_documents(claim_id, directory, manifest)
    inconsistencies = find_inconsistencies(claim_id, directory, manifest)
    state = manifest.get("recommended_case_state") or manifest.get("recommended_state") or "UNKNOWN"
    return {
        "claim_id": claim_id,
        "case_state": state,
        "missing_documents": missing["missing"],
        "missing_details": missing,
        "inconsistencies": inconsistencies,
        "finding_count": len(inconsistencies) + (1 if missing["missing"] else 0),
    }


@app.post("/claims/{claim_id}/ask")
def ask_claim(claim_id: str, request: AskRequest) -> AskResponse:
    claim_dir(claim_id)
    manifest = manifest_for(claim_id)
    retrieved = search(request.question, claim_id=claim_id, k=request.k)
    validation = find_missing_documents(claim_id, claim_dir(claim_id), manifest)
    inconsistencies = find_inconsistencies(claim_id, claim_dir(claim_id), manifest)

    context_blocks: list[str] = []
    for finding in inconsistencies:
        context_blocks.append(
            "[Contrôle déterministe] "
            + finding["summary"]
            + " "
            + str(finding.get("details", {}))
        )
    if validation["missing"]:
        context_blocks.append(
            "[Contrôle déterministe] Pièces manquantes : " + ", ".join(validation["missing"])
        )

    for chunk in retrieved:
        page = chunk.get("page_start")
        page_text = f"p.{page}" if page is not None else "p.?"
        context_blocks.append(
            f"[Source: {chunk['file_name']}, {page_text}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_blocks)
    system_prompt = (
        "Tu es un assistant d'analyse de dossiers d'assurance. "
        "Réponds uniquement à partir du contexte fourni. "
        "Cite toujours la source sous la forme (fichier.pdf, p.X). "
        "Si une information est absente, dis exactement qu'elle n'est pas présente dans les documents. "
        "N'invente rien et ne détermine pas de responsabilité."
    )
    user_prompt = f"Contexte:\n{context}\n\nQuestion: {request.question}"

    try:
        completion = get_groq_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=900,
        )
        answer = completion.choices[0].message.content or ""
    except Exception as exc:  # pragma: no cover - depends on network/runtime
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc

    citations: list[Citation] = []
    seen: set[tuple[str, int | None]] = set()
    for chunk in retrieved:
        key = (chunk["file_name"], chunk.get("page_start"))
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(
                file_name=chunk["file_name"],
                page=chunk.get("page_start"),
                score=float(chunk.get("score", 0.0)),
            )
        )

    return AskResponse(
        claim_id=claim_id,
        question=request.question,
        answer=answer,
        citations=citations,
        retrieved_chunks=retrieved,
    )


@app.get("/claims/{claim_id}/dashboard")
def dashboard(claim_id: str) -> dict:
    directory = claim_dir(claim_id)
    manifest = manifest_for(claim_id)
    docs = get_documents(claim_id)
    chunks = get_chunks(claim_id)
    missing = find_missing_documents(claim_id, directory, manifest)
    inconsistencies = find_inconsistencies(claim_id, directory, manifest)
    return {
        "claim_id": claim_id,
        "claim_type": manifest.get("claim_type"),
        "document_count": len(docs),
        "chunk_count": len(chunks),
        "missing_count": len(missing["missing"]),
        "inconsistency_count": len(inconsistencies),
        "case_state": manifest.get("recommended_case_state") or manifest.get("recommended_state"),
    }