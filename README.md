# Case-File Intelligence — démonstration RAG assurance

Démonstration locale combinant **Docling**, **FAISS** et **FastAPI** pour analyser des dossiers d'assurance synthétiques (dégât des eaux et accident automobile).

## Démarrage

```bash
cd /home/dominique/Case-File-Intelligence
uv sync --group dev
cp .env.example .env  # renseigner GROQ_API_KEY si besoin
uv run python -m ingestion.build_index
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Interface minimale : http://127.0.0.1:8000/ui  
Documentation API : http://127.0.0.1:8000/docs

## Tests

```bash
uv run pytest -q
```

## Fonctionnalités démontrées
- Parsing PDF avec Docling et conservation du numéro de page.
- Classification des documents par type.
- Extraction de métadonnées (dates, montants, références, identité).
- Vérification des pièces obligatoires manquantes.
- Détection déterministe des incohérences (dates, montants, cotisation, versions des conducteurs).
- Recherche vectorielle FAISS avec embeddings BGE-M3.
- Réponses LLM avec citations `fichier, page`.
- Interface HTML interactive servie par FastAPI : question/réponse, citations, validation avec synthèse et détails JSON masqués par défaut, anomalies propres au dossier sélectionné.

## États du dossier

L'application calcule un état déterministe du dossier :

- `COMPLETE` — aucune pièce manquante, aucune incohérence et aucune classification ambiguë.
- `INCOMPLETE` — des pièces manquantes, sans incohérence ni classification ambiguë.
- `CONFLICTING` — des incohérences détectées, sans pièce manquante ni classification ambiguë.
- `INCOMPLETE_AND_CONFLICTING` — à la fois des pièces manquantes et des incohérences.
- `REVIEW_REQUIRED` — au moins un document a une classification ambiguë.
- `UNKNOWN` — aucun élément fourni pour déterminer l'état.

Le badge affiché dans l'interface comporte une info-bulle au survol pour expliquer l'état courant.

## Avertissement
Données entièrement fictives. Aucune valeur contractuelle, juridique ou probante.