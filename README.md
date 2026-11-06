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
- Page HTML minimale servie par FastAPI.

## Avertissement
Données entièrement fictives. Aucune valeur contractuelle, juridique ou probante.