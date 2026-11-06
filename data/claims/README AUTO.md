# Dataset synthétique — AUTO-2026-0087

Dossier fictif d'accident automobile avec dommages matériels uniquement : 10 PDF en français et un manifest JSON.

## Documents
01 Déclaration d'accident ; 02 Conditions particulières ; 03 Attestation de simulation ; 04 Résumé reconstitué du constat (non officiel/non signé) ; 05 Compte rendu conducteur ; 06 Rapport d'expertise provisoire ; 07 Devis garage ; 08 Assistance/remorquage ; 09 Courriels reconstitués ; 10 Note interne.

## Pièces manquantes et anomalies
- Constat amiable officiel signé par les deux conducteurs absent.
- Photographies originales mentionnées mais non fournies.
- Conditions générales complètes absentes.
- Ordre de réparation et facture finale absents.
- Versions divergentes sur la priorité et l'engagement du véhicule B.
- Montants TTC : assuré 6 800 €, expert 6 600 € (provisoire), garage 7 420 €.
- Périmètres des travaux potentiellement différents.

## Tests RAG
1. Inventorier les pièces reçues et manquantes.
2. Extraire date, heure, lieu, conducteurs, véhicules et contrat.
3. Comparer les récits sans décider de la responsabilité.
4. Comparer les montants et expliquer les différences de périmètre.
5. Répondre avec nom du fichier et numéro de page.
6. Signaler les informations insuffisantes plutôt que d'inventer une conclusion.

Pipeline suggéré : Docling -> classification -> extraction structurée -> contrôles Python/SQL -> embeddings -> FAISS -> LLM avec citations.

Avertissement : jeu entièrement inventé, sans valeur officielle, contractuelle ou probante. Aucun avis juridique.
