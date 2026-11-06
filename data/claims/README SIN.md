# Jeu de données synthétique — SIN-2026-0042

Ce jeu contient 9 documents PDF en français et un manifest JSON.

## Contenu
1. Déclaration de sinistre
2. Conditions particulières du contrat
3. Relevé de cotisations
4. Rapport d'expertise
5. Devis Bâti-Tours
6. Devis Bâti Plus Centre
7. Fiche d'intervention du plombier
8. Échanges de courriels reconstitués
9. Note interne d'instruction

## Anomalies intentionnelles à détecter
- Date du sinistre : 12/10/2026 dans la déclaration et la fiche plombier, 13/10/2026 dans le rapport d'expertise.
- Échéance de cotisation du 01/07/2026 marquée impayée. Cela ne permet pas de conclure à une suspension ou résiliation.
- Montants différents : 8 450 €, 6 300 €, 4 301 € et 6 875 € TTC.
- Facture finale de réparation absente.
- Conditions générales complètes absentes.

## Utilisation RAG
- Parser les PDF avec Docling.
- Conserver le nom du fichier, le numéro de page et le texte source.
- Classifier les documents par rôle.
- Extraire les dates, montants, références de contrat et identités dans un schéma structuré.
- Faire les contrôles déterministes en Python/SQL.
- Utiliser FAISS pour retrouver les passages pertinents.
- Générer les réponses avec citations document + page.

## Important
Tous les noms, adresses, références, montants et échanges sont inventés.
Les domaines example.invalid sont réservés et non distribuables.
Ce dataset est un support technique de démonstration, pas un contrat réel ni un avis juridique.
