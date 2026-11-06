from ingestion.classify_documents import classify_document


def test_water_claim_files() -> None:
    assert classify_document("04_rapport_expertise.pdf", "rapport d'expertise") == "expert_report"
    assert classify_document("05_devis_reparation_alpha.pdf", "devis") == "repair_quote"
    assert classify_document("08_echanges_courriels.pdf", "échanges de courriels") == "correspondence"


def test_auto_claim_files() -> None:
    assert classify_document("04_constat_amiable_reconstitue.pdf", "constat") == "accident_report"
    assert classify_document("08_assistance_remorquage.pdf", "remorquage") == "assistance"
