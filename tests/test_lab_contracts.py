import hashlib
import json
import re
import unicodedata
from collections import deque
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb"


def cell_source(cell_id):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "".join(next(c for c in notebook["cells"] if c.get("id") == cell_id)["source"])


def base_namespace():
    def norm_space(value):
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        return re.sub(r"\s+", " ", str(value)).strip()

    def sha1(value):
        return hashlib.sha1(str(value).encode("utf-8", errors="ignore")).hexdigest()

    return {
        "np": np,
        "pd": pd,
        "json": json,
        "re": re,
        "unicodedata": unicodedata,
        "SequenceMatcher": SequenceMatcher,
        "tqdm": tqdm,
        "deque": deque,
        "norm_space": norm_space,
        "sha1": sha1,
        "SEED": 42,
    }


def test_near_dedup_uses_lsh_and_merges_repost_only():
    ns = base_namespace()
    exec(cell_source("near-dedup-minhash-lsh"), ns)
    base_tokens = [f"token{i}" for i in range(260)]
    repost_tokens = base_tokens[:120] + ["small", "verified", "update"] + base_tokens[120:]
    boilerplate = " ".join(f"common{i}" for i in range(45))
    frame = pd.DataFrame({
        "article_id": ["base", "repost", "unrelated-a", "unrelated-b"],
        "title": ["Base", "Repost", "A", "B"],
        "published_date": ["2026-01-01"] * 4,
        "text": [
            " ".join(base_tokens),
            " ".join(repost_tokens),
            boilerplate + " " + " ".join(f"alpha{i}" for i in range(220)),
            boilerplate + " " + " ".join(f"beta{i}" for i in range(220)),
        ],
    })
    deduped, audit = ns["near_deduplicate_minhash_lsh"](frame)
    assert len(deduped) == 3
    assert len(audit) == 1
    assert audit.iloc[0].jaccard_5gram >= 0.82


def test_extraction_contract_rejects_wrong_direction_and_hallucinated_evidence():
    ns = base_namespace()
    exec(cell_source("cb79b195"), ns)
    meta = {"published_date": "2026-01-02", "text": "Microsoft acquired GitHub in 2018."}
    valid = {
        "source": "Microsoft", "source_type": "Company", "relation": "ACQUIRED",
        "target": "GitHub", "target_type": "Company",
        "evidence": "Microsoft acquired GitHub in 2018.", "confidence": 0.95,
    }
    triple, reasons = ns["_validate_relation_candidate"](valid, "c1", meta, 0.60)
    assert triple is not None and reasons == []

    wrong = dict(valid, relation="WORKED_AT", target_type="Technology")
    _, reasons = ns["_validate_relation_candidate"](wrong, "c1", meta, 0.60)
    assert "INVALID_RELATION_DIRECTION_OR_ENDPOINT_TYPES" in reasons

    hallucinated = dict(valid, evidence="Microsoft bought an unrelated company.")
    _, reasons = ns["_validate_relation_candidate"](hallucinated, "c1", meta, 0.60)
    assert "EVIDENCE_NOT_VERBATIM_IN_CHUNK" in reasons

    generic_meta = {"published_date": "2026-01-02", "text": "The company uses Git."}
    generic = dict(
        valid,
        source="the company",
        relation="USES",
        target="Git",
        target_type="Technology",
        evidence="The company uses Git.",
    )
    _, reasons = ns["_validate_relation_candidate"](generic, "c2", generic_meta, 0.60)
    assert "GENERIC_UNRESOLVED_ENTITY" in reasons

    planned_meta = {"published_date": "2026-01-02", "text": "Microsoft plans to acquire GitHub."}
    planned = dict(valid, evidence="Microsoft plans to acquire GitHub.")
    _, reasons = ns["_validate_relation_candidate"](planned, "c3", planned_meta, 0.60)
    assert "NON_COMPLETED_ACQUISITION" in reasons


def test_entity_guard_blocks_ticker_person_and_product_false_merges():
    ns = base_namespace()
    ns["ALLOWED_NODE_TYPES"] = {"Company", "Person", "Technology"}
    exec(cell_source("223090d5"), ns)
    assert ns["merge_guard"]("Microsoft Inc.", "Microsoft Corporation", "Company")[0]
    assert not ns["merge_guard"]("IBM", "IBN", "Company")[0]
    assert not ns["merge_guard"]("Sam Altman", "Steve Altman", "Person")[0]
    assert not ns["merge_guard"]("Apple", "Apple Music", "Technology")[0]


def test_supernode_policy_caps_edge_fetch_at_50():
    ns = base_namespace()
    exec(cell_source("4cba2582"), ns)
    calls = []
    ns["match_seeds"] = lambda query: [{"id": "super", "name": "Super", "type": "Company"}]
    ns["node_degree"] = lambda node_id: 150

    def recent_edges(node_id, limit):
        calls.append(limit)
        return [{
            "source_id": "super", "source_name": "Super", "source_type": "Company",
            "relation": "USES", "target_id": f"t{i}", "target_name": f"T{i}",
            "target_type": "Technology", "source_chunk_id": f"c{i}",
            "published_date": "2026-01-01", "evidence": "evidence", "neighbor_id": f"t{i}",
        } for i in range(limit)]

    ns["recent_edges"] = recent_edges
    result = ns["retrieve_graph_context"]("query", return_debug=True)
    assert calls and all(limit == 50 for limit in calls)
    assert len(result["edges"]) <= ns["GLOBAL_EDGE_CAP"]
    assert result["diagnostics"]["supernode_events"][0]["limit"] == 50


def test_notebook_uses_local_output_paths_and_unwind_allowlisted_queries():
    source = NOTEBOOK.read_text(encoding="utf-8")
    assert 'OUTPUT_DIR / \\"graphrag_eval_results.csv\\"' in source
    assert "UNWIND $rows AS row" in source
    assert "for rel in sorted(ALLOWED_RELATIONS)" in source
    assert 'GOLDEN_PATH = DATA_DIR / \\"graphrag_golden_50_first5000.csv\\"' in source
