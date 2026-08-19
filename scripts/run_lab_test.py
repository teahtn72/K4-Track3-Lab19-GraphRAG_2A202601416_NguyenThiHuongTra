#!/usr/bin/env python3
"""Run a reproducible, evidence-covering Lab 19 integration benchmark.

The script executes the implementation cells from the submitted notebook so the
test exercises the deliverable itself. It never uses per-question gold row IDs
during retrieval; the union of all gold evidence rows is only used once to make
sure the controlled 400-document corpus contains answerable questions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from IPython.display import display


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb"
DATASET = ROOT / "hackernoon_subset.csv"
GOLDEN = ROOT / "data" / "graphrag_golden_50_first5000.csv"
GOLDEN_DETAILED = ROOT / "data" / "graphrag_golden_50_first5000_detailed.csv"
OUTPUT = ROOT / "outputs"

CELL_IDS = [
    "06814082",  # imports/config
    "2c8a6124",  # Neo4j/schema
    "adb244b9",  # loader/exact dedup/chunking
    "near-dedup-minhash-lsh",
    "863fddb0",  # LLM wrapper
    "7a778170",  # coreference
    "cb79b195",  # extraction/validation
    "223090d5",  # entity resolution
    "c040c675",  # bulk insert
    "8a3e0b51",  # graph checks
    "6479a2e3",  # flat index
    "e1238321",  # seed matching
    "4cba2582",  # graph traversal
    "ac291ee5",  # answer generation
    "12e252b2",  # golden validation
    "6245ee0c",  # judge
    "fd02f02c",  # comparison
    "f1bb87ea",  # failure checks
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-size", type=int, default=400)
    parser.add_argument("--eval-size", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--pipeline-provider", choices=["groq", "openai"], default="openai")
    parser.add_argument("--pipeline-model", default="gpt-4o-mini")
    parser.add_argument("--skip-coref", action="store_true")
    parser.add_argument("--skip-neo4j", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def load_notebook_namespace() -> dict:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    by_id = {cell.get("id"): cell for cell in notebook["cells"]}
    missing = [cell_id for cell_id in CELL_IDS if cell_id not in by_id]
    if missing:
        raise RuntimeError(f"Notebook cell IDs not found: {missing}")

    namespace = {"__name__": "__lab_notebook__", "display": display}
    for cell_id in CELL_IDS:
        source = "".join(by_id[cell_id].get("source", []))
        exec(compile(source, f"{NOTEBOOK.name}:{cell_id}", "exec"), namespace)
    return namespace


def gold_evidence_ids(detailed: pd.DataFrame) -> set[int]:
    result: set[int] = set()
    for value in detailed["evidence_row_ids_0based"]:
        result.update(int(x) for x in json.loads(value))
    return result


def build_controlled_corpus(
    raw_first_5000: pd.DataFrame,
    detailed: pd.DataFrame,
    corpus_size: int,
    seed: int,
) -> tuple[pd.DataFrame, list[int]]:
    evidence_ids = gold_evidence_ids(detailed)
    if corpus_size < len(evidence_ids):
        raise ValueError(f"corpus_size must be >= {len(evidence_ids)} evidence rows")

    descriptions = raw_first_5000["description"].fillna("").astype(str)
    eligible = set(raw_first_5000.index[descriptions.str.len() >= 80].astype(int))
    missing = evidence_ids - eligible
    if missing:
        raise ValueError(f"Gold evidence rows missing usable text: {sorted(missing)}")

    distractor_pool = sorted(eligible - evidence_ids)
    rng = __import__("numpy").random.default_rng(seed)
    need = min(corpus_size - len(evidence_ids), len(distractor_pool))
    distractors = set(int(x) for x in rng.choice(distractor_pool, size=need, replace=False))
    selected_ids = sorted(evidence_ids | distractors)

    corpus = raw_first_5000.loc[selected_ids].copy()
    corpus.insert(0, "id", [f"row_{idx:05d}" for idx in selected_ids])
    return corpus, selected_ids


def stratified_eval_ids(golden: pd.DataFrame, eval_size: int) -> list[str]:
    preferred = [
        "G5000-03", "G5000-13", "G5000-24", "G5000-41", "G5000-47",
        "G5000-01", "G5000-06", "G5000-14", "G5000-31", "G5000-48",
        "G5000-02", "G5000-15", "G5000-25", "G5000-36", "G5000-43",
    ]
    available = set(golden["id"])
    selected = [item for item in preferred if item in available][:eval_size]
    if len(selected) < eval_size:
        selected += [x for x in golden["id"] if x not in selected][: eval_size - len(selected)]
    return selected


def expected_chunk_ids(
    question_id: str,
    detailed_by_id: dict[str, dict],
    duplicate_to_canonical: dict[str, str],
) -> set[str]:
    row = detailed_by_id[question_id]
    result = set()
    for row_id in json.loads(row["evidence_row_ids_0based"]):
        article_id = f"row_{int(row_id):05d}"
        canonical = duplicate_to_canonical.get(article_id, article_id)
        result.add(f"{canonical}::c0000")
    return result


def retrieval_recall(retrieved: set[str], expected: set[str]) -> float:
    return len(retrieved & expected) / len(expected) if expected else 0.0


def main() -> int:
    args = parse_args()
    os.chdir(ROOT)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    load_dotenv(ROOT / ".env")
    os.environ["PIPELINE_PROVIDER"] = args.pipeline_provider
    os.environ["PIPELINE_MODEL"] = args.pipeline_model
    OUTPUT.mkdir(parents=True, exist_ok=True)

    ns = load_notebook_namespace()
    ns["PIPELINE_PROVIDER"] = args.pipeline_provider
    ns["PIPELINE_MODEL"] = args.pipeline_model
    seed = int(ns["SEED"])

    raw_first_5000 = pd.read_csv(DATASET, nrows=5000)
    golden = pd.read_csv(GOLDEN)
    detailed = pd.read_csv(GOLDEN_DETAILED)
    ns["validate_golden"](golden, require_answers=True)
    previous_article_limit = ns["LAB_MAX_ARTICLES"]
    ns["LAB_MAX_ARTICLES"] = 0
    first5000_exact = ns["standardize_news"](raw_first_5000)
    first5000_near, first5000_near_audit = ns["near_deduplicate_minhash_lsh"](first5000_exact)
    ns["LAB_MAX_ARTICLES"] = previous_article_limit
    first5000_near_audit.to_csv(OUTPUT / "near_dedup_audit_all.csv", index=False)
    ns["build_merge_audit_sample"](first5000_near_audit).to_csv(
        OUTPUT / "near_dedup_audit_sample.csv", index=False
    )
    corpus_raw, selected_ids = build_controlled_corpus(
        raw_first_5000, detailed, args.corpus_size, seed
    )

    exact_df = ns["standardize_news"](corpus_raw)
    news_df, near_audit = ns["near_deduplicate_minhash_lsh"](exact_df)
    chunks_df = ns["build_chunks"](news_df)
    near_audit.to_csv(OUTPUT / "controlled_corpus_near_dedup_audit.csv", index=False)

    extraction_source = chunks_df.copy()
    if args.skip_coref:
        extraction_source["resolved_text"] = extraction_source["text"]
        extraction_source["unresolved_mentions"] = [[] for _ in range(len(extraction_source))]
    else:
        coref_df = ns["run_coref"](chunks_df, batch_size=args.batch_size)
        extraction_source = extraction_source.merge(coref_df, on="chunk_id", how="left")
    coref_failures = int(
        extraction_source["unresolved_mentions"].map(
            lambda values: "COREF_BATCH_FAILED" in (values if isinstance(values, list) else [])
        ).sum()
    )

    raw_triples, extraction_errors, triple_rejections = ns["run_extraction"](
        extraction_source, batch_size=args.batch_size
    )
    raw_triples.to_csv(OUTPUT / "raw_triples.csv", index=False)
    extraction_errors.to_csv(OUTPUT / "extraction_errors.csv", index=False)
    triple_rejections.to_csv(OUTPUT / "triple_rejections.csv", index=False)
    if raw_triples.empty:
        raise RuntimeError("No valid triples were extracted; inspect triple_rejections.csv")

    entity_map, entity_audit = ns["build_resolution_map"](raw_triples)
    triples = ns["canonicalize_triples"](raw_triples, entity_map)
    triples, insertion_rejections = ns["validate_triples_for_insert"](triples)
    nodes = ns["build_nodes"](triples)
    entity_audit.to_csv(OUTPUT / "entity_resolution_audit.csv", index=False)
    insertion_rejections.to_csv(OUTPUT / "insertion_rejections.csv", index=False)
    triples.to_csv(OUTPUT / "validated_triples.csv", index=False)
    nodes.to_csv(OUTPUT / "nodes.csv", index=False)

    graph_counts = {
        "nodes": len(nodes),
        "edges": len(triples),
        "invalid_provenance_edges": 0,
    }
    top_degree = pd.DataFrame()
    inserted_nodes = 0
    edge_insert_stats = {"accepted_edges": 0, "rejected_edges": 0, "matched_and_written": 0}
    if not args.skip_neo4j:
        ns["connect_neo4j"]()
        ns["setup_graph_schema"]()
        inserted_nodes = ns["bulk_insert_nodes"](nodes)
        edge_insert_stats, edge_rejects = ns["bulk_insert_edges"](triples)
        if not edge_rejects.empty:
            edge_rejects.to_csv(OUTPUT / "neo4j_edge_rejections.csv", index=False)
        graph_counts, top_degree = ns["graph_checks"]()
        top_degree.to_csv(OUTPUT / "graph_top_degree.csv", index=False)

    ns["build_flat_index"](chunks_df)
    ns["build_entity_matcher"](nodes)

    eval_results = pd.DataFrame()
    comparison = pd.DataFrame()
    if not args.skip_eval:
        selected_eval_ids = stratified_eval_ids(golden, args.eval_size)
        eval_golden = golden.set_index("id").loc[selected_eval_ids].reset_index()
        detailed_by_id = detailed.set_index("id").to_dict("index")
        duplicate_to_canonical = dict(
            zip(near_audit.get("duplicate_article_id", []), near_audit.get("canonical_article_id", []))
        )
        rows = []
        checkpoint = OUTPUT / "graphrag_eval_checkpoint.csv"
        for q in ns["tqdm"](
            eval_golden.itertuples(index=False), total=len(eval_golden), desc="Benchmark"
        ):
            flat = ns["answer_flat_rag"](q.question)
            graph = ns["answer_graph_rag"](q.question)
            judge_flat = ns["judge_answer"](
                q.question, q.reference_answer, flat["answer"], flat["context"]
            )
            judge_graph = ns["judge_answer"](
                q.question, q.reference_answer, graph["answer"], graph["context"]
            )

            expected = expected_chunk_ids(q.id, detailed_by_id, duplicate_to_canonical)
            flat_ids = set(flat["retrieved"].get("chunk_id", []))
            graph_edge_ids = set(graph["graph_debug"]["edges"].get("source_chunk_id", []))
            graph_vector_ids = set(graph["vector_docs"].get("chunk_id", []))
            graph_ids = graph_edge_ids | graph_vector_ids
            diagnostics = graph["graph_debug"]["diagnostics"]
            rows.append({
                "id": q.id,
                "group": q.group,
                "question": q.question,
                "reference_answer": q.reference_answer,
                "flat_answer": flat["answer"],
                "graph_answer": graph["answer"],
                "flat_comprehensiveness": judge_flat["comprehensiveness"],
                "graph_comprehensiveness": judge_graph["comprehensiveness"],
                "flat_faithfulness": judge_flat["faithfulness"],
                "graph_faithfulness": judge_graph["faithfulness"],
                "flat_multi_hop_reasoning": judge_flat["multi_hop_reasoning"],
                "graph_multi_hop_reasoning": judge_graph["multi_hop_reasoning"],
                "flat_latency_s": flat["latency_s"],
                "graph_latency_s": graph["latency_s"],
                "flat_generation_latency_s": flat["generation_latency_s"],
                "graph_generation_latency_s": graph["generation_latency_s"],
                "flat_total_tokens": flat.get("total_tokens"),
                "graph_total_tokens": graph.get("total_tokens"),
                "flat_context_chars": len(flat["context"]),
                "graph_context_chars": len(graph["context"]),
                "flat_evidence_recall": retrieval_recall(flat_ids, expected),
                "graph_evidence_recall": retrieval_recall(graph_ids, expected),
                "flat_judge_rationale": judge_flat["rationale"],
                "graph_judge_rationale": judge_graph["rationale"],
                "graph_collected_edges": diagnostics.get("collected_edges", 0),
                "graph_supernode_events": len(diagnostics.get("supernode_events", [])),
                "generator_model": flat.get("model", args.pipeline_model),
                "judge_model": ns["JUDGE_MODEL"],
            })
            pd.DataFrame(rows).to_csv(checkpoint, index=False)

        eval_results = pd.DataFrame(rows)
        eval_results.to_csv(OUTPUT / "graphrag_eval_results.csv", index=False)
        grouped = ns["comparison_table"](eval_results)
        overall_input = eval_results.copy()
        overall_input["group"] = "overall"
        overall = ns["comparison_table"](overall_input)
        comparison = pd.concat([grouped, overall], ignore_index=True)
        for group_name, frame in list(eval_results.groupby("group")) + [("overall", eval_results)]:
            comparison.loc[len(comparison)] = {
                "Loại câu hỏi": group_name,
                "Metric": "Evidence recall@k",
                "Flat RAG": round(frame["flat_evidence_recall"].mean(), 3),
                "GraphRAG": round(frame["graph_evidence_recall"].mean(), 3),
                "Nhận xét phân tích": "Recall trên evidence row IDs; IDs không được dùng khi retrieval.",
            }
        comparison.to_csv(OUTPUT / "graphrag_vs_flatrag_summary.csv", index=False)

    manifest = {
        "run_timestamp": pd.Timestamp.now(tz="Asia/Ho_Chi_Minh").isoformat(),
        "scope": {
            "source_rows": 5000,
            "controlled_corpus_rows": len(selected_ids),
            "gold_evidence_rows_included": len(gold_evidence_ids(detailed)),
            "evaluation_questions": len(eval_results),
            "selection_note": "Union of gold evidence rows + fixed random distractors; no per-query gold lookup during retrieval.",
        },
        "models": {
            "embedding": ns["EMBED_MODEL"],
            "pipeline_provider": args.pipeline_provider,
            "generator_and_extractor": args.pipeline_model,
            "judge_provider": ns["JUDGE_PROVIDER"],
            "judge_model": ns["JUDGE_MODEL"],
        },
        "preprocessing": {
            "first5000_after_exact_dedup": len(first5000_exact),
            "first5000_after_near_dedup": len(first5000_near),
            "first5000_near_duplicates_merged": len(first5000_near_audit),
            "standardized_after_exact_dedup": len(exact_df),
            "after_near_dedup": len(news_df),
            "near_duplicates_merged": len(near_audit),
            "chunks": len(chunks_df),
            "coref_batch_failures": coref_failures,
        },
        "extraction": {
            "accepted_raw_triples": len(raw_triples),
            "rejected_candidates": len(triple_rejections),
            "failed_batches": len(extraction_errors),
            "validated_triples": len(triples),
            "entity_resolution_audit_rows": len(entity_audit),
            "preinsert_rejections": len(insertion_rejections),
        },
        "neo4j": {
            "skipped": args.skip_neo4j,
            "inserted_nodes": inserted_nodes,
            "edge_insert_stats": edge_insert_stats,
            "graph_checks": graph_counts,
        },
        "evaluation_overall": {} if eval_results.empty else {
            "flat_comprehensiveness": eval_results["flat_comprehensiveness"].mean(),
            "graph_comprehensiveness": eval_results["graph_comprehensiveness"].mean(),
            "flat_faithfulness": eval_results["flat_faithfulness"].mean(),
            "graph_faithfulness": eval_results["graph_faithfulness"].mean(),
            "flat_multi_hop_reasoning": eval_results["flat_multi_hop_reasoning"].mean(),
            "graph_multi_hop_reasoning": eval_results["graph_multi_hop_reasoning"].mean(),
            "flat_latency_s": eval_results["flat_latency_s"].mean(),
            "graph_latency_s": eval_results["graph_latency_s"].mean(),
            "flat_total_tokens": eval_results["flat_total_tokens"].mean(),
            "graph_total_tokens": eval_results["graph_total_tokens"].mean(),
            "flat_evidence_recall": eval_results["flat_evidence_recall"].mean(),
            "graph_evidence_recall": eval_results["graph_evidence_recall"].mean(),
        },
    }
    (OUTPUT / "lab_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=float))
    if ns.get("driver") is not None:
        ns["driver"].close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
