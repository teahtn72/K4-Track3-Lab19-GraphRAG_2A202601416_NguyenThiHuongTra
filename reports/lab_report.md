# Báo cáo Lab 19 — Production GraphRAG vs Flat RAG

**Học viên:** Nguyễn Thị Hương Trà

**Khóa:** AICB-K34 · Track 3

**Ngày:** 19/08/2026

## 1. Kết quả chạy

Pipeline đọc 5.000 dòng đầu của `hackernoon_subset.csv`. Sau chuẩn hóa, lọc nội dung ngắn và exact dedup còn 2.105 bài; MinHash/LSH loại thêm 9 near-duplicates, còn 2.096. Benchmark dùng corpus kiểm soát cố định 100 bài (51 evidence rows hợp của golden set + 49 distractors cố định), 100 chunks và 15 câu cân bằng ba nhóm. Gold row IDs chỉ dùng để cấu tạo corpus có thể trả lời và tính recall, không được dùng trong retrieval từng câu.

Extraction ban đầu sinh 15 triples. Failure audit phát hiện 2 cạnh sai về modality/generic entity, sau sửa còn 13 triples. Neo4j Aura cuối có 20 nodes/13 edges; mọi kiểm tra missing provenance, relation allowlist, endpoint type, node subtype và edge ID đều bằng 0 lỗi.

| Metric tổng thể | Flat RAG | GraphRAG | Nhận xét |
|---|---:|---:|---|
| Comprehensiveness (1–5) | 3,733 | 3,800 | Graph +0,067 |
| Faithfulness (1–5) | 3,867 | 3,867 | Hòa |
| Multi-hop reasoning (1–5) | 3,733 | 3,800 | Graph +0,067 |
| Latency end-to-end | 1,949 s | 5,593 s | Graph 2,87× chậm hơn |
| Token trung bình | 600,7 | 540,8 | Graph thấp hơn 10,0% trong sample |
| Evidence recall@k | 0,811 | 0,811 | Hòa |

Theo nhóm, Graph hơn ở factoid comprehensiveness `5,0` so với `4,8`; multi-hop hòa `3,0`; cross-doc hòa `3,4`. Kết quả không chứng minh Graph tốt hơn đáng kể: sample nhỏ, graph nhỏ, và generator/judge cùng `gpt-4o-mini` do Groq rate-limit/model retirement. Cần judge độc lập và nhiều lần chạy trước quyết định production.

## 2. Near-duplicate design và audit

Giải pháp dùng MinHash 128 permutations trên word 5-shingles. LSH threshold **0,75** tạo candidates; chỉ candidates được exact Jaccard verify ở **0,82**, kèm length ratio **≥0,80**. Vì vậy không có pairwise cosine/Jaccard `O(N²)` toàn dataset.

`near_dedup_audit_all.csv` lưu canonical/duplicate IDs, Jaccard, length ratio, title, date và preview. Reviewer ưu tiên cặp sát threshold, mở nội dung gốc theo hai ID rồi ghi `TP/FP` và lý do. Audit lần này xem toàn bộ 9 merge: **9 TP, 0 FP; observed false-positive = 0%**. Do n=9 nhỏ, đây chỉ là số quan sát, không phải bảo đảm production.

## 3. Các quyết định kỹ thuật chính

- Coreference được xử lý conservative; generic mention không chắc chắn bị chặn khỏi graph.
- Triple schema chỉ nhận `Company`, `Person`, `Technology` và 8 relation allowlisted. Mỗi edge bắt buộc có `source_chunk_id`, `published_date`, `evidence`, `confidence` và stable `edge_id`.
- Entity resolution dùng HNSW ANN theo node type, cosine threshold 0,90, manual aliases và lexical guards. Probe `H2O AI Cloud`–`H2O AI Cloud Platform` có cosine 0,932456 vẫn bị chặn vì product containment.
- Neo4j insert dùng `UNWIND` và query tách riêng theo relation allowlist; không ghép relation do LLM sinh trực tiếp vào Cypher.
- Traversal dùng degree threshold 100, cap 50 edge mới nhất ở super-node và `GLOBAL_EDGE_CAP=250`. Synthetic test với degree 150 đã pass.

## 4. Failure analysis tóm tắt

`G5000-36` là failure retrieval: cả hai recall 0 và điểm 1/5 vì không lấy hai Amazon AI evidence chunks. `G5000-31` recall 0,75, bỏ mốc plug-ins tháng 3 nên timeline thiếu. `G5000-15` là ca Flat thắng: cả hai recall 1 nhưng Graph rút gọn quá mức, điểm 4,333 so với 5. `G5000-02` là ca Graph thắng 5 so với 4,333 nhờ biểu diễn tiến trình planned → completed, song chính audit cũng phát hiện edge planned ban đầu bị model hóa như completed.

Chi tiết root cause và biện pháp nằm trong [failure_analysis.md](failure_analysis.md).

## 5. Scale và trade-off

Flat RAG phù hợp fast path; GraphRAG nên route cho query nhiều entity/time-hop. Bottleneck đầu tiên ở 350 MB là LLM coreference/extraction: cần streaming partitions, content/near dedup trước LLM, async queue, rate-limit, checkpoint/idempotency và canary golden tests. Entity resolution dùng blocking + HNSW, graph write dùng batched `UNWIND`, retrieval dùng community/time partition và fan-out budget.

Không dùng pairwise `O(N²)` là quyết định kiểm soát AI Agent quan trọng nhất. ANN/LSH chỉ tạo candidate; deterministic guards và audit con người mới quyết định merge/write.

## 6. Reflection

Lỗi `/content/hackernoon_subset.csv` được sửa bằng loader portable ưu tiên `LOCAL_DATA_PATH` và file trong repo rồi mới fallback Hugging Face. Bài học lớn nhất là evidence verbatim chưa đủ đảm bảo fact đúng: validator phải hiểu modality và identity. Đánh giá answer cũng không thay thế graph audit.

Mapping bài giảng, bài học debugging và action plan đầy đủ nằm trong [reflection_NguyenThiHuongTra.md](reflection_NguyenThiHuongTra.md). Mười câu bảo vệ kiến trúc nằm trong [technical_defense.md](technical_defense.md).

## 7. Tính tái lập và artifacts

- Runner: `python scripts/run_lab_test.py --corpus-size 100 --eval-size 15 --batch-size 5`
- Unit/contracts: `python -m pytest -q`
- Raw details: `outputs/graphrag_eval_results.csv`
- Grouped comparison: `outputs/graphrag_vs_flatrag_summary.csv`
- Run configuration/counts: `outputs/lab_run_manifest.json`
- Merge audits: `outputs/near_dedup_audit_all.csv`, `outputs/near_dedup_manual_audit.csv`
- Entity/edge audits: `outputs/entity_resolution_audit.csv`, `outputs/post_benchmark_guard_rejections.csv`

Benchmark CSV là snapshot trước khi hai semantic edges bị loại; current graph/`validated_triples.csv` là bản sau audit. Không tuyên bố guard mới cải thiện score nếu chưa benchmark lại.
