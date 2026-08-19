# Reflection & Action Plan — Nguyễn Thị Hương Trà

## Mapping bài giảng vào code

| Khái niệm | Module/hàm | Quan sát thực tế |
|---|---|---|
| Streaming, exact dedup, chunking | `load_news()`, `standardize_news()`, `build_chunks()` | Đường dẫn ưu tiên file local, không còn phụ thuộc `/content`; 5.000 dòng còn 2.105 sau lọc + exact dedup. |
| Near-duplicate detection | `near_deduplicate_minhash_lsh()` | LSH chỉ sinh candidate, exact 5-shingle Jaccard mới quyết định merge; tránh pairwise `O(N²)`. |
| Conservative coreference | `resolve_coref_batch()`, `run_coref()` | 20 batch không lỗi, nhưng generic mention vẫn lọt qua extraction; cần validator độc lập với LLM. |
| Schema + allowlist | `_validate_relation_candidate()`, `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` | Relation, endpoint type, evidence, confidence, modality và generic entity đều được kiểm trước Cypher. |
| Entity resolution | `build_resolution_map()`, `merge_guard()`, `UF` | HNSW tìm candidate theo type; cosine 0,90 + lexical guard; audit 43 candidate thay vì merge mù. |
| Bulk Cypher | `bulk_insert_nodes()`, `bulk_insert_edges()` | `UNWIND`, stable IDs và allowlist relation; graph cuối 20 nodes/13 edges. |
| Flat retrieval | `build_flat_index()`, `retrieve_flat_context()` | FAISS FlatIP đơn giản, latency thấp và coverage ngang Graph trong sample. |
| Graph traversal | `match_seeds()`, `retrieve_graph_context()` | BFS có recent-edge ordering, degree cap 50 và global cap 250; graph nhỏ chưa kích hoạt cap thật. |
| Golden evaluation | `validate_golden()`, `evaluate_systems()`, `judge_answer()` | 15 câu cân bằng 3 nhóm; checkpoint từng câu; evidence recall tính độc lập với LLM judge. |
| Failure audit | `graph_checks()`, `test_supernode_policy()` và pytest | Answer-quality test chưa đủ; semantic audit phát hiện 2 edge sai dù schema hợp lệ. |

## Debugging và bài học

Lỗi vận hành đầu tiên là notebook hard-code `/content/hackernoon_subset.csv`, nên chạy local báo `FileNotFoundError`. Loader được đổi sang `pathlib`: ưu tiên biến `LOCAL_DATA_PATH`, sau đó file ở repo, cuối cùng mới stream Hugging Face. Lỗi gated dataset trước đó không còn chặn local run; nếu stream thì token vẫn phải được process Jupyter nhận qua `.env`/Colab Secrets.

Lỗi khó nhất về chất lượng là “evidence verbatim nhưng fact sai trạng thái”. Triple `Ericsson ACQUIRED Aeris` vượt validation vì câu evidence tồn tại, nhưng từ “will be transferred” cho biết sự kiện chưa hoàn tất. Tôi học được rằng extraction validation cần kiểm ít nhất bốn lớp: schema, grounding, entity identity và event modality. Cần audit raw output bằng dữ liệu thật; test schema đơn thuần không bắt được lỗi ngữ nghĩa này.

Một bài học khác là đánh giá hệ thống phải tách retrieval và generation. `G5000-36` điểm 1 vì evidence recall 0; đổi prompt generator không sửa được. Ngược lại `G5000-02` có answer tốt dù graph chứa một edge sai, nên điểm answer không thay thế graph audit.

## Kế hoạch áp dụng

**Dự án đề xuất:** trợ lý nghiên cứu doanh nghiệp/công nghệ theo dòng thời gian.

Tôi chọn Hybrid RAG: Flat retrieval làm đường nhanh cho factoid; GraphRAG chỉ bật khi query chứa nhiều entity, quan hệ hoặc mốc thời gian. Router này tránh trả latency Graph cho mọi câu nhưng vẫn hỗ trợ cross-document reasoning.

Schema dự kiến:

- Nodes: `Company`, `Person`, `Technology`, `Product`, `Event`, `Document`.
- Relations: `ACQUIRED`, `DEVELOPED`, `INVESTED_IN`, `FOUNDED`, `WORKED_AT`, `PARTNERED_WITH`, `USES`, `LEADS`, cùng `MENTIONED_IN` và `SUPERSEDES` cho event timeline.
- Event có `state ∈ {rumored, planned, announced, completed, cancelled}`, `event_date`, `published_date`, `confidence` và provenance.

Entity resolution dùng alias registry đã review, block theo node type/ngôn ngữ/domain, HNSW candidate generation, lexical guard và human queue cho vùng bất định. Không tự merge cặp chỉ vì embedding cao. Super-node dùng relation-aware/time-aware ranking, cap theo query, community partition và global edge budget.

Action plan:

1. Xây golden set 100 câu, gồm negation, planned/completed và alias hard cases.
2. Đo retrieval recall trước, rồi mới judge answer; dùng judge khác họ model.
3. Audit ngẫu nhiên hàng tuần các edge mới và toàn bộ merge sát threshold.
4. Canary deploy prompt/threshold, rollback nếu false edge/false merge tăng.
5. Chỉ mở rộng full corpus khi chi phí extraction, retry rate và provenance checks đạt SLO.

## Tự đánh giá

| Tiêu chí | Điểm (1–5) | Lý do |
|---|---:|---|
| Hiểu GraphRAG | 4 | Đã triển khai đủ pipeline và phân tách retrieval/generation failure. |
| Kiểm soát AI Coding Agent | 5 | Từ chối `O(N²)`, thêm contracts, audit và regression tests. |
| Chất lượng knowledge graph | 4 | Schema/provenance sạch; corpus nhỏ và event-state schema còn cần mở rộng. |
| Phân tích/debug | 4 | Truy được root cause và sửa 2 semantic edges; cần benchmark lại bằng judge độc lập. |
