# Thuyết minh kỹ thuật — Lab 19 GraphRAG vs Flat RAG

**Học viên:** Nguyễn Thị Hương Trà

**Ngày chạy:** 19/08/2026

**Phạm vi đo:** 5.000 dòng đầu, corpus benchmark cố định 100 bài/100 chunks, 15 câu (5 factoid, 5 multi-hop, 5 cross-doc).

## 1. Coreference sai ở tình huống nào?

Ở `row_03714::c0000`, extractor tạo chủ thể `the company` cho câu “The company plans to use ... chip-stacking tech”. Đây không phải một thực thể định danh: coreference không đủ chắc chắn để thay đại từ bằng công ty đúng nhưng extractor vẫn biến nó thành node. Hậu quả là một node giả và cạnh `the company -[USES]-> chip-stacking tech`; traversal từ node này vừa gây nhiễu vừa có thể gán kế hoạch của công ty A cho công ty B.

Sau audit, validator từ chối danh sách generic mentions (`the company`, `it`, `they`, ...), đồng thời từ chối hành động dự kiến đối với `USES/DEVELOPED`. Hai node mồ côi và cạnh sai đã được xóa khỏi Aura. Với coreference không chắc chắn, lựa chọn an toàn là giữ nguyên mention và không tạo triple, thay vì đoán.

## 2. Entity threshold bao nhiêu, vì sao?

Entity resolution dùng FAISS `IndexHNSWFlat` với inner product trên embedding đã L2-normalize, tương đương cosine similarity. Ngưỡng merge là **0,90**, `top_k=5`, HNSW `M=32`, `efConstruction=80`, `efSearch=64`. Ngưỡng cao phù hợp với knowledge graph vì false merge nguy hiểm hơn missed merge: một merge sai làm lan truyền toàn bộ cạnh của hai thực thể.

Kết quả thật có 43 candidate audit; điểm cao nhất là `Llama 2`–`Code Llama` = **0,7835**, nên bị `REJECT_BELOW_THRESHOLD`. `Aeris Communications`–`Aeris` = **0,7388** cũng không tự gộp; đây là false negative có chủ ý và nên xử lý bằng alias được review, không hạ ngưỡng toàn cục.

Để kiểm tra yêu cầu “similarity cao nhưng không nên merge”, probe đối kháng `H2O AI Cloud`–`H2O AI Cloud Platform` đạt cosine **0,932456** nhưng bị lexical guard chặn với `PRODUCT_CONTAINMENT_CONFLICT`: tên thứ hai có thể là platform/family rộng hơn, không được suy ra là cùng một product chỉ từ embedding. Probe được lưu ở `outputs/entity_resolution_guard_probe.csv`. Ngoài ra guard còn chặn ticker nếu không có manual alias và kiểm tra họ/tên cho `Person`.

## 3. Candidate nào similarity cao nhưng không nên merge?

Candidate điển hình là `H2O AI Cloud` và `H2O AI Cloud Platform` ở trên. ANN có nhiệm vụ **sinh ứng viên**, không phải tự quyết định identity. Dù cosine 0,932456 vượt 0,90, quan hệ containment khiến hai tên có thể biểu diễn product và product family khác nhau; do đó quyết định đúng là `REJECT_GUARD`. Nếu có nguồn chính thức chứng minh alias, cặp này mới được đưa vào `MANUAL_ALIASES` qua review.

## 4. Top 3 node theo degree

Graph cuối sau failure audit có:

| Hạng | Entity | Type | Degree |
|---:|---|---|---:|
| 1 | artificial intelligence | Technology | 3 |
| 2 | Google Cloud | Company | 3 |
| 3 | Ericsson | Company | 2 |

NVIDIA cũng degree 2, đồng hạng ba. Graph nhỏ chưa có super-node thật. Test synthetic tạo node degree 150 xác nhận mỗi lần fetch chỉ lấy 50 cạnh; toàn traversal vẫn bị chặn bởi `GLOBAL_EDGE_CAP=250`.

## 5. Vì sao ưu tiên 50 edge mới nhất có thể đúng hoặc sai?

Ưu điểm: giới hạn fan-out, latency, token context và nguy cơ một node phổ biến như “AI” lấn át kết quả; ưu tiên thời gian mới cũng hợp với câu hỏi tin tức hiện hành. Rủi ro: câu hỏi lịch sử có thể cần cạnh cũ bị cắt, và `published_date` mới không đồng nghĩa relevance cao. Production nên kết hợp time decay với query relevance, diversity theo relation/source và cho phép time filter từ câu hỏi; cap 50 là safety bound chứ không phải ranking duy nhất.

## 6. Flat RAG thắng nhóm nào?

Về chất lượng trung bình, Flat không thắng một nhóm hoàn chỉnh: factoid là `4,8/5,0`, multi-hop `3,0/3,0`, cross-doc `3,4/3,4` (Flat/Graph, comprehensiveness). Flat thắng rõ về latency ở cả ba nhóm và thắng riêng câu `G5000-15`: điểm trung bình judge **5,0** so với Graph **4,333**. Cả hai lấy đúng hai evidence chunks, nhưng câu Graph ngắn hơn và bỏ bớt chi tiết “Word, Excel, Outlook/Edge”, nên bị trừ comprehensiveness.

## 7. GraphRAG thắng nhóm nào?

GraphRAG thắng nhẹ nhóm factoid về quality (`5,0` so với `4,8`) và hòa hai nhóm còn lại ở mức trung bình. Ví dụ tốt nhất là `G5000-02` và `G5000-47`, mỗi câu Graph hơn Flat **0,667** điểm trung bình. Với `G5000-02`, graph context giúp phân biệt các bài đầu nói **planned transfer** với bằng chứng 18/01 nói giao dịch đã hoàn tất. Với `G5000-47`, Graph trả lời dứt khoát Palo Alto Networks chỉ là nguồn được dẫn, không phải partner trong deal Keysight–Synopsys.

## 8. Latency/token trade-off

| Metric tổng thể | Flat RAG | GraphRAG | Graph − Flat |
|---|---:|---:|---:|
| Comprehensiveness | 3,733 | 3,800 | +0,067 |
| Faithfulness | 3,867 | 3,867 | 0,000 |
| Multi-hop reasoning | 3,733 | 3,800 | +0,067 |
| Latency end-to-end | 1,949 s | 5,593 s | +3,645 s / 2,87× |
| Total tokens | 600,7 | 540,8 | −59,9 / −10,0% |
| Evidence recall@k | 0,811 | 0,811 | 0,000 |

Graph có quality gain nhỏ nhưng latency tăng do seed matching, Cypher/BFS và generation. Token lại thấp hơn trong sample vì graph context có cấu trúc và ngắn hơn; không nên khái quát rằng Graph luôn rẻ. Indexing overhead cũng lớn hơn: extraction bằng LLM, entity resolution và Neo4j write, trong khi Flat chỉ cần embedding + FAISS.

Judge và generator cùng dùng `gpt-4o-mini` do Groq model cấu hình ban đầu đã retired/rate-limit. Vì vậy kết quả có rủi ro same-family bias; cần lặp lại bằng judge khác họ model và báo confidence interval trước quyết định production.

## 9. Đề xuất AI Agent nào không dùng, vì sao?

Tôi không dùng pairwise cosine/Jaccard trên toàn bộ dataset vì độ phức tạp `O(N²)` và không phù hợp 100.000 bài. Near-dedup dùng MinHash 128 permutations trên word 5-shingles, LSH threshold **0,75** để sinh candidate, sau đó chỉ verify các candidate bằng exact Jaccard threshold **0,82** và length ratio **≥0,80**. Trên 5.000 dòng: exact dedup còn 2.105, near-dedup còn 2.096, merge 9 cặp.

Audit thủ công toàn bộ 9 cặp merge cho kết quả 9 TP, 0 FP, tức **false-positive quan sát 0/9 = 0%**. Đây là mẫu nhỏ nên không được coi là upper bound thống kê. Audit lưu ID hai bài, score, length ratio, title/date/preview trong `near_dedup_audit_all.csv`; reviewer mở nội dung gốc theo ID, điền `TP/FP` và lý do. File đã review là `near_dedup_manual_audit.csv`. Khi có FP, tăng hard threshold, thêm rule theo domain/date hoặc đưa cặp sát ngưỡng vào review queue.

## 10. Scale 350 MB: bottleneck đầu tiên là gì?

Bottleneck đầu tiên là các lượt LLM cho coreference và triple extraction, không phải HNSW/LSH: 100.000 bài tạo nhiều batch, tốn tiền, rate-limit và khó retry đúng-once. Sau đó mới đến graph write amplification và traversal quanh hub.

Kiến trúc scale: stream theo partition; content hash + MinHash LSH trước LLM; hàng đợi async có rate limiter, retry/backoff và checkpoint theo `chunk_id`; chỉ coreference các chunk có mention mơ hồ; extraction batch có schema validator; entity candidates qua type/domain blocking + HNSW; write Neo4j bằng `UNWIND` idempotent và edge ID ổn định; theo dõi reject/merge drift. Retrieval dùng community/tenant/time partition, super-node cap và cache. Một golden canary set phải chạy lại sau mỗi thay đổi prompt/threshold.

## Ghi chú tính toàn vẹn kết quả

Benchmark CSV là snapshot đo **trước** failure audit cuối. Audit phát hiện và loại 2/15 cạnh: một `ACQUIRED` chưa hoàn tất và một `USES` có chủ thể generic. Graph hiện tại có 20 nodes/13 edges và tất cả contract checks bằng 0; không dùng kết quả benchmark cũ để tuyên bố hai guard đã cải thiện điểm số.
