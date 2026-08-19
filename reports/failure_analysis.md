# Failure analysis — Flat RAG và GraphRAG

Phân tích dựa trên `graphrag_eval_results.csv`, evidence row IDs trong golden set và audit graph sau benchmark. Latency là end-to-end; không dùng gold IDs trong retrieval.

## Case 1 — G5000-36: cả hai retriever lấy sai sự kiện

**Câu hỏi:** Hai row 2532 và 2537 có headline Amazon AI gần giống nhau; sự kiện nào cần lưu một lần và chi tiết nào được union an toàn?

**Quan sát:** Flat và Graph cùng đạt điểm trung bình 1/5 và evidence recall 0. Flat trả lời bằng một sự kiện ServiceNow không liên quan; Graph lấy tin AWS đang cân nhắc chip AMD, cũng không phải bài Amazon AI service cần hỏi. Graph thu được 0 graph edge cho câu này.

**Root cause:**

1. Candidate retrieval không lấy `row_02532::c0000` và `row_02537::c0000`; lỗi xảy ra trước generation.
2. Entity seed “Amazon/AWS” có nhiều nội dung gần ngữ nghĩa, trong khi graph extraction không có edge cho đúng event.
3. Near-dedup corpus benchmark không merge cặp này vì controlled corpus sau exact dedup chỉ còn một representation phù hợp; query vẫn nhắc hai row IDs/headline, tạo lexical mismatch.

**Khắc phục:** thêm hybrid BM25 + dense retrieval cho tên riêng/row-like cues; index title riêng; nếu seed traversal không có edge thì fallback vector k lớn hơn; tạo event node hoặc canonical event ID để hai bài cùng trỏ đến một sự kiện. Regression gate: evidence recall của G5000-36 phải >0 trước khi chấm câu trả lời.

## Case 2 — G5000-31: timeline thiếu mốc March

**Câu hỏi:** Sắp xếp các bước của OpenAI từ March đến July: plug-ins, kế hoạch open-source model, marketplace và hợp tác AP.

**Quan sát:** cả hai đạt 1/5; evidence recall 0,75, nghĩa là chỉ lấy 3/4 evidence chunks. Câu trả lời bỏ mốc plug-ins tháng 3, nên timeline không đầy đủ dù các mốc sau đúng.

**Root cause:** top-k context budget ưu tiên các bài May–July có semantic overlap cao hơn. Graph có 3 collected edges nhưng thiếu edge/event cho plug-ins; traversal không thể khôi phục thông tin chưa được extract. Đây là coverage failure, không phải lỗi suy luận của generator.

**Khắc phục:** query decomposition thành bốn sub-events, retrieval theo time buckets March/May/June/July, diversity/MMR theo `published_date`, và completeness checker yêu cầu mỗi noun phrase trong câu hỏi có evidence. Nếu một mốc thiếu, self-correction mở rộng hop/k rồi vector fallback.

## Case 3 — Audit graph phát hiện hai false edges

**Sai lệch 1:** `row_01746::c0000` tạo `Ericsson -[ACQUIRED]-> Aeris`, trong khi evidence nói các tài sản *will be transferred*; vừa sai modality vừa có nguy cơ sai hướng buyer/seller.

**Sai lệch 2:** `row_03714::c0000` tạo `the company -[USES]-> chip-stacking tech` từ câu “plans to use”; chủ thể chưa resolve và hành động chưa xảy ra.

**Root cause:** prompt/validator ban đầu kiểm schema, confidence và evidence verbatim nhưng chưa kiểm modality/state và generic entity. Evidence có thật không đồng nghĩa proposition đã hoàn tất.

**Khắc phục đã triển khai:**

- prompt cấm biến planned/considered action thành fact hoàn tất và nhắc hướng `buyer -[ACQUIRED]-> target`;
- `_validate_relation_candidate()` và `validate_triples_for_insert()` cùng có defense-in-depth guards;
- từ chối generic mentions và các pattern tentative cho `ACQUIRED`, `DEVELOPED`, `USES`;
- loại 2 edge và 2 orphan nodes khỏi Aura; graph cuối 20 nodes/13 edges, 0 contract violation;
- bổ sung regression tests cho generic subject, planned acquisition, endpoint direction và hallucinated evidence.

## Case đối chứng — G5000-02: Graph tốt hơn nhưng phơi bày lỗi modeling

Graph đạt 5/5, Flat 4,333/5 và cả hai recall 1,0. Graph trả lời tốt về tiến trình “planned transfer → completed acquisition”. Tuy nhiên audit sau đó chứng minh một cạnh M&A ban đầu đã biểu diễn trạng thái planned như completed. Điều này cho thấy answer đúng không bảo đảm graph đúng: context văn bản/provenance có thể cứu generator dù edge sai. Vì vậy evaluation câu trả lời phải đi cùng graph-contract và semantic-edge audit.

## Ưu tiên sửa lỗi

1. Retrieval coverage gate theo evidence recall và query decomposition.
2. Event modality/state trong schema (`planned`, `announced`, `completed`) thay vì ép mọi tin vào relation hoàn tất.
3. Independent judge model và lặp nhiều seed để giảm variance.
4. Dashboard theo dõi extraction rejection, false-merge sample, missing provenance và super-node fan-out.
