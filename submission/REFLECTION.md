# Reflection — Lab 19

**Tên:** _Ha Hoang Tuan Hung_
**Cohort:** _A20_
**Path đã chạy:** _lite_

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

- **Exact queries:** Keyword (BM25) thắng, vì nó match chính xác các từ khóa hiếm hoặc mã lỗi (ví dụ "error 404", mã SKU), thứ mà vector embeddings đôi khi hiểu lầm hoặc bỏ qua.
- **Paraphrase queries:** Vector (Semantic) thắng, vì nó nắm bắt được ý nghĩa tương đồng của câu (ví dụ "co giãn linh hoạt" vs "auto-scaling"), điều mà BM25 sẽ trượt do khác token.
- **Mixed queries:** Hybrid thắng, bù đắp yếu điểm của cả hai.

**Khi nào không dùng Hybrid:**
- Pure BM25: Hệ thống e-commerce chuyên tìm kiếm mã sản phẩm (SKUs), số serial, part number, hoặc khi latency & tài nguyên tính toán bị giới hạn nghiêm ngặt.
- Pure Vector: Hệ thống FAQ/Q&A thuần ngữ nghĩa nơi người dùng luôn đặt câu hỏi tự nhiên dài (ít từ khóa cụ thể) và tài nguyên vector index đủ mạnh.

---

## Điều ngạc nhiên nhất khi làm lab này

Mức độ rớt recall nghiêm trọng (Recall Cliff) của kỹ thuật Post-filter khi filter condition quá hẹp, và sự hiệu quả của Filtered-ANN trong Qdrant.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _<tên đồng đội nếu có>_
