# Shopee Review Crawler & Cleaner

Dự án cào đánh giá sản phẩm Shopee và làm sạch dữ liệu review (lọc spam, phân loại cảm xúc, phát hiện bot/trùng lặp).

## Cấu trúc thư mục

```
shopee_review_crawler/
├── shopee_crawler_persistent.py  # Script cào review chính (dùng Playwright lưu phiên đăng nhập)
├── clean_reviews.py              # Script làm sạch review, gán nhãn và xuất CSV/JSON
├── raw_reviews/                  # Thư mục lưu các file review JSON gốc từ Shopee
├── cleaned_reviews/              # Thư mục lưu file review đã được làm sạch (.csv và .json)
├── shopee_chrome_profile/        # Profile trình duyệt lưu session đăng nhập Shopee
└── README.md                     # Tài liệu hướng dẫn sử dụng
```

---

## Hướng dẫn sử dụng

### 1. Cào đánh giá sản phẩm Shopee
Chạy lệnh:
```powershell
python shopee_crawler_persistent.py
```
- Chương trình sẽ hỏi bạn dán link sản phẩm Shopee và số lượng review tối đa muốn cào (mặc định 1000).
- Trình duyệt Chrome sẽ tự động mở trang sản phẩm.
- Nếu chưa đăng nhập: Hãy đăng nhập (quét mã QR bằng app Shopee) rồi nhấn `ENTER` trên terminal để bắt đầu cào tự động.
- File kết quả gốc sẽ được lưu tự động vào thư mục `raw_reviews/` với tên theo 7 từ đầu của sản phẩm (ví dụ: `raw_reviews/Hop_dung_bao_ngon_tay_cat_giu.json`).

### 2. Làm sạch dữ liệu đánh giá
Chạy lệnh:
```powershell
python clean_reviews.py
```
- Script tự động quét các file trong `raw_reviews/`.
- Nếu có 1 file, script sẽ tự xử lý ngay. Nếu có nhiều file, bạn có thể chọn số thứ tự hoặc gõ `all` để làm sạch toàn bộ.
- File kết quả sau khi làm sạch sẽ được lưu vào thư mục `cleaned_reviews/` gồm 2 định dạng:
  - `.csv` (mở tốt trên Excel với mã hóa UTF-8 BOM)
  - `.json`
