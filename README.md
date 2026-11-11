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

### 3. Pipeline làm sạch cho phân loại cảm xúc (`vi_clean/`)
Gộp UIT-VSFC (đặt vào `data/uit_vsfc/<split>/sents.txt` + `sentiments.txt`) với `raw_reviews/*.json`, rồi chạy 5 bước: NFC, regex, teencode, loại trùng MD5, lọc TTR thấp.
```powershell
python -m vi_clean.merge        # ghi data/clean/dataset.csv và report.json
uvicorn vi_clean.api:app        # POST /clean, /clean/batch, /jobs/merge-clean
```
- Từ điển teencode: `vi_clean/data/teencode_v1.json` (sửa tay, có `version`).
- Mọi tham số nằm trong `vi_clean/config.py`; `report.json` so sánh số mẫu cuối với 34,250 và ghi rõ chênh lệch.

### 4. Fine-tune PhoBERT cho cảm xúc (`vi_train/`)
Fine-tune 2 giai đoạn (UIT-VSFC, rồi Shopee), đánh giá leave-one-product-out trên các sản phẩm Shopee.
```powershell
python -m vi_train.train --shopee-dir raw_reviews   # tải vinai/phobert-base (~543 MB), lưu checkpoints/<run>/ và data/train/preds/<run>/
python -m vi_train.evaluate --run phobert-p1.0      # ghi reports/<run>.json (Macro-F1, recall từng lớp, độ trễ)
uvicorn vi_clean.api:app                            # thêm POST /predict và /predict/batch (503 nếu chưa có checkpoints/<run>/final)
```
- Mỗi lần chạy là một "run": `--run NAME` đặt tên, `--model` đổi mô hình (vd. `FacebookAI/xlm-roberta-base`), `--weight-power P` làm mềm trọng số lớp (`(N/(K*n_c))**P`, mặc định 1.0), `--no-segment` bỏ tách từ (dùng cho XLM-R). Huấn luyện từ chối ghi đè một run đã có; dùng `--overwrite` nếu thật sự muốn.
- API phục vụ run trong biến môi trường `VI_RUN` (mặc định `phobert-p0.5`, run được chọn bằng điểm validation); `/predict` trả `label: null` cho văn bản trống.
- Kết quả xem trong `reports/<run>.json`; tập giữ lại không phải Gold Set do người gán nhãn, và nhãn Shopee suy ra từ số sao.

### 5. Trang web phân tích review (`vi_web/`)
Dán link sản phẩm Shopee để xem EDA cảm xúc do mô hình dự đoán và mở từng review; hoặc mở một sản phẩm đã lưu.
```powershell
$env:SAVED_REVIEWS_DIR = "C:\duong\dan\raw_reviews"   # thư mục các file review đã cào (mặc định <repo>\raw_reviews)
uvicorn vi_clean.api:app                              # mở http://127.0.0.1:8000/
```
- **Chỉ dùng trên máy của bạn.** API không có xác thực; đừng mở ra mạng ngoài. Không có gì được lưu lại, review chỉ nằm trong bộ nhớ của tiến trình.
- **Sản phẩm đã lưu:** dùng dự đoán giữ-lại (held-out) của leave-one-product-out, nên mô hình chưa thấy sản phẩm đó. Review trùng với review của sản phẩm khác không có dự đoán giữ-lại và hiện là "not predicted". Chế độ này không có xác suất.
- **Link Shopee:** máy chủ mở một cửa sổ Chrome bằng hồ sơ đăng nhập đã lưu (`SHOPEE_PROFILE_DIR`, mặc định `<repo>\shopee_chrome_profile`). Nếu Shopee yêu cầu đăng nhập, hãy đăng nhập trong cửa sổ đó rồi bấm "I'm ready". Dùng mô hình cuối cùng (`VI_RUN`), nên sản phẩm nằm trong dữ liệu huấn luyện sẽ có cảnh báo.
- Chrome chỉ cho một phiên trên mỗi hồ sơ: đóng crawler tự chạy của bạn trước khi dùng. Shopee có thể từ chối (403, đăng nhập, captcha); trang chỉ báo lỗi, không tự thử lại và không vượt qua. Cào Shopee trái điều khoản của họ, chỉ dùng cho vài sản phẩm.
- Một lần cào mặc định tối đa 1.000 review (trần 3.000).
