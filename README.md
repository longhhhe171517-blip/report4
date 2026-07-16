# DSP391m — Nhóm 8: EV vs Gasoline Cost API

API dự đoán **giá bán lại**, **mức tiêu thụ năng lượng/nhiên liệu**, và **Total Cost of
Ownership (TCO)** cho xe điện (EV) và xe xăng (Gasoline). Đóng gói từ notebook
`DSP391M_Report4_Final.ipynb` (mục II. Đóng gói và Triển khai) của đồ án
DSP391m — Data Science Capstone Project. Repo này để **push lên GitHub rồi deploy miễn phí
trên Render** (đăng nhập Render bằng tài khoản GitHub).

Sau khi deploy xong, mở URL Render cấp cho bạn để dùng giao diện demo (`/`), hoặc `/docs`
để xem Swagger UI và gọi thử từng endpoint.

## ⚠️ BƯỚC BẮT BUỘC TRƯỚC KHI DEPLOY: thêm file mô hình vào `models/`

Repo này **không tự huấn luyện lại** mô hình — nó chỉ nạp các file `.pkl`/`.json` đã được
huấn luyện sẵn. Vì Render đọc code trực tiếp từ GitHub (không có nút "kéo-thả upload" như
Hugging Face Spaces), **các file mô hình phải được commit vào repo** trước khi push:

1. Mở `DSP391M_Report4_Final.ipynb` trên Google Colab, chạy tuần tự từ đầu đến hết
   **mục II.1 (Đóng gói Mô hình)** — cell cuối cùng của mục này sẽ tự động tạo và
   tải xuống file `models_package.zip`.
2. Giải nén `models_package.zip`, bạn sẽ có 9 file (xem danh sách trong
   `models/.gitkeep`).
3. Copy toàn bộ 9 file đó vào thư mục `models/` của repo này (đè lên `models/.gitkeep`
   nếu muốn dọn cho gọn — không bắt buộc phải xoá).
4. `git add`, `commit`, `push` lên GitHub (xem bước dưới) rồi mới tạo Web Service trên
   Render, hoặc nếu Web Service đã tồn tại, Render sẽ tự động redeploy sau khi push
   (đã bật `autoDeploy: true` trong `render.yaml`).

> Lưu ý dung lượng: mỗi file `.pkl` XGBoost thường chỉ vài trăm KB → vài MB, hoàn toàn
> đẩy lên GitHub bình thường (không cần Git LFS), miễn tổng dưới giới hạn 100MB/file
> của GitHub.

## 📤 Bước 1 — Đẩy code lên GitHub

```bash
cd github_render          # thư mục chứa toàn bộ file trong gói này (đã có models/ đầy đủ)
git init
git add .
git commit -m "DSP391m Nhom 8 - EV vs Gasoline TCO API"
git branch -M main
git remote add origin https://github.com/<username>/<ten-repo>.git
git push -u origin main
```
(Hoặc tạo repo trống trên github.com trước, rồi làm theo hướng dẫn "…or push an existing
repository from the command line" mà GitHub hiển thị.)

## 🚀 Bước 2 — Deploy trên Render (đăng nhập bằng GitHub)

**Cách A — Dùng Blueprint (`render.yaml`) — khuyến nghị, tự cấu hình hết:**
1. Vào https://dashboard.render.com → **New +** → **Blueprint**
2. Đăng nhập/liên kết tài khoản GitHub nếu chưa, chọn repo vừa push ở Bước 1
3. Render tự đọc `render.yaml` trong repo và điền sẵn: Runtime `Python`, gói **Free**,
   build command `pip install -r requirements.txt`, start command
   `uvicorn app:app --host 0.0.0.0 --port $PORT` → bấm **Apply**/**Deploy**
4. Đợi build xong (theo dõi tab **Logs**), service sẽ chạy tại
   `https://<tên-service>.onrender.com`

**Cách B — Tạo Web Service thủ công (nếu không muốn dùng Blueprint):**
1. Vào https://dashboard.render.com → **New +** → **Web Service**
2. Chọn repo GitHub vừa push
3. Điền:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** Free
4. Bấm **Create Web Service** → đợi build & deploy xong

> Muốn deploy bằng Docker thay vì Python runtime: ở Bước 2 chọn **Environment: Docker**,
> Render sẽ tự build theo `Dockerfile` có sẵn trong repo (không cần khai báo build/start
> command vì đã nằm trong Dockerfile).

## ℹ️ Lưu ý về gói Free của Render

- Free Web Service sẽ **tự "ngủ" (spin down) sau ~15 phút không có request nào** — request
  đầu tiên sau đó sẽ mất khoảng **~1 phút để "thức dậy"** (cold start), các request sau đó
  nhanh bình thường. Phù hợp để demo/nộp báo cáo, nhưng nếu cần chạy 24/7 không có độ trễ
  cần nâng cấp gói trả phí.
- Ổ đĩa là **ephemeral** (tạm thời) — mọi thay đổi file cục bộ (nếu có) sẽ mất khi service
  restart/redeploy. Vì mô hình được nạp từ file đã commit trong repo (không ghi file lúc
  chạy), điều này không ảnh hưởng tới API.
- Mỗi lần `git push`, vì `autoDeploy: true`, Render sẽ tự động build & deploy lại.

## 📡 Các endpoint chính

| Method | Endpoint | Mô tả |
|---|---|---|
| `GET`  | `/` | Trang demo (chọn ví dụ EV/Gasoline → gọi thử `/predict_tco`) |
| `GET`  | `/docs` | Swagger UI tự động của FastAPI |
| `GET`  | `/health` | Kiểm tra mô hình đã nạp thành công chưa (Render dùng endpoint này để healthcheck) |
| `GET`  | `/features` | Danh sách trường đầu vào cần thiết cho từng mô hình |
| `POST` | `/predict_price` | Dự đoán giá bán lại (`Price($)`) |
| `POST` | `/predict_consumption` | Dự đoán tiêu thụ năng lượng (kWh/100km) hoặc nhiên liệu (L/100km) |
| `POST` | `/predict_tco` | Ước tính khấu hao + chi phí năng lượng cộng dồn sau N năm (TCO) |

### Ví dụ gọi `/predict_tco` (sau khi đã có URL Render)
```bash
curl -X POST https://<ten-service>.onrender.com/predict_tco \\
  -H "Content-Type: application/json" \\
  -d '{
        "fuel_group": "EV",
        "car_features": {
          "CarAge": 3, "Mileage(km)": 45000, "Horsepower": 300, "Torque": 400,
          "Doors": 4, "Seats": 5, "NumOptions": 5, "Condition": "Used",
          "Transmission": "Automatic", "DriveType": "AWD", "BodyType": "SUV",
          "Brand": "Tesla", "AccidentHistory": "None", "Insurance": "Yes",
          "RegistrationStatus": "Registered", "City": "Hanoi"
        },
        "ownership_years": 5
      }'
```

## 🗂️ Cấu trúc thư mục

```
.
├── app.py               # FastAPI app (endpoints + trang demo HTML)
├── requirements.txt      # Thư viện Python cần cài
├── render.yaml            # Blueprint để Render tự cấu hình (Cách A ở Bước 2)
├── Dockerfile              # Tuỳ chọn, nếu muốn deploy bằng Docker thay vì Python runtime
├── README.md               # File này
└── models/                 # ⚠️ Cần copy 9 file .pkl/.json từ models_package.zip vào đây
```
