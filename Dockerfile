# (Tuỳ chọn) Dùng file này nếu muốn deploy trên Render bằng môi trường Docker thay vì
# Python runtime mặc định trong render.yaml. Khi tạo Web Service thủ công trên Render,
# chọn Environment: Docker — Render sẽ tự build từ Dockerfile này.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Render tự gán biến môi trường PORT khi chạy container — dùng shell form để $PORT
# được thay thế đúng lúc container khởi động. Mặc định 8000 khi chạy Docker cục bộ.
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
