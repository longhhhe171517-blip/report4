"""
DSP391m - Nhóm 8
API dự đoán giá bán lại, mức tiêu thụ năng lượng/nhiên liệu, và
Total Cost of Ownership (TCO) cho xe điện (EV) và xe xăng (Gasoline).

Nạp các mô hình đã huấn luyện (.pkl) và metadata.json được xuất ra từ notebook
DSP391M_Report4_Final.ipynb (mục II. Đóng gói và Triển khai).
"""

import json
import os
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

EV_FEATURES = [
    "CarAge", "Mileage(km)", "Horsepower", "Torque", "Doors", "Seats", "NumOptions",
    "Condition", "Transmission", "DriveType", "BodyType", "Brand",
    "AccidentHistory", "Insurance", "RegistrationStatus", "City",
]
GAS_FEATURES = [
    "CarAge", "Mileage(km)", "EngineSize(L)", "Horsepower", "Torque",
    "FuelEfficiency(L/100km)", "Doors", "Seats", "NumOptions",
    "Condition", "Transmission", "DriveType", "BodyType", "Brand",
    "AccidentHistory", "Insurance", "RegistrationStatus", "City",
]
BEV_FEATURES = ["Motor (kW)", "Range (km)", "Recharge time (h)", "Vehicle class", "Model year"]
GASFUEL_FEATURES = ["Engine size (L)", "Cylinders", "Fuel type", "Vehicle class", "Smog rating"]


# ──────────────────────────────────────────────────────────────────────────
# Nạp mô hình + encoders + metadata khi ứng dụng khởi động
# ──────────────────────────────────────────────────────────────────────────
def _load(filename, required=True):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"Thiếu file '{filename}' trong thư mục models/.")
        return None
    if filename.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return joblib.load(path)


_startup_error: Optional[str] = None
try:
    xgb_ev = _load("xgb_ev_price.pkl")
    xgb_gas = _load("xgb_gas_price.pkl")
    xgb_bev = _load("xgb_bev_consumption.pkl")
    xgb_gasfuel = _load("xgb_gasfuel_consumption.pkl")

    encoders_ev = _load("encoders_ev.pkl")
    encoders_gas = _load("encoders_gas.pkl")
    encoder_bev_vc = _load("encoder_bev_vehicle_class.pkl")
    encoders_gasfuel = _load("encoders_gasfuel.pkl")  # {'fuel_type':..., 'vehicle_class':...}

    metadata = _load("metadata.json")
except Exception as exc:  # noqa: BLE001
    _startup_error = (
        f"{exc}. Hãy chạy notebook DSP391M_Report4_Final.ipynb, tải models_package.zip "
        f"và giải nén toàn bộ nội dung vào thư mục models/ của repo này, commit + push lên "
        f"GitHub rồi để Render tự deploy lại."
    )
    xgb_ev = xgb_gas = xgb_bev = xgb_gasfuel = None
    encoders_ev = encoders_gas = encoder_bev_vc = encoders_gasfuel = None
    metadata = {}


def _check_ready():
    if _startup_error:
        raise HTTPException(status_code=503, detail=_startup_error)


def _encode_row(car_features: dict, feats: list, encoders: dict) -> pd.DataFrame:
    missing = [f for f in feats if f not in car_features]
    if missing:
        raise HTTPException(status_code=422, detail=f"Thiếu các trường bắt buộc: {missing}")
    row = pd.DataFrame([car_features])[feats].copy()
    for col, le in encoders.items():
        if col in row.columns:
            try:
                row[col] = le.transform(row[col].astype(str))
            except ValueError as exc:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Giá trị '{row[col].iloc[0]}' của trường '{col}' chưa từng xuất hiện "
                        f"lúc huấn luyện. Các giá trị hợp lệ: {list(le.classes_)}"
                    ),
                ) from exc
    return row


# ──────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DSP391m Nhóm 8 — EV vs Gasoline Cost API",
    description=(
        "API dự đoán giá bán lại, mức tiêu thụ năng lượng/nhiên liệu, và "
        "Total Cost of Ownership (TCO) cho xe điện (EV) và xe xăng (Gasoline)."
    ),
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


class PriceInput(BaseModel):
    fuel_group: Literal["EV", "Gasoline"]
    car_features: dict = Field(..., description="Xem GET /features để biết danh sách trường cần thiết")


class ConsumptionInput(BaseModel):
    fuel_group: Literal["EV", "Gasoline"]
    car_features: dict


class TcoInput(BaseModel):
    fuel_group: Literal["EV", "Gasoline"]
    car_features: dict
    ownership_years: int = Field(5, ge=1, le=20)


@app.get("/health")
def health():
    return {"status": "ok" if not _startup_error else "error", "detail": _startup_error}


@app.get("/features")
def features():
    return {
        "EV_FEATURES": EV_FEATURES,
        "GAS_FEATURES": GAS_FEATURES,
        "BEV_FEATURES": BEV_FEATURES,
        "GASFUEL_FEATURES": GASFUEL_FEATURES,
    }


@app.post("/predict_price")
def predict_price(payload: PriceInput):
    """Dự đoán giá bán lại (Price($)) dựa trên đặc trưng xe cũ."""
    _check_ready()
    if payload.fuel_group == "EV":
        model, feats, encoders = xgb_ev, EV_FEATURES, encoders_ev
    else:
        model, feats, encoders = xgb_gas, GAS_FEATURES, encoders_gas
    row = _encode_row(payload.car_features, feats, encoders)
    price = float(model.predict(row)[0])
    return {"fuel_group": payload.fuel_group, "predicted_resale_price_usd": round(price, 0)}


@app.post("/predict_consumption")
def predict_consumption(payload: ConsumptionInput):
    """Dự đoán mức tiêu thụ năng lượng (kWh/100km) hoặc nhiên liệu (L/100km)."""
    _check_ready()
    if payload.fuel_group == "EV":
        row = _encode_row(payload.car_features, BEV_FEATURES, {"Vehicle class": encoder_bev_vc})
        value = float(xgb_bev.predict(row)[0])
        return {"fuel_group": "EV", "predicted_kwh_per_100km": round(value, 2)}
    else:
        row = _encode_row(
            payload.car_features, GASFUEL_FEATURES,
            {"Fuel type": encoders_gasfuel["fuel_type"], "Vehicle class": encoders_gasfuel["vehicle_class"]},
        )
        value = float(xgb_gasfuel.predict(row)[0])
        return {"fuel_group": "Gasoline", "predicted_liters_per_100km": round(value, 2)}


@app.post("/predict_tco")
def predict_tco(payload: TcoInput):
    """
    Ước tính Total Cost of Ownership (TCO) sau N năm cho MỘT chiếc xe cụ thể:
    - Khấu hao = giá dự đoán ở tuổi xe hiện tại − giá dự đoán ở tuổi xe (hiện tại + N năm)
    - Chi phí năng lượng/nhiên liệu = mức tiêu thụ TRUNG VỊ của cả nhóm (EV/Gasoline, từ
      metadata — xem README) × số km/năm × giá điện hoặc giá xăng.
    """
    _check_ready()
    fuel_group = payload.fuel_group
    if fuel_group == "EV":
        model, feats, encoders = xgb_ev, EV_FEATURES, encoders_ev
        median_energy_100km = metadata["median_energy_consumption"]["ev_kwh_100km"]
        price_per_unit = metadata["ASSUMPTIONS"]["electricity_price_usd_per_kwh"]
    else:
        model, feats, encoders = xgb_gas, GAS_FEATURES, encoders_gas
        median_energy_100km = metadata["median_energy_consumption"]["gas_l_100km"]
        price_per_unit = metadata["ASSUMPTIONS"]["gasoline_price_usd_per_l"]

    if "CarAge" not in payload.car_features:
        raise HTTPException(status_code=422, detail="Thiếu trường 'CarAge'.")

    row_now = _encode_row(payload.car_features, feats, encoders)
    price_now = float(model.predict(row_now)[0])

    future_features = dict(payload.car_features)
    future_features["CarAge"] = payload.car_features["CarAge"] + payload.ownership_years
    row_future = _encode_row(future_features, feats, encoders)
    price_future = float(model.predict(row_future)[0])

    depreciation_loss = max(price_now - price_future, 0.0)
    annual_mileage = metadata["ASSUMPTIONS"]["annual_km"]
    annual_energy_cost = (median_energy_100km / 100) * annual_mileage * price_per_unit
    total_energy_cost = annual_energy_cost * payload.ownership_years
    tco = depreciation_loss + total_energy_cost

    return {
        "fuel_group": fuel_group,
        "ownership_years": payload.ownership_years,
        "predicted_resale_price_now_usd": round(price_now, 0),
        "predicted_resale_price_after_usd": round(price_future, 0),
        "depreciation_loss_usd": round(depreciation_loss, 0),
        "annual_energy_cost_usd": round(annual_energy_cost, 0),
        "total_energy_cost_usd": round(total_energy_cost, 0),
        "estimated_tco_usd": round(tco, 0),
    }


# ──────────────────────────────────────────────────────────────────────────
# Trang demo đơn giản (HTML thuần, không cần thư viện ngoài)
# ──────────────────────────────────────────────────────────────────────────
_EXAMPLES = {
    "EV": {
        "fuel_group": "EV",
        "car_features": {
            "CarAge": 3, "Mileage(km)": 45000, "Horsepower": 300, "Torque": 400,
            "Doors": 4, "Seats": 5, "NumOptions": 5, "Condition": "Used",
            "Transmission": "Automatic", "DriveType": "AWD", "BodyType": "SUV",
            "Brand": "Tesla", "AccidentHistory": "None", "Insurance": "Yes",
            "RegistrationStatus": "Registered", "City": "Hanoi",
        },
        "ownership_years": 5,
    },
    "Gasoline": {
        "fuel_group": "Gasoline",
        "car_features": {
            "CarAge": 3, "Mileage(km)": 45000, "EngineSize(L)": 2.0, "Horsepower": 180,
            "Torque": 220, "FuelEfficiency(L/100km)": 7.5, "Doors": 4, "Seats": 5,
            "NumOptions": 5, "Condition": "Used", "Transmission": "Automatic",
            "DriveType": "FWD", "BodyType": "Sedan", "Brand": "Toyota",
            "AccidentHistory": "None", "Insurance": "Yes",
            "RegistrationStatus": "Registered", "City": "Hanoi",
        },
        "ownership_years": 5,
    },
}

HTML_PAGE = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8" />
<title>DSP391m Nhóm 8 — EV vs Gasoline TCO API</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 780px; margin: 32px auto; padding: 0 16px; color: #1f2937; }}
  h1 {{ font-size: 1.4rem; }}
  .badge {{ display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.75rem; margin-right: 6px; }}
  .ev {{ background:#d1fae5; color:#065f46; }}
  .gas {{ background:#fee2e2; color:#991b1b; }}
  textarea {{ width: 100%; min-height: 220px; font-family: monospace; font-size: 0.85rem; box-sizing: border-box; padding: 8px; }}
  select, button {{ font-size: 0.95rem; padding: 6px 12px; margin: 8px 4px 8px 0; }}
  button {{ cursor: pointer; background:#111827; color:white; border:none; border-radius: 6px; }}
  button:hover {{ background:#374151; }}
  pre#result {{ background:#0f172a; color:#e2e8f0; padding: 14px; border-radius: 8px; overflow-x:auto; min-height: 60px; }}
  .links a {{ margin-right: 14px; }}
  code {{ background:#f1f5f9; padding: 1px 5px; border-radius: 4px; }}
</style>
</head>
<body>
  <h1>🚗 DSP391m — Nhóm 8: EV <span class="badge ev">EV</span> vs Gasoline <span class="badge gas">Gasoline</span> Cost API</h1>
  <p>API dự đoán giá bán lại, mức tiêu thụ năng lượng/nhiên liệu và Total Cost of Ownership (TCO)
  cho xe điện và xe xăng, đóng gói từ notebook <code>DSP391M_Report4_Final.ipynb</code>.</p>
  <p class="links">
    <a href="/docs" target="_blank">📄 Swagger UI (/docs)</a>
    <a href="/health" target="_blank">💚 /health</a>
    <a href="/features" target="_blank">📋 /features</a>
  </p>

  <h3>Thử nhanh endpoint <code>/predict_tco</code></h3>
  <label>Chọn ví dụ mẫu:
    <select id="example">
      <option value="EV">Xe điện (EV)</option>
      <option value="Gasoline">Xe xăng (Gasoline)</option>
    </select>
  </label>
  <br/>
  <textarea id="payload"></textarea><br/>
  <button onclick="run()">▶ Gửi request /predict_tco</button>
  <pre id="result">Kết quả sẽ hiển thị ở đây…</pre>

<script>
const EXAMPLES = {json.dumps(_EXAMPLES, ensure_ascii=False)};
const ta = document.getElementById('payload');
const sel = document.getElementById('example');
function loadExample() {{ ta.value = JSON.stringify(EXAMPLES[sel.value], null, 2); }}
sel.addEventListener('change', loadExample);
loadExample();

async function run() {{
  const out = document.getElementById('result');
  out.textContent = 'Đang gọi API…';
  try {{
    const body = JSON.parse(ta.value);
    const res = await fetch('/predict_tco', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify(body),
    }});
    const data = await res.json();
    out.textContent = res.status + ' ' + JSON.stringify(data, null, 2);
  }} catch (e) {{
    out.textContent = 'Lỗi: ' + e;
  }}
}}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE
