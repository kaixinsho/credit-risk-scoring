import ctypes
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Server Vercel tidak punya libgomp (dibutuhkan LightGBM), jadi muat salinan sendiri dulu
if sys.platform.startswith("linux"):
    _gomp = ROOT / "lib" / "libgomp.so.1"
    if _gomp.exists():
        ctypes.CDLL(str(_gomp), mode=ctypes.RTLD_GLOBAL)

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

bundle = joblib.load(ROOT / "models" / "lgbm_v1.joblib")
model = bundle["model"]
categories = bundle["categories"]
features = bundle["features"]

app = FastAPI(title="Credit Risk Scoring API")


class Applicant(BaseModel):
    contract_type: str = "Cash loans"
    gender: str = "F"
    own_car: str = "N"
    own_realty: str = "Y"
    children: int = Field(default=0, ge=0, le=20)
    income_total: float = Field(gt=0)
    credit_amount: float = Field(gt=0)
    annuity: float = Field(gt=0)
    goods_price: float | None = None
    age_years: float = Field(ge=18, le=100)
    years_employed: float | None = None  # kosongkan jika tidak bekerja / pensiun
    income_type: str = "Working"
    education: str = "Secondary / secondary special"
    family_status: str = "Married"
    housing_type: str = "House / apartment"
    ext_source_1: float | None = Field(default=None, ge=0, le=1)
    ext_source_2: float | None = Field(default=None, ge=0, le=1)
    ext_source_3: float | None = Field(default=None, ge=0, le=1)


def nan(x):
    return np.nan if x is None else x


def to_frame(a: Applicant) -> pd.DataFrame:
    row = {col: np.nan for col in features}
    row.update({
        "NAME_CONTRACT_TYPE": a.contract_type,
        "CODE_GENDER": a.gender,
        "FLAG_OWN_CAR": a.own_car,
        "FLAG_OWN_REALTY": a.own_realty,
        "CNT_CHILDREN": a.children,
        "AMT_INCOME_TOTAL": a.income_total,
        "AMT_CREDIT": a.credit_amount,
        "AMT_ANNUITY": a.annuity,
        "AMT_GOODS_PRICE": nan(a.goods_price),
        "NAME_INCOME_TYPE": a.income_type,
        "NAME_EDUCATION_TYPE": a.education,
        "NAME_FAMILY_STATUS": a.family_status,
        "NAME_HOUSING_TYPE": a.housing_type,
        "DAYS_BIRTH": -a.age_years * 365,
        "AGE_YEARS": a.age_years,
        "DAYS_EMPLOYED": nan(None if a.years_employed is None else -a.years_employed * 365),
        "DAYS_EMPLOYED_ANOM": 1 if a.years_employed is None else 0,
        "EXT_SOURCE_1": nan(a.ext_source_1),
        "EXT_SOURCE_2": nan(a.ext_source_2),
        "EXT_SOURCE_3": nan(a.ext_source_3),
    })
    df = pd.DataFrame([row])[features]
    for col, cats in categories.items():
        df[col] = pd.Categorical(df[col], categories=cats)
    return df


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    page = ROOT / "web" / "index.html"
    if not page.exists():
        return HTMLResponse(
            "<p>Halaman web belum tersedia. Dokumentasi API ada di <a href='/docs'>/docs</a>.</p>",
            status_code=404,
        )
    return HTMLResponse(page.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {"status": "ok", "model_auc": round(bundle["auc"], 4)}


@app.post("/predict")
def predict(applicant: Applicant):
    proba = float(model.predict_proba(to_frame(applicant))[0, 1])
    return {"default_probability": round(proba, 4), "model_version": "v1"}
