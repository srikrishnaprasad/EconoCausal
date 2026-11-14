"""main.py - FastAPI app for EconoCausal.

Put this file at src/api/main.py and run from the project root:
    uvicorn src.api.main:app --reload
"""
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from src.api import services
from src.optimization.allocator import METHODS, allocate

ROOT = Path(__file__).resolve().parents[2]

app = FastAPI(title="EconoCausal API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def _dataset(dataset_id):
    try:
        return services.get(dataset_id)
    except KeyError:
        raise HTTPException(404, "Unknown dataset_id. Please load the data again.")


def _model(model):
    if model not in services.MODEL_LABELS:
        raise HTTPException(400, f"Unknown model '{model}'.")
    return model


def _level(k):
    if k not in (10, 20):
        raise HTTPException(400, "discount must be 10 or 20.")
    return k


def _scores(dataset_id, model):
    return _dataset(dataset_id)["scores"][_model(model)]


def _find_index():
    """Find the dashboard HTML file (edit INDEX_FILE below if yours is elsewhere)."""
    for rel in ("dashboard/index.html", "frontend/index.html", "static/index.html",
                "web/index.html", "templates/index.html", "src/dashboard/index.html",
                "src/frontend/index.html", "index.html", "dashboard.html"):
        if (ROOT / rel).is_file():
            return ROOT / rel
    for p in ROOT.rglob("*.html"):
        if any(part in (".venv", "node_modules", "site-packages") for part in p.parts):
            continue
        try:
            if "EconoCausal" in p.read_text(encoding="utf-8", errors="ignore")[:3000]:
                return p
        except OSError:
            pass
    return None


INDEX_FILE = _find_index()


# ------------------------------------------------------------------
# pages and status
# ------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def index():
    if INDEX_FILE is None:
        raise HTTPException(404, "Dashboard HTML file not found. Set INDEX_FILE in src/api/main.py.")
    return FileResponse(INDEX_FILE)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health-check")
def health_check():
    return services.health_checks()


# ------------------------------------------------------------------
# loading data
# ------------------------------------------------------------------
@app.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(400, f"Could not read the CSV: {e}")
    try:
        return services.register(df, file.filename or "uploaded.csv")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        raise HTTPException(500, str(e))


@app.post("/load-project")
def load_project(n: int = 0):
    if not services.RAW.exists():
        raise HTTPException(500, f"Project dataset not found: {services.RAW}")
    df = pd.read_csv(services.RAW)
    name = services.RAW.name
    if n and 0 < n < len(df):
        df = df.sample(n, random_state=42).reset_index(drop=True)
        name += f" (sample of {n:,})"
    try:
        return services.register(df, name)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(500, str(e))


# ------------------------------------------------------------------
# dashboard
# ------------------------------------------------------------------
@app.get("/summary")
def summary(dataset_id: str, model: str = "causal_forest"):
    return services.summary(_dataset(dataset_id), _model(model))


@app.get("/ite-hist")
def ite_hist(dataset_id: str, model: str = "causal_forest", discount: int = 20):
    return services.ite_hist(_scores(dataset_id, model), _level(discount))


@app.get("/uplift-curve")
def uplift_curve(source: str = "predicted", dataset_id: Optional[str] = None,
                 model: str = "causal_forest", discount: int = 20):
    _model(model)
    _level(discount)
    if source == "validated":
        try:
            return services.validated_curve(model, discount)
        except Exception as e:
            raise HTTPException(404, f"Held-out test results not available: {e}")
    return services.predicted_curve(_scores(dataset_id, model), discount)


# ------------------------------------------------------------------
# causal analysis, customers, monitoring
# ------------------------------------------------------------------
@app.get("/model-compare")
def model_compare():
    return services.model_compare()


@app.get("/propensity")
def propensity():
    try:
        return services.propensity_info()
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.get("/customers")
def customers(dataset_id: str, model: str = "causal_forest", category: str = "All",
              age_group: str = "All", search: str = "", sort: str = "ite_20",
              desc: bool = True, page: int = 1, page_size: int = 25,
              profit_per_sale: float = 100.0):
    return services.customers(_dataset(dataset_id), _model(model), category, age_group,
                              search, sort, desc, page, page_size, profit_per_sale)


@app.get("/drift")
def drift(dataset_id: str):
    return services.drift_report(_dataset(dataset_id)["raw"])


# ------------------------------------------------------------------
# optimization
# ------------------------------------------------------------------
class OptRequest(BaseModel):
    dataset_id: str
    model: str = "causal_forest"
    method: str = "lagrangian"
    budget: float = 20000
    profit_per_sale: float = 100
    levels: List[int] = [10, 20]


def _prepare(req: OptRequest):
    sc = _scores(req.dataset_id, req.model)
    levels = tuple(sorted(set(req.levels)))
    if not levels or any(k not in (10, 20) for k in levels):
        raise HTTPException(400, "Allowed discounts must be $10 and/or $20.")
    if req.method not in METHODS:
        raise HTTPException(400, f"Unknown method '{req.method}'.")
    return sc, levels


def _cols(df):
    keep = ["user_id", "discount", "p0", "ite_10", "ite_20", "expected_cost", "incremental_profit"]
    return [c for c in keep if c in df.columns]


@app.post("/optimize")
def optimize(req: OptRequest):
    sc, levels = _prepare(req)
    out, totals = allocate(sc, max(req.budget, 0.0), levels, req.profit_per_sale, req.method)
    targeted = out[out["discount"] > 0].sort_values("incremental_profit", ascending=False)
    services.RESULTS[req.dataset_id] = targeted
    totals["allocation"] = services._clean(targeted.head(300)[_cols(targeted)].round(5))
    return totals


@app.post("/optimize/compare")
def optimize_compare(req: OptRequest):
    sc, levels = _prepare(req)
    rows = []
    for m in METHODS:
        _, t = allocate(sc, max(req.budget, 0.0), levels, req.profit_per_sale, m)
        rows.append({"method": m, "label": METHODS[m], "profit": t["incremental_profit"],
                     "cost": t["total_cost"], "targeted": t["n_targeted"],
                     "extra_purchases": t["extra_purchases"], "seconds": t["seconds"],
                     "note": t["note"]})
    return rows


@app.post("/optimize/sweep")
def optimize_sweep(req: OptRequest):
    sc, levels = _prepare(req)
    # the exact MILP is too slow to repeat 10 times, so the sweep uses the fast Lagrangian method
    full = allocate(sc, 1e12, levels, req.profit_per_sale, "lagrangian")[1]["total_cost"]
    points = []
    for frac in np.linspace(0.1, 1.0, 10):
        b = float(full * frac)
        _, t = allocate(sc, b, levels, req.profit_per_sale, "lagrangian")
        points.append({"budget": round(b, 2), "profit": t["incremental_profit"], "cost": t["total_cost"]})
    return {"points": points}


@app.get("/allocation.csv")
def allocation_csv(dataset_id: str):
    df = services.RESULTS.get(dataset_id)
    if df is None:
        raise HTTPException(404, "No allocation yet. Run the optimization first.")
    return Response(df[_cols(df)].to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="allocation.csv"'})


# ------------------------------------------------------------------
# report files (docs folder)
# ------------------------------------------------------------------
@app.get("/files/{name}")
def files(name: str):
    docs = services.DOCS.resolve()
    p = (docs / name).resolve()
    if p.parent != docs or not p.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(p)