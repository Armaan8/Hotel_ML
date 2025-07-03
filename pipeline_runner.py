# pipeline_runner.py
import os
import yaml  # type: ignore
import importlib
import joblib
from pathlib import Path
import json
from datetime import datetime
import pandas as pd

from src.data_loader import (
    ensure_master_files,
    load_last_500_rows,
    load_new_booking,
    append_to_master,
)
from src.utils import log_results

# ─── Paths & Config ─────────────────────────────────────────
CONFIG_PATH   = Path("config.yml")
DATA_DIR      = Path("data")
OCC_MASTER    = DATA_DIR / "occupancy_dataset.xlsx"
PRI_MASTER    = DATA_DIR / "pricing_dataset.xlsx"
NEW_BOOK_PATH = DATA_DIR / "new_booking.xlsx"
MODELS_DIR    = Path("models")
METRICS_DIR   = Path("metrics")


def run_pipeline():
    print("[Runner] Starting MLOps pipeline...")
    ensure_master_files()
    METRICS_DIR.mkdir(exist_ok=True)

    # 1️⃣ Load config
    try:
        cfg = yaml.safe_load(open(CONFIG_PATH))
        model_names = cfg["models"]
        print(f"Models to run: {model_names}")
    except Exception as e:
        print("❌ Failed to load config.yml:", e)
        return

    # 2️⃣ Load master data
    occ_df = load_last_500_rows(OCC_MASTER)
    pri_df = load_last_500_rows(PRI_MASTER)

    # 3️⃣ Train or load each model
    results = {}
    model_objs = {}  # store trained models here

    for name in model_names:
        print(f"   • Model: {name}")
        module = importlib.import_module(f"src.{name}_model")
        retrain_flag = os.getenv("RETRAIN_ON_START", "true").lower() == "true"
        latest_file = MODELS_DIR / name / "latest.pkl"

        if not retrain_flag and latest_file.exists():
            print(f"     – Skipping retrain (found latest.pkl)")
            model = joblib.load(latest_file)
            metrics = None
        else:
            df = occ_df if name == "occupancy" else pri_df
            model, metrics = module.train_and_save(df)
            print(f"     – Trained {name}: {metrics}")

        results[f"{name}_metrics"] = metrics
        model_objs[name] = model

    # 4️⃣ Save metrics JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_payload = {
        name: results[f"{name}_metrics"] for name in model_names
    }
    metrics_path = METRICS_DIR / f"{timestamp}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, default=str, indent=2)
    print(f"✅ Saved run metrics to {metrics_path}")

    # 5️⃣ Predict & append if booking file exists
    if NEW_BOOK_PATH.exists():
        booking = load_new_booking(NEW_BOOK_PATH)
        print("📥 Booking found → running predictions…")

        occ_mod = importlib.import_module("src.occupancy_model")
        rew_mod = importlib.import_module("src.rewards_model")
        pri_mod = importlib.import_module("src.pricing_model")

        occ_cls = occ_mod.predict(model_objs["occupancy"], booking)
        rew_cls = rew_mod.predict(model_objs["rewards"], booking)
        price   = pri_mod.predict(model_objs["pricing"], booking)

        print(f"   → Occ: {occ_cls}, Rew: {rew_cls}, Price: {price:.2f}")

        # ⬇ Append to occupancy master
        occ_row = occ_mod.make_occ_features(booking)
        occ_row["occ_class"] = occ_cls
        occ_row = occ_row.reindex(
            occ_df.columns.tolist() + ["occ_class"], axis=1, fill_value=pd.NA
        )
        append_to_master(OCC_MASTER, occ_row)

        # ⬇ Append to pricing master
        price_row = pri_mod.make_pricing_features(booking)
        price_row["points_class"] = rew_cls
        price_row["final_price"] = price
        price_row = price_row.reindex(
            pri_df.columns.tolist() + ["points_class", "final_price"],
            axis=1, fill_value=pd.NA
        )
        append_to_master(PRI_MASTER, price_row)

        # ✅ Log readable results
        log_results(
            occ_cls,
            rew_cls,
            price,
            results["occupancy_metrics"],
            results["rewards_metrics"],
            results["pricing_metrics"],
        )
        print("✅ Predictions appended and logged.")
    else:
        print("ℹ️ No new_booking.xlsx — training-only run.")

    print("🏁 [Runner] Finished.")


if __name__ == "__main__":
    run_pipeline()
