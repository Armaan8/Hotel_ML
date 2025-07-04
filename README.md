# Hotel MLOps Pipeline

An end-to-end, local MLOps pipeline for hotel booking predictions: occupancy classification, rewards tier classification, and price regression.

---

## Project Overview

This pipeline automates the complete lifecycle:

1. **Data Preparation**: Validate and preprocess raw booking data.
2. **Model Training**: Retrain three inter-dependent models on the latest 500 records:
   - **Occupancy**: Predict occupancy percentage, binned into Low/Medium/High classes.
   - **Rewards**: Predict reward tier (Low/Medium/High) using occupancy class.
   - **Pricing**: Predict final room rate (regression) using original features + predicted classes.
3. **Model Evaluation**: Log metrics (accuracy, R², MSE) and top features.
4. **Sequential Prediction**: When a new booking file appears, predict through all models.
5. **Versioning & Logging**: Save timestamped models and metrics, append predictions to master datasets, log human- and machine-readable summaries.
6. **Deployment**: Run locally via file-watcher; CI/CD validation via GitHub Actions.

---

## Directory Structure

```plaintext
├── data/                         # All input datasets
│   ├── raw/                      # Raw CSV/Excel files
│   │   ├── occupancy_master.csv  # Historical occupancy data (cols: booking_id, date, rooms_available, market_region, hotel_brand, lead_time, occupancy_pct)
│   │   ├── pricing_master.csv    # Combined rewards & pricing data (cols: booking_id, ... , total_points, final_price)
│   │   └── new_booking.xlsx      # Incoming booking(s) for prediction
│   └── processed/                # Processed slices for training (latest 500 rows)
├── models/                       # Saved model artifacts
│   ├── occupancy/                # Timestamped & latest occupancy model (.pkl)
│   ├── rewards/                  # Timestamped & latest rewards model
│   └── pricing/                  # Timestamped & latest pricing model
├── logs/                         # Logs and metrics
│   ├── pipeline.log              # Human-readable run summaries
│   ├── predictions.log           # Prediction outputs with timestamps
│   └── metrics/                  # JSON files with detailed metrics per run
├── src/                          # Source code modules
│   ├── data_loader.py            # Load, validate, and preprocess data
│   ├── feature_engineering.py    # Feature creation and encoding
│   ├── occupancy_model.py        # Training & prediction logic for occupancy
│   ├── rewards_model.py          # Training & prediction logic for rewards
│   ├── pricing_model.py          # Training & prediction logic for pricing
│   ├── pipeline_runner.py        # Orchestration: retrain + predict
│   └── utils.py                  # Helpers: logging, versioning, file I/O
├── file_watcher.py               # Monitors data/raw/new_booking.xlsx and triggers pipeline
├── config.yml                    # Hyperparameters, file paths, and toggles
├── Dockerfile                    # Defines container environment for pipeline
├── docker-compose.yml            # Defines service setup for Dockerized pipeline
├── .github/workflows/            # CI/CD: nightly validation & on-push tests
│   └── pipeline-ci.yml
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md                     # This file
```

---

## Prerequisites

- Python 3.8+
- `pip` package manager
- `DVC` installed (`pip install dvc`)
- Docker & Docker Compose (optional, for container builds)

---

## Installation & Setup

1. **Clone the repo**:
   ```bash
   git clone https://github.com/Armaan8/Hotel_ML.git
   cd Hotel_MLOps
   ```
2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate       # Windows
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Initialize DVC & Git** (first-time only):
   ```bash
   dvc init
   git add .dvc .gitignore
   git commit -m "Initialize DVC"
   ```
5. **Configure**:
   - Edit `config.yml` to set paths (e.g., `data/raw/`, `models/`) and toggles:
     ```yaml
     data:
       raw_dir: data/raw
       processed_dir: data/processed
     models_dir: models
     logs_dir: logs
     retrain_on_start: true
     ```

---

## Usage

### 1. Training & Prediction (Manual)

Run the full pipeline (retrain + predict) on `new_booking.xlsx`:
```bash
python src/pipeline_runner.py --mode train_predict
```

### 2. Prediction Only

Skip retraining; use existing models:
```bash
python src/pipeline_runner.py --mode predict_only
```

### 3. Real-Time Monitoring

Continuously watch for new bookings and auto-trigger:
```bash
python file_watcher.py
```

---

## Outputs & Artifacts

- **Processed Data**: `data/processed/occupancy_<timestamp>.csv`
- **Model Binaries**: `models/{occupancy,rewards,pricing}/{timestamp}.pkl` + `latest.pkl`
- **Metrics JSON**: `logs/metrics/metrics_<timestamp>.json`
- **Pipeline Log**: `logs/pipeline.log`
- **Predictions Log**: `logs/predictions.log`

---

## Detailed Steps

1. **Data Loading**: `data_loader.py` reads raw files, checks schema, imputes missing values.
2. **Feature Engineering**: `feature_engineering.py` encodes categoricals, scales numerics, creates lag features.
3. **Model Training**:
   - Pull the last 500 rows per domain from `data/processed`.
   - Train models with hyperparameters from `config.yml`.
   - Evaluate: occupancy accuracy, rewards classification report, pricing R² & MSE.
4. **Versioning**:
   - Save models with timestamp and symlink `latest.pkl`.
   - `dvc add` new artifacts and commit.
5. **Prediction Flow**:
   - Load `new_booking.xlsx`; apply same preprocessing & features.
   - Sequentially predict occupancy ➔ rewards ➔ pricing.
   - Append predictions to `data/raw/occupancy_master.csv` & `pricing_master.csv`.
6. **Logging**:
   - Append a summary line to `logs/pipeline.log`.
   - Save detailed metrics JSON.
   - Append raw predictions to `logs/predictions.log`.

---

## CI/CD Validation

- **GitHub Actions** (`.github/workflows/pipeline-ci.yml`):
  - On push & nightly at 02:00 UTC, pull latest DVC data, run `pipeline_runner.py --mode train_predict`, and push DVC pointers.
  - Alerts on failures via configured Slack/Webhook.

