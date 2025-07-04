# Hotel MLOps Pipeline

An end-to-end, local MLOps pipeline for hotel booking predictions, including occupancy, rewards, and pricing models.

---

## Project Overview

This repository automates the training, evaluation, and prediction workflows for three sequential machine learning models:

1. **Occupancy Model**: Classifies booking occupancy into Low, Medium, or High.
2. **Rewards Model**: Classifies reward points category (Low, Medium, High) using occupancy predictions.
3. **Pricing Model**: Regresses the final price of a booking using predictions from the first two models and other features.

All models retrain on the latest 500 rows of historical data and predict on incoming bookings (`new_booking.xlsx`). Predictions and metadata (timestamps, metrics) are versioned and logged seamlessly.

---

## Prerequisites

- Python 3.8+
- `pip` package manager
- Required Python packages listed in `requirements.txt`

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/Hotel_MLOps.git
   cd Hotel_MLOps
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/macOS
   venv\Scripts\activate         # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

- \`\` (optional): Override default hyperparameters or file paths.
- **Retraining Toggle**: In `pipeline_runner.py`, set `RETRAIN = True` to retrain models; `False` to skip and use existing.

---

## Usage

### 1. Manual Run

Train models on historical data and predict on the sample booking:

```bash
python pipeline_runner.py --mode train_predict
```

### 2. Watch for New Bookings

Automatically trigger prediction when `new_booking.xlsx` is updated:

```bash
python file_watcher.py
```

---

## Outputs & Logs

- **Model Artifacts**: Saved under `models/{model_name}/` with timestamped folders and `latest` symlink.
- **Metrics Log** (`logs/metrics.log`): Records training metrics (accuracy, R², MSE) for each run.
- **Predictions Log** (`logs/predictions.log`): Records prediction outputs with timestamps.

---

## Contribution

1. Fork the repository
2. Create a new branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request

---

*Built by Armaan Sharma & Adarsh Kumar*

