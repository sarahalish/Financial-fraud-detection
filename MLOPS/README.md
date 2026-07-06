# AML/KYC Fraud Detection — Lightweight MLOps Pipeline

An Isolation-Forest-based transaction risk engine, refactored from a single
Jupyter Notebook into a modular, reproducible, pure-Python project. No Docker,
no Linux dependencies — runs on Windows and macOS inside VS Code.

## How it works

1. **Ingestion** — loads raw client and transaction CSVs and left-joins client
   profiles onto transactions.
2. **Preprocessing** — log-transforms transaction amounts (`log1p`) so small
   structured patterns (smurfing) aren't drowned out by volume, and fills
   missing values (median/mode).
3. **Behavioral engine** — an Isolation Forest (`max_samples=256` subsampling)
   fitted on `[amount_log, structuring, rapid_movement, trade_mispricing]`,
   producing a 0–1 behavioral risk score.
4. **Static risk** — mean of the binary compliance flags
   (PEP, sanctions, FATF country, OFAC country).
5. **Ensemble score** — `final_fraud_risk = 0.7 × behavioral + 0.3 × static`,
   with a dynamic alert threshold at the 95th percentile.
6. **Peer group analysis** — transaction amount vs. its sector's mean amount
   (`peer_deviation`) re-ranks alerts into a `refined_priority` queue,
   reducing false positives from legitimately high-volume sectors.
7. **Evaluation** — metrics + plots against a *proxy* ground truth
   (no verified labels exist in this synthetic dataset).

Everything needed for consistent inference — fitted scaler, fitted model,
score-normalization bounds, alert threshold, and sector baselines — is
serialized into **one joblib bundle** (`models/aml_model_bundle.joblib`).

## Project structure

```
aml-fraud-detection/
├── config/
│   └── config.yaml          # all paths, features & hyperparameters
├── data/
│   ├── raw/                 # input CSVs (clients + transactions)
│   └── processed/           # scored training data (generated)
├── logs/                    # pipeline.log (generated)
├── models/                  # serialized model bundle (generated)
├── reports/                 # metrics.json, plots, scored outputs (generated)
├── src/
│   ├── config.py            # YAML loader, path resolution
│   ├── logger.py            # console + file logging
│   ├── data_ingestion.py    # load & merge raw CSVs
│   ├── preprocessing.py     # feature engineering & NA handling
│   ├── train.py             # fit model, build & save bundle
│   ├── evaluate.py          # proxy labels, metrics, diagnostics figure
│   └── inference.py         # score new raw data with the saved bundle
├── run_pipeline.py          # entry point: full training pipeline
├── predict.py               # entry point: score new raw data (CLI)
├── requirements.txt
└── .gitignore
```

## Setup in VS Code (Windows / macOS)

1. Open the `aml-fraud-detection` folder in VS Code
   (**File → Open Folder…**).

2. Open the integrated terminal (**Terminal → New Terminal**) and create a
   virtual environment:

   **Windows (PowerShell):**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   (If activation is blocked, run once:
   `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)

   **macOS / Linux:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Tell VS Code to use the venv: press **Ctrl+Shift+P** (Cmd+Shift+P on Mac)
   → *Python: Select Interpreter* → choose `.venv`.

## Running the pipeline

Train, evaluate, and save the model bundle (from the project root):

```bash
python run_pipeline.py
```

Outputs:
- `models/aml_model_bundle.joblib` — the serialized model bundle
- `data/processed/scored_training_data.csv` — full scored training set
- `reports/metrics.json` and `reports/evaluation_plots.png`
- `logs/pipeline.log` — full traceability of the run

## Scoring new data

```bash
python predict.py --transactions path/to/new_transactions.csv --clients path/to/clients.csv --output reports/scored_new_data.csv
```

The output CSV contains `behavioral_risk_score`, `static_risk_score`,
`final_fraud_risk`, `is_fraud_alert`, `peer_deviation`, and
`refined_priority` for every transaction, and the terminal prints the
top-priority alert queue for compliance review.


## Interactive dashboard

After running the pipeline (and optionally `predict.py`), launch the
dashboard from the project root:

```bash
streamlit run dashboard.py
```

Your browser opens at `http://localhost:8501`. The dashboard reads the
saved CSVs in `reports/` and `data/processed/` — it never re-trains or
modifies anything. Features:

- switch between scored datasets (training vs. new predictions)
- filter by risk score, sector, peer deviation, alerts-only, or client ID
- risk distribution, alerts-by-sector, and amount-vs-risk charts
- sortable priority queue table with risk heat-coloring
- download the filtered view as CSV

Stop it with **Ctrl+C** in the terminal.

## Experimenting

All knobs live in `config/config.yaml` — no code changes needed:

- Isolation Forest hyperparameters (`n_estimators`, `max_samples`, …)
- ensemble weights (`behavioral_weight` / `static_weight`)
- alert percentile (95 → top 5% flagged)
- peer-deviation threshold for the refined queue
- feature lists for both engines

Change a value, re-run `python run_pipeline.py`, and compare
`reports/metrics.json` between runs.

## Notes & known limitations

- **Proxy labels:** evaluation uses a rule-based proxy ground truth
  (structuring/rapid-movement + PEP/sanctions/FATF). Replace with verified
  compliance labels before drawing real conclusions from the metrics.
- The dynamic threshold (95th percentile) is a silent default —
  domain experts should validate it before any production use.
- The original notebook referenced a non-existent `ofac_country_flag_tx`
  column; this refactor uses the correct post-merge names
  (`fatf_country_flag_tx` from transactions, `ofac_country_flag` from the
  client profile).
