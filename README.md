# Aegis — Enhanced Machine Health Monitoring Dashboard

## What is included
- Modern dark-navy hero homepage and redesigned KPI cards
- Executive fleet overview and priority queue
- Fleet risk register with filters and CSV export
- Engine lifecycle view with failure probability, RUL, health index, and engine comparison
- Sensor trend explorer and feature-importance chart
- Isolation Forest sensor anomaly detection
- Rule-based maintenance action report
- What-if alert-threshold scenario lab
- Group-wise model evaluation to reduce same-engine train/test leakage
- Dataset quality explorer

## Run locally
1. Keep `dashboard.py` and `requirements.txt` in your existing project folder.
2. Make sure your existing data file is located at `data/processed_data.csv`.
3. Open a terminal in that folder and run:

   ```bash
   pip install -r requirements.txt
   streamlit run dashboard.py
   ```

4. Streamlit will print a local URL (usually `http://localhost:8501`) to open in your browser.

## Expected columns
Required: `engine_id`, `cycle`, `RUL`, `failure_label`, plus numeric columns whose names include `sensor` or `setting`.

This version expects the already processed CSV from your existing project. It does not include the dataset because the dataset was not attached with the source code.

## Important limitations
- The NASA CMAPSS FD001 data is historical/simulated turbofan data. This app is not connected to live sensors.
- The health index is a project-defined score based on predicted RUL divided by 125.
- Isolation Forest flags unusual patterns; it does not diagnose a physical fault.
- Maintenance recommendations are academic triage suggestions, not operational safety instructions.
- Model metrics will differ from the previous version because evaluation is now grouped by engine to reduce data leakage.
