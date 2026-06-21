# ParkSense AI — Illegal Parking Hotspot Detection & Congestion Impact Predictor

## Problem Statement
On-street illegal parking near commercial areas, metro stations, and junctions chokes
carriageways. Enforcement is patrol-based and reactive with no heatmap of violations
vs congestion impact and no way to prioritize enforcement zones.

## Our Solution
ParkSense AI is a two-module intelligence platform built on 298,450 real parking
violation records from Bengaluru Traffic Police (Nov 2023 – Apr 2024).

**Module 1 — Violation Intelligence Dashboard**
- Interactive heatmap of 1,788 high-density hotspot clusters
- Breakdown by violation type, vehicle type, police station load, and junction
- Monthly and hourly trend analysis

**Module 2 — Enforcement Prediction Engine**
- Predicts violation risk score for 12 major zones for the next 6 hours
- Uses 3 signals: hourly violation profile, day-of-week pattern, congestion multiplier
- Simulate any hour/day scenario interactively

## Files
| File | Description |
|---|---|
| `dashboard.html` | Full interactive dashboard — open in any browser, no server needed |
| `predict_demand.py` | ML model for demand prediction (Round 1 model) |
| `requirements.txt` | Python dependencies |

## How to Run

### Dashboard (no install needed)
```
Open dashboard.html in Chrome or any modern browser
```

### ML Model
```bash
pip install -r requirements.txt
python predict_demand.py
```
Place `train.csv` and `test.csv` in the same folder before running.

## Tech Stack
- **Frontend**: Vanilla HTML, CSS, JavaScript, Chart.js
- **ML**: Python, pandas, numpy, scikit-learn, XGBoost, LightGBM
- **Data**: 298,450 Bengaluru Traffic Police violation records (Nov 2023 – Apr 2024)

## Key Findings
- 50% of violations occur near junctions
- Safina Plaza & KR Market are the top hotspots
- 4–6 AM is the peak enforcement window (45% of all violations)
- Scooters + motorcycles = 46% of all violating vehicles
