# Reservoir Pressure Forecasting

One-month-ahead forecasting of field-average reservoir pressure from production and injection history at the Volve field. This is the capstone project (Track C) for **Machine Learning for Petroleum Engineers & Geoscientists**.

## Project question

> How well can we forecast next month's reservoir pressure using production and injection data?

The target is `Pressure_psia` (psia). A model is considered useful only if it beats **persistence** (next month's pressure = this month's) by a margin that survives resampling.

The full write-up is in [`report/report.pdf`](report/report.pdf) and the slides are in [`report/presentation.pdf`](report/presentation.pdf).

## Key findings

- **Persistence is a hard baseline.** It scores RMSE 85.3 psia and MAE 49.5 psia on the 18 held-out months.
- **Predicting the monthly pressure change (ΔP) fixes the Random Forest problem.** Random Forests that predict the pressure *level* lose to persistence by 9-12% because trees cannot extrapolate beyond the pressures seen in training.
- **The best models tie persistence.** Ridge on ΔP with drivers reaches RMSE 82.0 psia (3.9% lower), but the gain is within sampling noise: a paired block bootstrap over the test months gives a 90% interval of -3.4% to +6.4%, and MAE is not better (50.3 vs 49.5 psia).
- **One event drives the error.** The 2016-09 and 2016-10 shut-in (+277 and +151 psia) carries most of the squared error, and no model anticipates it, because the training period contains no shut-in.
- **Uncertainty.** An empirical 90% prediction interval of about ±70 psia (from walk-forward errors on the training months) covers 16 of 18 held-out months, and all months outside the shut-in.
- **Negative results are reported.** Forecasting oil rate instead of pressure, adding drivers to the LSTM, and six-month recursive forecasts did not beat a flat line.

## Repository structure

```text
.
├── data/
│   └── volve_field_production.csv       # Monthly Volve field data
├── notebooks/
│   └── capstone.ipynb                   # Full analysis: EDA, baselines, experiments, uncertainty
├── report/
│   ├── report.pdf                       # Written report (4-6 pages) with model card appendix
│   └── presentation.pdf                 # Final presentation slides
├── results/
│   ├── capstone_results_log.csv         # Every experiment, including the ones that failed
│   └── figures/                         # Figures saved by the notebook (adjust to your FIG_PATH)
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

## Dataset

`data/volve_field_production.csv` has 112 monthly observations (2007-09 to 2016-12) and 8 columns, with no missing values:

- `Pressure_psia`: average reservoir pressure
- `CumOil_STB`, `CumGas_SCF`, `CumWater_STB`: cumulative production
- `GasInj_SCF`: gas injection (zero throughout, so unused)
- `WaterInj_STB`: cumulative water injection
- `GOR_scf_per_stb`: gas-oil ratio

The data is a public, community-processed version (`yohanesnuwara/pyreservoir`) of Equinor's Volve reservoir-simulation history-match output, so these are **simulation-derived field averages, not raw gauge measurements**. Production starts in 2008-02 and water injection in 2008-04.

The notebook turns the cumulative columns into monthly oil rate and water injection rate (MMSTB/month) by differencing, then uses pressure, oil rate and water injection rate as inputs.

## Method

- **Split:** chronological. 88 training months (2008-03 to 2015-06) and 18 test months (2015-07 to 2016-12). The first six months only supply lag history. A shuffled split would leak neighbouring months into training.
- **Target:** the monthly pressure change, ΔP = P(t) - P(t-1). The forecast is the last known pressure plus the predicted change, which makes persistence the special case ΔP = 0.
- **Inputs:** six months of lagged pressure changes, optionally with six months of lagged oil rate and water injection rate.
- **Models:** persistence, Random Forest (on the pressure level), Ridge regression (on ΔP) and a small LSTM (one layer, 16 hidden units; mean of five fixed seeds).
- **Leakage control:** scalers are fitted on training months only, and lag length and regularisation are chosen by walk-forward cross-validation (`TimeSeriesSplit`) on the training months only. The test months are used to report results, not to choose models.
- **Metrics:** RMSE (primary) and MAE, both in psia.

## Results

All rows use the same split (the last 18 months held out). Percentages are RMSE relative to persistence.

| Model | RMSE (psia) | MAE (psia) | vs persistence |
|---|---:|---:|---:|
| Persistence (baseline) | 85.3 | 49.5 | 0.0% |
| Random Forest, pressure lags only | 92.8 | 68.1 | -8.8% |
| Random Forest, pressure + drivers | 95.1 | 68.4 | -11.5% |
| Ridge on ΔP, pressure only (6 lags) | 85.5 | 49.4 | -0.2% |
| **Ridge on ΔP, pressure + drivers (6 lags)** | **82.0** | **50.3** | **+3.9%** |
| LSTM on ΔP, pressure only (6 lags) | 81.1 | 51.8 | +4.9% |
| LSTM on ΔP, pressure + drivers (6 lags) | 82.0 | 57.1 | +3.9% |
| LSTM on ΔP, pressure + drivers (12 lags, best LSTM by training CV) | 90.7 | 72.3 | -6.3% |
| LSTM on ΔP, pressure only (3 lags, lowest test RMSE, not chosen by CV) | 79.0 | 49.7 | +7.4% |

Ridge with drivers is reported as the final model because it is the best Ridge variant by training CV, it is deterministic, and with only 88 training samples a simple linear model is the lower-risk choice. This is a judgement about simplicity and stability, **not** evidence of higher accuracy: its RMSE is within noise of persistence, and a shared walk-forward CV on the training months ranks persistence first.

The full set of runs, including the lag grid, oil-rate and recursive experiments, is in `results/capstone_results_log.csv`.

## Reproducing the results

Python 3.11 was used (the pinned versions in `requirements.txt` were run on Python 3.11.15).

1. Clone the repository:

   ```bash
   git clone https://github.com/sadaraka-kadi/ml4pe-pressure-forecasting.git
   cd ml4pe-pressure-forecasting
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # macOS/Linux
   .venv\Scripts\activate           # Windows PowerShell
   ```

3. Install the dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Launch Jupyter, open `notebooks/capstone.ipynb`, and run **Kernel -> Restart & Run All**:

   ```bash
   jupyter notebook
   ```

The notebook uses paths relative to the `notebooks/` directory (`DATA_DIR = "../data"`, `LOG_PATH = "../results/..."`), so start Jupyter from the repository root and open the notebook from there, or edit those two variables if your working directory differs. A global seed (`42`) is set, and each LSTM result is the mean of five fixed seeds. The LSTM experiments take the longest to run. Individual LSTM runs vary with the seed, so small differences in the LSTM rows are expected on other hardware.

## Limitations

- Small sample: 112 monthly points, 88 for training and 18 for testing. The conclusion rests on one 18-month window, and two shut-in months determine most of the test error.
- Simulation-derived field averages, so the series is smoother than gauge data and results may not transfer to measured data or other fields.
- Persistence is not clearly beaten, and the LSTM is unstable at this data size.
- Forecasts are one month ahead. Recursive six-month forecasts, which hold the drivers constant, did no better than a flat line.
- The prediction interval assumes future months resemble the training months. It failed in the two shut-in months that did not.

## Next steps

- Add an operational flag for planned shut-ins or rate cuts.
- Evaluate with rolling-origin backtests over the whole history rather than a single window.
- Test on well-level pressure data (Volve gauge records) and other fields.

## Data source and license

Data: Equinor's public Volve dataset, in the community-processed form from [`yohanesnuwara/pyreservoir`](https://github.com/yohanesnuwara/pyreservoir), subject to the Equinor Volve data license. Code: see [LICENSE](LICENSE).
