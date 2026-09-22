# Reservoir Pressure Forecasting

Time-series forecasting of reservoir pressure using production and injection data from the Volve field. This project is part of a capstone for **Machine Learning for Petroleum Engineers & Geoscientists** and investigates whether an LSTM can outperform simpler forecasting baselines.

## Project question

> How well can we forecast next month's reservoir pressure using production and injection data?

The target is `Pressure_psia`. The notebook derives monthly oil-production and water-injection rates from cumulative volumes, creates lagged features, and evaluates one-month-ahead forecasts using a chronological train/test split.

## Repository structure

```text
.
├── data/
│   └── volve_field_production.csv       # Monthly Volve field data
├── notebooks/
│   └── capstone.ipynb                   # EDA, baselines, and model experiments
├── results/
│   └── capstone_results_log.csv         # Experiment metrics
├── requirements.txt
└── README.md
```

## Dataset

`data/volve_field_production.csv` contains 112 monthly observations from September 2007 through December 2016. The source fields include:

- Average reservoir pressure (`Pressure_psia`)
- Cumulative oil production (`CumOil_STB`)
- Cumulative gas production (`CumGas_SCF`)
- Cumulative water production (`CumWater_STB`)
- Gas injection (`GasInj_SCF`)
- Water injection (`WaterInj_STB`)
- Gas-oil ratio (`GOR_scf_per_stb`)

The notebook converts cumulative oil and water volumes into monthly rates and uses pressure, oil rate, and water-injection rate as the principal drivers. The data is a public, community-processed version of the Volve reservoir simulation history-match output.

## Models and evaluation

The notebook currently includes:

- **Persistence baseline:** predicts the next pressure value using the previous month’s pressure.
- **Random Forest — pressure lags only.**
- **Random Forest — pressure lags plus production/injection drivers.**
- **LSTM - Level Pressure.**
- **LSTM - Pressure Change.**

The evaluation uses the final 18 months as a held-out test period and reports RMSE and MAE in psia. The current logged baseline results are:

| Model | RMSE (psia) | MAE (psia) |
|---|---:|---:|
| Persistence | 85.3 | 49.5 |
| Random Forest — pressure lags only | 92.8 | 68.1 |
| Random Forest — pressure + drivers | 95.1 | 68.4 |
| LSTM - level pressure | 87.5 | 67.2 |
| LSTM - pressure change | 79.4 | 51.2|

The persistence model was the current benchmark to beat. 
The LSTM predicting pressure change beat the persistence model in regards with RMSE and is comparable by MAE.
Results are recorded in `results/capstone_results_log.csv`.

## Installation

Python 3.9 or newer is recommended.

1. Clone the repository and enter the project directory:

   ```bash
   git clone https://github.com/sadaraka-kadi/ml4pe-pressure-forecasting.git
   cd ml4pe-pressure-forecasting
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # macOS/Linux
   # .venv\\Scripts\\activate   # Windows PowerShell
   ```

3. Install the dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Running the notebook

Launch Jupyter from the repository root:

```bash
jupyter notebook
```

Open `notebooks/capstone.ipynb` and run the cells from top to bottom. Because the notebook uses paths relative to the `notebooks/` directory, running it with a Jupyter working directory set to `notebooks/` is recommended. Alternatively, update `DATA_DIR` and `LOG_PATH` if your environment uses a different working directory.

The notebook uses a fixed random seed (`42`) for reproducibility. It writes experiment results to `results/capstone_results_log.csv`.

## Notes and limitations

- This is a small time-series dataset, so an LSTM may overfit easily.
- The chronological split is intentional: future observations must not be used to train the model or fit preprocessing steps.
  
## License

See [LICENSE](LICENSE).
