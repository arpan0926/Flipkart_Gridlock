# Flipkart Gridlock - Traffic Demand Forecasting Pipeline

## 🎯 Project Overview
This repository contains a highly optimized, competition-grade Machine Learning pipeline designed to predict traffic demand for the Flipkart Gridlock Hackathon. The architecture leverages spatio-temporal feature engineering and a tuned LightGBM regressor to accurately map complex geographical and time-series traffic patterns.

---

## 📁 Repository Structure
The project is built using a modular, pip-installable package structure to ensure clean imports and scalability.

```text
Flipkart_Gridlock/
├── dataset/                    # Directory for raw training and test data (ignored by git)
│   ├── train.csv
│   └── test.csv
├── scripts/
│   └── run_analysis.py         # The main execution script (handles data flow and predictions)
├── src/
│   └── Model/                  # Core ML logic (installed as a local package)
│       ├── __init__.py
│       ├── data_loader.py      # Handles reading and validating the CSV files
│       ├── pipeline.py         # The mathematical engine (preprocessing, feature engineering, model)
│       └── model.py            # Legacy/Helper model classes
├── tests/                      # Unit tests for the pipeline components
├── pyproject.toml              # Package configuration for local installation
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

⚙️ Core Components & Architecture
1. Data Loader (src/Model/data_loader.py) Uses pathlib to dynamically locate the dataset/ folder, ensuring the code runs flawlessly on any teammate's machine regardless of where they cloned the repository.
  
2.  Feature Engineering & Pipeline (src/Model/pipeline.py) This is the brain of the project. It explicitly handles spatio-temporal data without causing data leakage.
   Cyclical Time Encoding: Converts linear time (hour, day_of_week) into continuous circles using sine and cosine functions. This allows the model to understand  that Midnight and 11:00 PM are continuous.

   Geohash Decoding: Translates alphanumeric geohash strings into raw lat and lon coordinates using the pygeohash library.
   Native Categorical Handling: Passes text columns (RoadType, Weather, geohash) directly to LightGBM without One-Hot Encoding, utilizing LightGBM's highly   optimized categorical splitting algorithm.
   
   Explicit Domain Flags: Generates specific flags like is_rush_hour to explicitly hand the model human-behavior context.

3. The ModelThe pipeline uses LightGBM (LGBMRegressor). It is configured to handle medium-sized datasets (~60k rows) while preventing overfitting through controlled leaf structures and subsampling mechanisms.
   
5. The Execution Script (scripts/run_analysis.py)The orchestrator script. It enforces Chronological Sorting before splitting the train/validation data. This is critical for time-series forecasting to ensure the model never looks into the future to predict the past. It automatically evaluates the $R^2$ and RMSE scores locally and generates a timestamped predictions.csv.

🚀 Setup & Installation Instructions1.

Clone the repository and navigate into it.
Bash
git clone <repository_url>cd Flipkart_Gridlock
2. Create a virtual environment.
Bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
3. Install dependencies.
Bash
pip install -r requirements.txt
4. Install the project in editable mode.This allows scripts/run_analysis.py to import from src/Model/ without path errors.
Bash
pip install -e .
5. Add the dataset.Ensure train.csv and test.csv are placed directly inside a dataset/ folder at the root of the project.

🏃 How to Run the Pipeline

To execute the entire pipeline (train the model, evaluate the local validation score, and generate the hackathon submission file), 
run:Bash
python scripts/run_analysis.py
The script will output the validation metrics in the terminal and save a file named predictions_YYYYMMDD_HHMMSS.csv to the root directory.
