# Flipkart Gridlock

This repository contains a Python data science project for the Flipkart Gridlock challenge.

## Structure

- `dataset/` - existing raw dataset files
- `src/` - project source code
- `scripts/` - runnable workflows
- `tests/` - unit tests
- `notebooks/` - exploratory analysis notebooks

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Usage

Run the analysis script:

```powershell
python scripts\run_analysis.py
```

## Notes

- Keep raw dataset files in `dataset/`
- Add notebooks to `notebooks/`
- Add domain-specific modeling code under `src/flipkart_gridlock`
