# XGBoost Classifier — Streamlit App

A live web app built from your XGBoost notebook. Upload any CSV, configure the model from the sidebar, and get accuracy, confusion matrix, and feature importance charts instantly.

## Deploy in 5 minutes (free)

### Step 1 — Push to GitHub
```bash
# Create a new repo on github.com, then:
git init
git add .
git commit -m "xgboost streamlit app"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click **New app**
4. Select your repo → branch: `main` → file: `app.py`
5. Click **Deploy** — live URL in ~2 minutes

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## What it does
- Upload any CSV (last column = target)
- Configure test split, CV folds, and XGBoost hyperparams from sidebar
- See test accuracy, CV accuracy, confusion matrix, feature importance

## Your dataset format
Last column must be the target (y). All other columns are features (X).
