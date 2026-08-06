<<<<<<< HEAD
# Customer Churn Dashboard

A small end-to-end project: predict which customers are likely to cancel, and surface that in a dashboard the customer success team can actually use.

It's built around the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) as a stand-in for real production data - swap `training/Telco-Customer-Churn.csv` for an export from your own CRM/billing system and the rest of the pipeline doesn't need to change.

## What's in here

- `training/train_churn_model.py` - cleans the data, tunes Logistic Regression / Random Forest / XGBoost with GridSearchCV, picks the best one, and optimizes the classification threshold instead of just defaulting to 0.5
- `api/index.py` - FastAPI service that reads the model's output and serves it as JSON (KPIs, filtered customer lists, per-customer detail)
- `src/App.jsx` - the dashboard itself (React + Tailwind + Recharts)
- `.github/workflows/retrain.yml` - retrains the model weekly on GitHub Actions and commits the refreshed scores, which triggers Vercel to redeploy

## Why it's built this way

The training job needs a minute or two to run (GridSearchCV over three models), which is too long for a Vercel serverless function. Rather than standing up a server that runs 24/7 just to retrain once a week, the retraining happens on GitHub Actions instead, and the result - a CSV of risk scores - gets committed back to the repo. Vercel picks up the push and redeploys with fresh data. No database, no queue, no always-on compute. It's not the architecture you'd want at a much bigger scale, but for a dataset this size it's simpler and cheaper than the alternative, not a shortcut.

## Running it locally

You'll need Node 18+ and Python 3.11+.

```bash
# install everything
npm install
pip install -r requirements.txt

# terminal 1 - the API
uvicorn api.index:app --reload --port 3000

# terminal 2 - the dashboard
npm run dev
```

Dashboard's at `http://localhost:5173`, API at `http://localhost:3000`. The Vite dev server proxies `/api` calls to port 3000 automatically (see `vite.config.js` if you need to change the port).

### Retraining locally

```bash
cd training
pip install -r requirements.txt
python train_churn_model.py --data Telco-Customer-Churn.csv --out ../data/risk_scores.csv --metrics_out ../data/model_metrics.json
```

You can also tune the model toward a specific recall target instead of the default (best F1):

```bash
python train_churn_model.py --target_recall 0.85
```

That's useful if the customer success team can only realistically reach out to, say, a few hundred accounts a week - you can dial the threshold to match capacity instead of accepting whatever the default gives you.

## Deploying

Push to GitHub, then import the repo on [vercel.com](https://vercel.com) with the Vite preset. That's genuinely it - `vercel.json` handles routing `/api/*` to the Python function, and the GitHub Actions workflow is already scheduled for Mondays at 06:00 UTC (also runnable manually from the Actions tab).

Full details in `DEPLOYMENT.md`.

## Known limitations

- The Telco dataset doesn't include login frequency or support ticket history, which the original spec called for - if you're working from real usage/support data those fields would meaningfully improve the model
- Risk scores update once a week; there's no real-time scoring path here
- The dashboard's customer list loads everything through a couple of API calls - fine at ~7K customers, would need actual pagination-aware indexing at much larger scale

## License

MIT - do what you want with it.
=======
# Customer-Churn-Prediction-Analytics-Dashboard
End-to-end churn prediction system: tuned ML model (XGBoost, GridSearchCV) with threshold optimization, serving a live dashboard for identifying and prioritizing at-risk customers
>>>>>>> 2b8bd48903d343d9e12c277dd996812c4f906abf
