# Fraud Detection API

A mobile-money fraud detection service built to show how a machine learning model
compares against the kind of static rule that's still common in production risk systems.
FastAPI + scikit-learn, trained on a synthetic dataset modeling **account takeover** fraud.

## The business problem

Mobile money platforms (M-Pesa-style wallets, mobile-first banks) face a fraud pattern
where an attacker compromises a victim's account credentials — phishing, SIM swap,
credential stuffing — and immediately drains the balance in a single transfer or
cash-out to an account they control (a "mule" account with little history on the
platform, often itself part of a longer laundering chain).

Many platforms still guard against this with a static rule: flag any transfer or
cash-out above a fixed dollar threshold. It's cheap, instant, and fully explainable
to a regulator. It's also nearly blind to this fraud pattern, because attackers don't
need to move huge amounts to do damage — they drain whatever the victim has, which
usually looks like an ordinary large transfer, not an outlier.

This project quantifies that gap and shows what a model that looks at *behavior*
instead of just *amount* buys you.

## Results

| | Static rule (amount > $200k) | Random Forest model |
|---|---|---|
| Recall (% of fraud caught) | 0.56% | 80.0% |
| Precision | 1.3% | 96.0% |
| AUC | — | 0.9996 |

The static rule catches essentially none of the fraud in this dataset — 4 out of 720
fraudulent transactions — because account-takeover fraud is sized to look like a normal
transfer, not a whale transaction. The model, using account-behavior features (how much
of the balance got drained, how established the destination account is, time of day),
catches 4 out of every 5 fraud cases while flagging fraud correctly 96% of the time it
raises an alert.

Full metrics, confusion matrices, and feature importances are served live at `GET /model-info`.

## Stack

- **API**: FastAPI, Pydantic v2, served with Uvicorn
- **Model**: scikit-learn `RandomForestClassifier` inside a `Pipeline` (preprocessing + model bundled into one artifact)
- **Data**: synthetic, generated with NumPy/Pandas (`scripts/generate_data.py`)

## Project structure

```
fraud-detection-api/
├── app/
│   ├── main.py          # FastAPI app: /predict, /model-info, /health
│   └── schemas.py        # Pydantic request/response models
├── scripts/
│   ├── generate_data.py  # synthetic dataset generator
│   └── train_model.py    # trains RF, benchmarks vs. static rule, saves artifacts
├── models/
│   ├── fraud_model.pkl        # trained sklearn Pipeline
│   └── model_metadata.json    # metrics consumed by /model-info
├── data/
│   └── transactions.csv       # 60k generated transactions
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt

# regenerate data + retrain (optional -- trained artifacts are already committed)
python scripts/generate_data.py
python scripts/train_model.py

# run the API
uvicorn app.main:app --reload
```

Then open `http://localhost:8000/docs` for interactive Swagger docs, or:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "type": "TRANSFER",
    "amount": 45230.50,
    "oldbalanceOrg": 46100.00,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 120.00,
    "newbalanceDest": 45350.50,
    "dest_txn_history": 1,
    "hour_of_day": 3
  }'
```

```json
{
  "fraud_score": 0.7814,
  "model_flag": true,
  "static_rule_flag": false,
  "flags_agree": false,
  "decision_threshold": 0.5
}
```

That last example is the whole point of the project in one response: the static rule
misses it (amount is under $200k), the model catches it (near-total balance drain to a
brand-new destination account at 3am is exactly the account-takeover signature).

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Score a transaction: returns `fraud_score`, model flag, static-rule flag, and whether they agree |
| `/model-info` | GET | Training metrics (recall/precision/AUC for both model and rule), feature importances |
| `/health` | GET | Liveness check |
| `/docs` | GET | Swagger UI |

## The dataset

60,000 synthetic mobile-money transactions, 1.2% labeled fraud (720 transactions),
generated in `scripts/generate_data.py`. Every fraud case follows the same typology —
account takeover — so the label isn't "fraud in general," it's specifically this pattern:

- Attacker drains most or all of the victim's balance in one `TRANSFER` or `CASH_OUT`
- Destination is typically a low-history account (a handful of prior transactions or fewer)
- Slight bias toward late-night hours (00:00–05:00), but this is soft — plenty of fraud
  happens in daylight, because a hard time-of-day rule would just be another rule to evade

**A note on why the distributions overlap on purpose.** The first version of this
generator produced a dataset that separated perfectly — AUC = 1.0 on the very first
model. That's not a win, it's a red flag: it almost always means a feature is leaking
the label (e.g., a value that's mechanically only ever nonzero for fraud rows). Real
fraud data never separates that cleanly; if it does, you're probably validating your
data pipeline, not your model. I recalibrated the generator so a meaningful slice of
legitimate transactions also drain most of a balance (someone closing out an account,
paying off a large bill) and go to newer destination accounts (a first-time payee), and
so a meaningful slice of fraud lands on "seasoned" mule accounts with some prior
history. That overlap is what makes recall <100% and precision <100% honest numbers
instead of an artifact of a too-easy dataset.

## Model

`RandomForestClassifier` (300 trees, max depth 12) trained on 8 features: transaction
amount, sender/recipient balances before and after, recipient transaction history,
hour of day, and two engineered features — `balance_drained_ratio` (how much of the
sender's balance moved) and `is_night`. Transaction `type` is one-hot encoded.
75/25 stratified train/test split.

**Why Random Forest over the static rule:** the rule can only look at one number
(amount) against one threshold. The model can combine several weak signals — a
partial balance drain to an established account isn't very suspicious, and neither
is a large transfer to a brand-new account by itself, but a *near-total drain to a
brand-new account* is a strong compound signal a flat threshold can't express.

## How I'd explain this in an interview

*"I built a fraud detection API that catches account-takeover fraud — someone
compromises a mobile money account and drains it to a mule account. I started by
generating a synthetic dataset because I didn't have access to real transaction data,
and I deliberately modeled one specific, well-understood fraud typology rather than
a vague 'fraud vs. not fraud' label, because that's closer to how a real fraud team
actually scopes a detection problem — one model per typology, not one model for
everything.*

*My first pass at the dataset gave me AUC = 1.0, which I flagged immediately as a
data leakage problem rather than a win — a perfect score on synthetic data usually
means you've encoded the label into a feature by accident. I went back and recalibrated
the generator so legitimate and fraudulent transactions genuinely overlap in feature
space — some legit transactions drain most of a balance, some fraud lands on
older mule accounts — which is what forces the model to actually learn a decision
boundary instead of memorizing a shortcut.*

*Then I benchmarked the model against the kind of static rule a lot of these platforms
actually run — flag anything over a dollar threshold. The rule caught under 1% of the
fraud in my test set, because account-takeover fraud is sized to look like a normal
transfer, not an outlier. The Random Forest, using behavioral features like balance-drain
ratio and destination account history, caught 80% of it at 96% precision. That gap is
the pitch: rules react to size, models can react to behavior.*

*I shipped it as a FastAPI service with a `/predict` endpoint that returns both the
model's score and the static rule's flag side by side, specifically so you can see where
they disagree — that's usually where the interesting fraud is."*

## Limitations

This is a portfolio project, not a production fraud system, and I want to be upfront
about what it doesn't do:

- **Single fraud typology.** The model only knows account-takeover fraud. It has never
  seen synthetic identity fraud, first-party fraud, merchant collusion, or any other
  pattern, and would need separate modeling (or a multi-typology dataset) to generalize.
- **No velocity features.** Real fraud systems lean heavily on velocity — transactions
  per hour, sudden change in transaction frequency, multiple accounts touched by the
  same device/IP in a short window. This model scores transactions independently, with
  no sense of a sequence of events, which is a significant blind spot for a lot of
  fraud rings that don't look suspicious in any single transaction.
- **Fixed 0.5 decision threshold.** `model_flag` in `/predict` is hardcoded to fire at
  a 0.5 probability cutoff. A real deployment would tune this against the actual cost
  of a false positive (blocking a legitimate transaction) vs. a false negative (letting
  fraud through), and would likely expose the threshold as a config value rather than
  a constant.
- **Synthetic data.** The dataset is generated, not observed. I designed it to avoid
  the trivial-separability trap, but it still reflects my assumptions about how
  account-takeover fraud behaves rather than empirical transaction data.
