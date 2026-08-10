"""
generate_data.py
-----------------
Generates a synthetic mobile-money transaction dataset for the Fraud Detection API
portfolio project.

Fraud typology modeled: ACCOUNT TAKEOVER.
An attacker gains control of a victim's mobile money account and immediately
drains the balance in a single TRANSFER or CASH_OUT to a "mule" account that
has little or no prior transaction history on the platform. These transactions
are modestly biased toward late-night hours (00:00-05:00), but this bias is
soft, not absolute, and fraud amounts intentionally overlap with the amount
distribution of legitimate high-value transfers.

Design note: an earlier version of this generator produced a dataset that was
trivially separable (AUC = 1.0), which is a red flag for synthetic fraud data
-- it usually means a feature leaks the label directly. This version was
recalibrated to inject overlap and noise so that fraud and legitimate
transactions occupy genuinely overlapping regions of feature space, which is
what makes the modeling problem realistic.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_TRANSACTIONS = 60_000
FRAUD_RATE = 0.012  # 1.2%

TXN_TYPES = ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"]
TXN_TYPE_PROBS = [0.35, 0.34, 0.20, 0.09, 0.02]
# Fraud (account takeover) can only manifest as a balance-draining transfer or cash-out
FRAUD_TXN_TYPES = ["TRANSFER", "CASH_OUT"]
FRAUD_TXN_PROBS = [0.55, 0.45]


def _clip_balance(x):
    return np.maximum(x, 0.0)


def generate_legitimate(n, rng):
    """Legitimate mobile money transactions: everyday payments, cash-ins/outs,
    transfers between known contacts. Balances move but rarely to zero, and
    destination accounts usually have an established transaction history."""
    txn_type = rng.choice(TXN_TYPES, size=n, p=TXN_TYPE_PROBS)

    # Amount: lognormal, heavier tail for TRANSFER/CASH_OUT than PAYMENT/DEBIT
    base_amount = rng.lognormal(mean=8.6, sigma=1.35, size=n)
    is_big_type = np.isin(txn_type, ["TRANSFER", "CASH_OUT"])
    base_amount = np.where(is_big_type, base_amount * rng.uniform(1.0, 1.8, size=n), base_amount)
    # A small legitimate tail of genuinely large transfers (rent, business, etc.)
    # This is what creates deliberate overlap with the fraud amount distribution.
    big_legit_mask = (rng.random(n) < 0.025) & is_big_type
    base_amount[big_legit_mask] *= rng.uniform(3.0, 9.0, size=big_legit_mask.sum())
    amount = np.round(np.clip(base_amount, 1, 950_000), 2)

    old_balance_org = np.round(np.clip(
        amount * rng.uniform(1.4, 12.0, size=n) + rng.exponential(3000, size=n), 0, None
    ), 2)

    # Legit spend rate: usually a modest fraction of the balance moves, but a
    # meaningful slice of legit transactions (e.g. someone closing out an account,
    # paying off a large bill) drain most of the balance -- overlapping with fraud.
    high_drain_mask = rng.random(n) < 0.05
    txn_amount_capped = np.minimum(amount, old_balance_org * 0.98 + 1)
    new_balance_orig = _clip_balance(old_balance_org - txn_amount_capped)
    new_balance_orig = np.where(
        high_drain_mask,
        _clip_balance(old_balance_org * rng.uniform(0.0, 0.15, size=n)),
        new_balance_orig,
    )
    # small noise / rounding artifacts typical of real ledgers
    new_balance_orig = np.round(new_balance_orig + rng.normal(0, 1.5, size=n), 2)
    new_balance_orig = _clip_balance(new_balance_orig)

    old_balance_dest = np.round(rng.exponential(9000, size=n) + rng.uniform(0, 4000, size=n), 2)
    new_balance_dest = np.round(old_balance_dest + txn_amount_capped * rng.uniform(0.9, 1.0, size=n), 2)

    # Destination account transaction history: legit counterparties are usually established,
    # but a non-trivial share of legit transfers go to genuinely new accounts
    # (new merchant, first-time payee, freshly onboarded contact).
    dest_txn_history = rng.negative_binomial(n=6, p=0.25, size=n)  # mean ~18
    newish_mask = rng.random(n) < 0.14
    dest_txn_history[newish_mask] = rng.integers(0, 8, size=newish_mask.sum())

    # Hour of day: legitimate activity concentrated in waking hours, small night tail
    hour = rng.choice(24, size=n, p=_daytime_hour_probs())

    return pd.DataFrame({
        "type": txn_type,
        "amount": amount,
        "oldbalanceOrg": old_balance_org,
        "newbalanceOrig": new_balance_orig,
        "oldbalanceDest": old_balance_dest,
        "newbalanceDest": new_balance_dest,
        "dest_txn_history": dest_txn_history,
        "hour_of_day": hour,
        "is_fraud": 0,
    })


def _daytime_hour_probs():
    hours = np.arange(24)
    # gentle bell centered ~14:00, small floor at night
    weights = np.exp(-0.5 * ((hours - 14) / 5.5) ** 2) + 0.12
    return weights / weights.sum()


def _night_biased_hour_probs():
    hours = np.arange(24)
    # bimodal: elevated 00:00-05:00, but plenty of daytime fraud too (soft bias, not absolute)
    night_weight = np.where((hours >= 0) & (hours <= 5), 2.6, 1.0)
    day_shape = np.exp(-0.5 * ((hours - 15) / 6.0) ** 2) + 0.25
    weights = day_shape * night_weight
    return weights / weights.sum()


def generate_fraud(n, rng):
    """Account-takeover fraud: attacker drains the account in one shot to a
    mule account with a thin transaction history. Amounts intentionally
    overlap with the legitimate transfer distribution -- fraudsters size
    transactions to look plausible, and some legitimate transfers are large
    too, so amount alone is not a reliable separator."""
    txn_type = rng.choice(FRAUD_TXN_TYPES, size=n, p=FRAUD_TXN_PROBS)

    # Amount distribution overlaps heavily with legit big-transfer tail on purpose.
    amount = np.round(np.clip(rng.lognormal(mean=9.7, sigma=1.0, size=n), 500, 900_000), 2)

    # Victim's pre-fraud balance: the attacker typically drains most/all of it,
    # but not with perfect precision -- some fraud is partial (attacker takes what's there).
    old_balance_org = np.round(np.maximum(amount * rng.uniform(0.95, 1.35, size=n), amount + 1), 2)
    # Drain completeness: broad spread so it genuinely overlaps with legit spend fractions
    # (which range up to ~0.98 for the heaviest legit spenders).
    drain_completeness = np.clip(rng.beta(4.0, 1.3, size=n), 0.25, 1.0)
    new_balance_orig = _clip_balance(old_balance_org - amount * drain_completeness)
    new_balance_orig = np.round(new_balance_orig, 2)

    # Mule accounts: thin history is a common signal, but far from universal --
    # a sizeable minority of mule accounts have been "seasoned" with prior transactions
    # that overlap with the low end of the legitimate account-history distribution.
    dest_txn_history = rng.negative_binomial(n=1.6, p=0.28, size=n)  # mean ~4
    seasoned_mask = rng.random(n) < 0.10
    dest_txn_history[seasoned_mask] = rng.integers(6, 22, size=seasoned_mask.sum())

    # Destination balance: mostly thin, but a meaningful slice of mule accounts
    # carry a passthrough balance that overlaps with legit destination balances.
    old_balance_dest = np.round(rng.exponential(600, size=n) + rng.normal(0, 250, size=n).clip(min=0), 2)
    overlap_dest_mask = rng.random(n) < 0.20
    old_balance_dest[overlap_dest_mask] = np.round(
        rng.exponential(9000, size=overlap_dest_mask.sum())
        + rng.uniform(0, 4000, size=overlap_dest_mask.sum()),
        2,
    )
    new_balance_dest = np.round(old_balance_dest + amount * rng.uniform(0.85, 1.0, size=n), 2)

    hour = rng.choice(24, size=n, p=_night_biased_hour_probs())

    return pd.DataFrame({
        "type": txn_type,
        "amount": amount,
        "oldbalanceOrg": old_balance_org,
        "newbalanceOrig": new_balance_orig,
        "oldbalanceDest": old_balance_dest,
        "newbalanceDest": new_balance_dest,
        "dest_txn_history": dest_txn_history,
        "hour_of_day": hour,
        "is_fraud": 1,
    })


def generate_dataset(n=N_TRANSACTIONS, fraud_rate=FRAUD_RATE, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n_fraud = int(round(n * fraud_rate))
    n_legit = n - n_fraud

    legit_df = generate_legitimate(n_legit, rng)
    fraud_df = generate_fraud(n_fraud, rng)

    df = pd.concat([legit_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "transaction_id", [f"txn_{i:07d}" for i in range(len(df))])

    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = "data/transactions.csv"
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df):,} transactions -> {out_path}")
    print(f"Fraud rate: {df['is_fraud'].mean() * 100:.2f}% ({df['is_fraud'].sum():,} fraudulent)")
    print(df.groupby("is_fraud")["amount"].describe()[["mean", "50%", "min", "max"]])
