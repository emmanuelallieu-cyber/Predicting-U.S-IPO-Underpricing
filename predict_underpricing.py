#!/usr/bin/env python3
"""
Predict IPO underpricing (target: underpriced = 1/0) with an expanding-window,
purged chronological design.  Default models (--models): Random Forest, Gradient Boosting,
SVM, Neural Network (+ an unpenalized logit as a confirmatory classical benchmark).
Logistic Regression, LightGBM and XGBoost remain available via --models LR,LGBM,XGB,...
Every model is run on each of the 8 pillars separately and on all pillars
combined ("ALL"), and a control set (ipo_year only) shows how much time alone
predicts.

VALIDATION DESIGN (matches the fold table in the request)
---------------------------------------------------------
* Test years 2016..last year in the data. Fold Y trains on EVERY IPO dated
  before 1 Jan Y (so earlier folds' test and dropped IPOs re-enter training).
* IPOs dated less than EMBARGO_DAYS (30) days after 31 Dec (i.e. <= 29 Jan) are
  DROPPED from that fold's test set (information spillover from training).
* Median/IQR scaling (logistic models) is fitted on the training rows of each fold only.
  No imputation or winsorization is applied (data already complete, no outliers).
* Hyper-parameters are tuned on a purged chronological validation block inside
  the training window (last 20% of training IPOs; a 30-day gap is removed
  before it).  Logistic C: grid.  RF / XGBoost / LightGBM: random search.
* Class imbalance handled with class weights (balanced / scale_pos_weight).

STATISTICAL CONFIRMATION
------------------------
* Pooled out-of-sample AUC with bootstrap 95% CI and a label-permutation p-value
* DeLong paired AUC tests (model vs model on ALL; ALL vs each pillar per model)
* McNemar tests on 0.5-threshold predictions, Wilcoxon on fold AUCs
* Unpenalized logit (own IRLS code, HC3 robust SEs): odds ratios, LR test per
  pillar, drop-one-pillar LR tests, McFadden R2, AIC/BIC, VIF
DATA-LEAKAGE CHECKS
-------------------
* Temporal integrity per fold (train < embargo < test, company overlap, etc.)
* Label-permutation REFIT: shuffle training labels, refit, score the real test
  set. Honest pipelines return AUC ~ 0.5.
* Univariate AUC screen of every feature (flags near-perfect predictors)
* Random 5-fold vs chronological CV gap (diagnostic only, never reported as
  the result), and the ipo_year-only control.

Usage (Windows Command Prompt)
------------------------------
    python predict_underpricing.py --input New Dataset Test.xlsx --outdir results
    python predict_underpricing.py --input New Dataset Test.xlsx --quick   (fast smoke test)

DATA HANDLING (leak-free): use the RAW workbook (blanks and outliers present).
Median/mode imputation and iterative winsorization (Tukey fences, repeated until no
outlier remains) are FITTED ON EACH FOLD'S TRAINING ROWS ONLY and then applied to that
fold's test rows; median/IQR scaling (logistic models) is fitted the same way. No future
IPO can influence a fill value or a winsorization bound.  --as-given skips imputation and
winsorization (only for a workbook you know is already complete and clean).

FEATURES: log_proceeds is dropped: it is ~ log_offer_price + log_shares_offered (VIF 85),
which made the unpenalized logit coefficients unstable.

OUTPUTS (in --outdir)
  underpricing_results.xlsx        results, significance tests, leakage checks, predictions
  preprocessed_data_by_fold.xlsx   the imputed + winsorized data each fold's models used
                                   (train and test rows) plus fill values / bounds per fold
  shap_values_probability.xlsx     SHAP in PROBABILITY units (global, pillar, yearly,
                                   per-IPO values, top drivers)
  shap_report.html + shap_*.png    explainability report
  descriptive_statistics_train_foldYYYY.xlsx + desc_*.png
                                   summary statistics / underpricing by year, sector, industry, underwriter,
                                   exchange, governance flags and every feature - TRAINING rows of ONE fold only
                                   (--desc-fold last|first) so nothing from the test years can leak
  ablation_results.xlsx + ablation_*.png
                                   how much common validation flaws inflate AUC (ex ablation_flaws.py), run on
                                   the same data / features / models with fixed default hyper-parameters
  IPO_underpricing_report.docx     Word report: executive summary, descriptives, full methodology, results,
                                   statistical tests, leakage checks, SHAP (probability units), ablation, limitations
SHAP is out-of-sample: every fold's fitted model (ALL-pillars feature set) explains that
fold's test IPOs. Values are changes in predicted P(underpriced) and add up to
prediction minus base value. The `shap` package is used if installed (TreeExplainer for
RF/XGBoost/LightGBM); otherwise a built-in permutation-Shapley estimator is used.
SHAP explains what the MODEL relies on, not what causes underpricing.
"""

import argparse
import os
import re
import sys
import time
import warnings
import zlib
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, log_loss, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None
try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
DATE_COL, TARGET = "ipo_date", "underpriced"
FIRST_TEST_YEAR = 2016
EMBARGO_DAYS = 30            # test-year IPOs less than 30 days after 31 Dec (<= 29 Jan) are dropped
VAL_FRAC, VAL_PURGE_DAYS, VAL_MIN = 0.20, 30, 30
WINSOR_K, WINSOR_MAX_ITER = 1.5, 100
CLEAN_IN_FOLD = True    # True: in-fold (training-only) imputation + iterative winsorization; --as-given turns it off

CONT_ALL = ["firm_age", "Free Cash Flow Margin", "Earnings Quality Ratio (FCF Conversion)",
            "profit_margin", "Price-to-Sales (P/S) Ratio", "Price-to-Earnings (P/E)",
            "us_gdp", "us_inflation_pct", "prime_rate", "yield_curve_spread",
            "vix", "nasdaq_composite", "ipo_count_30d",
            "n_known_underwriters", "log_prospectus_words", "risk_word_share",
            "log_offer_price", "log_shares_offered"]
BINARY = ["institutional_backing", "big4_auditor", "dual_class"]

# Reference categories (dropped to avoid the dummy-variable trap)
EXCH_LEVELS = ["exch_nyse", "exch_nyse_american"]            # reference: exch_nasdaq
UW_LEVELS = ["Tier 2", "Other", "Unknown"]                   # reference: Tier 1
SECTOR_RULES = [  # first match wins; reference sector = Biotech & Pharma
    ("Biotech & Pharma", r"^Biotechnology$|^Drug Manufacturers"),
    ("Financials & Real Estate", r"REIT|Real Estate|Banks|Capital Markets|Asset Management|Credit Services|"
                                 r"Insurance|Mortgage|Financial|Investors"),
    ("Consumer & Media", r"Retail"),
    ("Healthcare", r"Medic|Diagnostics|Health|Clinics"),
    ("Software & Internet", r"Software|Internet|Information Technology|Electronic Gaming|Computer Processing|"
                            r"^Technology$|Telecom"),
    ("Hardware & Semis", r"Semiconductor|Computer Hardware|Communication Equipment|Consumer Electronics|"
                         r"Electronic Components|Scientific & Technical|Solar"),
    ("Energy & Materials", r"Oil & Gas|Coal|Uranium|Copper|Steel|Chemicals|Agricultural Inputs|Lumber|Metals|"
                           r"Mining|Utilities|Building Materials|Packaging"),
    ("Industrials", r"Aerospace|Engineering|Machinery|Electrical Equipment|Building Products|Freight|Shipping|"
                    r"Airlines|Trucking|Waste|Metal Fabrication|Industrial|Staffing|Security|Conglomerates|"
                    r"Infrastructure|Rental|Ship &|Farm & Heavy|Tools|Consulting|Business Services|"
                    r"Professional Services|Services-Commercial|Services-Misc"),
    ("Consumer & Media", r"Restaurants|Apparel|Auto|Leisure|Food|Beverages|Grocery|Tobacco|Household|Personal|"
                         r"Furnishings|Recreational|Resorts|Gambling|Travel|Luxury|Footwear|Education|Farm Products|"
                         r"Discount|Entertainment|Media|Broadcasting|Advertising|Construction|Textile"),
]
SECTORS = ["Biotech & Pharma", "Financials & Real Estate", "Consumer & Media", "Healthcare",
           "Software & Internet", "Hardware & Semis", "Energy & Materials", "Industrials", "Other"]


def sec_col(s):
    return "sec_" + re.sub(r"[^A-Za-z]+", "_", s).strip("_")


SECTOR_COLS = [sec_col(s) for s in SECTORS[1:]]
UW_COLS = ["uw_" + re.sub(r"\W+", "", l) for l in UW_LEVELS]

PILLARS = {
    "1. Firm fundamentals": ["firm_age", "Free Cash Flow Margin",
                             "Earnings Quality Ratio (FCF Conversion)", "profit_margin"],
    "2. Valuation": ["Price-to-Sales (P/S) Ratio", "Price-to-Earnings (P/E)"],
    "3. Macroeconomic environment": ["us_gdp", "us_inflation_pct", "prime_rate", "yield_curve_spread"],
    "4. Market conditions and sentiment": ["vix", "nasdaq_composite", "ipo_count_30d"],
    "5. Certification and governance": ["institutional_backing", "n_known_underwriters", "big4_auditor",
                                        "dual_class"] + UW_COLS,
    "6. Disclosure and information": ["log_prospectus_words", "risk_word_share"],
    "7. Offer structure": ["log_offer_price", "log_shares_offered"],   # log_proceeds dropped (collinear)
    "8. Listing venue and sector": EXCH_LEVELS + SECTOR_COLS,
}
ALL_NAME = "ALL pillars"
FEATURE_SETS = dict(PILLARS)
FEATURE_SETS[ALL_NAME] = [c for cols in PILLARS.values() for c in cols]
CONTROL_NAME = "Control: ipo_year only"
CONTROL_SETS = {CONTROL_NAME: ["ipo_year_frac"]}

MODEL_LABEL = {"LR": "Logistic Regression", "LGBM": "LightGBM", "RF": "Random Forest",
               "XGB": "XGBoost", "GB": "Gradient Boosting", "SVM": "SVM", "NN": "Neural Network",
               "LOGIT": "Logit (unpenalized)"}
SCALED_MODELS = ("LR", "LOGIT", "SVM", "NN")   # models that need median/IQR-scaled inputs
ALL_MODEL_CODES = ("LR", "LGBM", "RF", "XGB", "GB", "SVM", "NN")

# User's expected fold table (n_train, n_test, n_dropped) for cross-checking
EXPECTED = {2016: (198, 27, 0), 2017: (225, 31, 2), 2018: (258, 43, 4), 2019: (305, 53, 0),
            2020: (358, 67, 2), 2021: (427, 141, 10), 2022: (578, 23, 3), 2023: (604, 41, 3),
            2024: (648, 60, 4), 2025: (712, 80, 4), 2026: (796, 33, 7)}


def map_sector(name):
    if not isinstance(name, str):
        return "Other"
    for sector, pat in SECTOR_RULES:
        if re.search(pat, name, flags=re.I):
            return sector
    return "Other"


def stable_seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) % (2 ** 31)


# --------------------------------------------------------------------------
# Data + features
# --------------------------------------------------------------------------
def load_data(path):
    df = pd.read_excel(path)
    df = df.dropna(how="all").dropna(subset=[DATE_COL, TARGET])
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL, kind="stable").reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)
    n_nan = int(df[CONT_ALL + BINARY].isna().sum().sum())
    print(f"Loaded {len(df)} IPOs ({df[DATE_COL].min():%Y-%m-%d} to {df[DATE_COL].max():%Y-%m-%d}); "
          f"missing feature cells: {n_nan}")
    if n_nan == 0:
        print("Input has no missing cells (imputation will be a no-op" +
              ("; winsorization is still fitted in-fold)." if CLEAN_IN_FOLD else "; used as given)."))
    return df


def build_features(df):
    X = pd.DataFrame(index=df.index)
    for c in CONT_ALL + BINARY:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    for c in BINARY:
        X.loc[~X[c].isin([0, 1]), c] = np.nan          # corrupted flags -> missing -> imputed in-fold
    exch = df["Exchange"].fillna("exch_nasdaq")         # constant fill (reference level), not data-dependent
    for lvl in EXCH_LEVELS:
        X[lvl] = (exch == lvl).astype(int)
    uw = df["underwriter_reputation"].fillna("Unknown")
    for lvl, col in zip(UW_LEVELS, UW_COLS):
        X[col] = (uw == lvl).astype(int)
    sector = df["industry"].map(map_sector)
    for s in SECTORS[1:]:
        X[sec_col(s)] = (sector == s).astype(int)
    X["ipo_year_frac"] = df[DATE_COL].dt.year + (df[DATE_COL].dt.dayofyear - 1) / 366.0
    return X, sector


# --------------------------------------------------------------------------
# Folds
# --------------------------------------------------------------------------
def make_folds(df, embargo_days=None):
    ed = EMBARGO_DAYS if embargo_days is None else embargo_days
    folds = []
    for yr in range(FIRST_TEST_YEAR, int(df[DATE_COL].dt.year.max()) + 1):
        start, end = pd.Timestamp(yr, 1, 1), pd.Timestamp(yr + 1, 1, 1)
        in_year = df[(df[DATE_COL] >= start) & (df[DATE_COL] < end)]
        cutoff = pd.Timestamp(yr - 1, 12, 31) + pd.Timedelta(days=ed - 1)
        dropped = in_year[in_year[DATE_COL] <= cutoff]
        test = in_year[in_year[DATE_COL] > cutoff]
        train = df[df[DATE_COL] < start]
        if len(test) == 0 or len(train) == 0:
            continue
        folds.append({"year": yr, "train": train.index.values, "test": test.index.values,
                      "dropped": dropped.index.values})
    return folds


def fold_table(df, folds):
    lab = lambda r, f="%Y-%m-%d": f"{r['symbol']} ({r[DATE_COL]:{f}})"
    rows = []
    for f in folds:
        tr, te, dr = df.loc[f["train"]], df.loc[f["test"]], df.loc[f["dropped"]]
        exp = EXPECTED.get(f["year"])
        got = (len(tr), len(te), len(dr))
        rows.append({"fold": f["year"], "n_train": got[0], "n_test": got[1], "n_dropped": got[2],
                     "train_first": lab(tr.iloc[0]), "train_last": lab(tr.iloc[-1]),
                     "test_first": lab(te.iloc[0]), "test_last": lab(te.iloc[-1]),
                     "dropped": ", ".join(lab(r, "%b %d") for _, r in dr.iterrows()),
                     "matches_requested_table": ("yes" if exp == got else
                                                 f"no (requested {exp}; dataset may hold newer IPOs)")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Preprocessing fitted on training rows only
# --------------------------------------------------------------------------
def iterative_bounds(s, k=WINSOR_K, max_iter=WINSOR_MAX_ITER):
    s = s.astype(float).copy()
    lo_c, hi_c = -np.inf, np.inf
    for _ in range(max_iter):
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            lo, hi = (float(x) for x in s.quantile([0.01, 0.99]))   # fallback when IQR = 0
            return float(max(lo_c, lo)), float(min(hi_c, hi))
        lo, hi = q1 - k * iqr, q3 + k * iqr
        if not ((s < lo) | (s > hi)).any():
            break
        s = s.clip(lo, hi)
        lo_c, hi_c = np.clip(lo_c, lo, hi), np.clip(hi_c, lo, hi)
    return float(lo_c), float(hi_c)


class FoldPreprocessor:
    """median/mode imputation -> iterative winsorization -> (optional) median/IQR scaling."""

    def __init__(self, cols):
        self.cols = list(cols)
        self.cont = [c for c in cols if c in CONT_ALL]
        self.other = [c for c in cols if c not in CONT_ALL]

    def fit(self, X):
        Xt = X[self.cols]
        if CLEAN_IN_FOLD:
            self.fill_ = {c: float(Xt[c].median()) for c in self.cont}
            self.fill_.update({c: float(Xt[c].mode().iloc[0]) if Xt[c].notna().any() else 0.0
                               for c in self.other})
            Xi = Xt.fillna(self.fill_)
            self.bounds_ = {c: iterative_bounds(Xi[c]) for c in self.cont}
        else:                                   # data already complete and outlier-free: leave untouched
            self.fill_, self.bounds_, Xi = {}, {}, Xt
        Xw = self._clip(Xi)
        self.scaler_ = RobustScaler().fit(Xw[self.cont]) if self.cont else None   # median/IQR, train rows only
        return self

    def _clip(self, Xi):
        out = Xi.copy()
        for c, (lo, hi) in self.bounds_.items():
            out[c] = out[c].clip(lo, hi)
        return out

    def transform(self, X, scaled=False):
        Xw = self._clip(X[self.cols].fillna(self.fill_) if self.fill_ else X[self.cols])
        if scaled and self.scaler_ is not None:
            Xw[self.cont] = self.scaler_.transform(Xw[self.cont])
        return Xw[self.cols].to_numpy(dtype=float)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def available_models(requested):
    out = []
    for m in requested:
        if m == "LGBM" and LGBMClassifier is None:
            print("WARNING: lightgbm not installed -> LightGBM skipped (pip install lightgbm)")
        elif m == "XGB" and XGBClassifier is None:
            print("WARNING: xgboost not installed -> XGBoost skipped (pip install xgboost)")
        else:
            out.append(m)
    return out


def param_grid(name, quick):
    if name == "LR":
        return [{"C": c} for c in (0.01, 0.1, 1.0, 10.0, 100.0)]
    if name == "RF":
        g = {"n_estimators": [100] if quick else [200, 400, 600], "max_depth": [3, 5, 8, None],
             "min_samples_leaf": [1, 3, 5, 10], "max_features": ["sqrt", 0.5, 1.0]}
    elif name == "XGB":
        g = {"n_estimators": [100, 200, 400], "max_depth": [2, 3, 4, 6], "learning_rate": [0.01, 0.03, 0.05, 0.1],
             "subsample": [0.6, 0.8, 1.0], "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 3, 5],
             "reg_lambda": [1.0, 5.0, 10.0]}
    elif name == "LGBM":
        g = {"n_estimators": [100, 200, 400], "num_leaves": [4, 8, 15, 31], "learning_rate": [0.01, 0.03, 0.05, 0.1],
             "subsample": [0.6, 0.8, 1.0], "colsample_bytree": [0.6, 0.8, 1.0], "min_child_samples": [5, 10, 20],
             "reg_lambda": [0.0, 1.0, 5.0, 10.0]}
    elif name == "GB":
        g = {"n_estimators": [100, 200, 400], "max_depth": [2, 3, 4], "learning_rate": [0.01, 0.03, 0.05, 0.1],
             "subsample": [0.6, 0.8, 1.0], "min_samples_leaf": [1, 3, 5, 10]}
    elif name == "SVM":
        g = {"C": [0.1, 1.0, 10.0, 100.0], "gamma": ["scale", 0.001, 0.01, 0.1]}
    elif name == "NN":
        g = {"hidden_layer_sizes": [(16,), (32,), (64,), (32, 16)], "alpha": [1e-4, 1e-3, 1e-2, 1e-1],
             "learning_rate_init": [1e-3, 3e-3, 1e-2]}
    else:
        return [{}]
    keys = list(g)
    return [dict(zip(keys, v)) for v in product(*g.values())]


def default_params(name):
    return {"LR": {"C": 1.0}, "RF": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 3,
                                      "max_features": "sqrt"},
            "XGB": {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8,
                    "colsample_bytree": 0.8, "min_child_weight": 3, "reg_lambda": 5.0},
            "LGBM": {"n_estimators": 200, "num_leaves": 8, "learning_rate": 0.05, "subsample": 0.8,
                     "colsample_bytree": 0.8, "min_child_samples": 10, "reg_lambda": 5.0},
            "GB": {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8,
                   "min_samples_leaf": 5},
            "SVM": {"C": 1.0, "gamma": "scale"},
            "NN": {"hidden_layer_sizes": (32,), "alpha": 1e-3, "learning_rate_init": 3e-3},
            "LOGIT": {}}[name]


class WeightedGB(GradientBoostingClassifier):
    """sklearn gradient boosting with balanced class weights applied as sample weights (training rows only)."""

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y)
        if sample_weight is None:
            n1 = max(int(y.sum()), 1)
            n0 = max(len(y) - n1, 1)
            sample_weight = np.where(y == 1, len(y) / (2.0 * n1), len(y) / (2.0 * n0))
        return super().fit(X, y, sample_weight=sample_weight)


class OversampledMLP(MLPClassifier):
    """MLP has no class weights: the minority class is randomly oversampled INSIDE the training rows
    handed to fit() (never touches test rows or validation rows)."""

    def fit(self, X, y):
        y = np.asarray(y)
        rng = np.random.default_rng(0 if self.random_state is None else int(self.random_state))
        idx1, idx0 = np.where(y == 1)[0], np.where(y == 0)[0]
        small, big = (idx1, idx0) if len(idx1) < len(idx0) else (idx0, idx1)
        if 0 < len(small) < len(big):
            extra = rng.choice(small, len(big) - len(small), replace=True)
            keep = np.concatenate([big, small, extra])
            X, y = X[keep], y[keep]
        return super().fit(X, y)


def make_model(name, params, ytr, seed, n_jobs):
    pos = max(int(ytr.sum()), 1)
    neg = max(len(ytr) - pos, 1)
    if name == "LR":
        return LogisticRegression(C=params["C"], class_weight="balanced", max_iter=5000, solver="lbfgs")
    if name == "RF":
        return RandomForestClassifier(class_weight="balanced_subsample", n_jobs=n_jobs, random_state=seed, **params)
    if name == "XGB":
        return XGBClassifier(scale_pos_weight=neg / pos, eval_metric="logloss", tree_method="hist",
                             n_jobs=n_jobs, random_state=seed, verbosity=0, **params)
    if name == "LGBM":
        return LGBMClassifier(class_weight="balanced", subsample_freq=1, random_state=seed,
                              n_jobs=n_jobs, verbose=-1, **params)
    if name == "GB":
        return WeightedGB(random_state=seed, **params)
    if name == "SVM":
        return SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=seed, **params)
    if name == "NN":
        return OversampledMLP(activation="relu", solver="adam", max_iter=300, random_state=seed, **params)
    raise ValueError(name)


# ---- unpenalized logit with HC3 robust errors (self-contained; no statsmodels needed)
def fit_logit(X, y, ridge=1e-6, max_iter=100, tol=1e-9):
    n, k = X.shape
    Xc = np.column_stack([np.ones(n), X])
    pen = np.eye(k + 1) * ridge
    pen[0, 0] = 0.0
    beta = np.zeros(k + 1)
    for _ in range(max_iter):
        eta = np.clip(Xc @ beta, -30, 30)
        p = expit(eta)
        w = np.clip(p * (1 - p), 1e-9, None)
        z = eta + (y - p) / w
        new = np.linalg.solve(Xc.T @ (Xc * w[:, None]) + pen, Xc.T @ (w * z))
        done = np.max(np.abs(new - beta)) < tol
        beta = new
        if done:
            break
    p = np.clip(expit(Xc @ beta), 1e-12, 1 - 1e-12)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    W = p * (1 - p)
    bread = np.linalg.inv(Xc.T @ (Xc * W[:, None]) + pen)
    h = np.clip(W * np.einsum("ij,jk,ik->i", Xc, bread, Xc), 0, 0.999)
    u = (y - p) / (1 - h)
    cov = bread @ (Xc.T @ (Xc * (u ** 2)[:, None])) @ bread
    return {"beta": beta, "cov": cov, "ll": ll, "n": n, "k": k + 1,
            "separation": bool(np.max(np.abs(beta)) > 15 or np.any((p < 1e-8) | (p > 1 - 1e-8)))}


class LogitModel:
    def fit(self, X, y):
        self.res_ = fit_logit(X, y)
        return self

    def predict_proba(self, X):
        p = expit(np.clip(np.column_stack([np.ones(len(X)), X]) @ self.res_["beta"], -30, 30))
        return np.column_stack([1 - p, p])


def fit_model(name, params, Xtr, ytr, seed, n_jobs):
    mdl = LogitModel() if name == "LOGIT" else make_model(name, params, ytr, seed, n_jobs)
    mdl.fit(Xtr, ytr)
    return mdl


def fit_predict(name, params, Xtr, ytr, Xte, seed, n_jobs):
    return fit_model(name, params, Xtr, ytr, seed, n_jobs).predict_proba(Xte)[:, 1]


def val_score(y, p):
    if len(np.unique(y)) == 2:
        return roc_auc_score(y, p)
    return -log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])


def tune(name, cache, seed, n_iter, quick, n_jobs):
    """Pick hyper-parameters on the purged validation block."""
    grid = param_grid(name, quick)
    if name != "LR" and len(grid) > 1:
        rng = np.random.default_rng(seed)
        grid = [grid[i] for i in rng.permutation(len(grid))[:n_iter]]
    scaled = name in SCALED_MODELS
    Xs, Xv = (cache["sub_sc"], cache["val_sc"]) if scaled else (cache["sub_raw"], cache["val_raw"])
    best, best_s = grid[0], -np.inf
    for prm in grid:
        try:
            s = val_score(cache["yval"], fit_predict(name, prm, Xs, cache["ysub"], Xv, seed, n_jobs))
        except Exception:
            continue
        if s > best_s:
            best, best_s = prm, s
    return best, best_s


# --------------------------------------------------------------------------
# Metrics and tests
# --------------------------------------------------------------------------
def fast_auc(y, p):
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metrics(y, p, thr=0.5):
    y = np.asarray(y)
    yhat = (p >= thr).astype(int)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return {"auc": fast_auc(y, p), "accuracy": accuracy_score(y, yhat),
            "balanced_accuracy": balanced_accuracy_score(y, yhat), "f1": f1_score(y, yhat, zero_division=0),
            "brier": brier_score_loss(y, pc), "log_loss": log_loss(y, pc, labels=[0, 1])}


def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    out = np.empty(N)
    out[J] = T + 1
    return out


def delong_test(y, p1, p2):
    """Paired DeLong test for the difference of two correlated AUCs."""
    y = np.asarray(y)
    order = (-y).argsort()
    preds = np.vstack([p1, p2])[:, order]
    m = int(y.sum())
    n = len(y) - m
    pos, neg = preds[:, :m], preds[:, m:]
    tx, ty, tz = (np.empty((2, m)), np.empty((2, n)), np.empty((2, m + n)))
    for r in range(2):
        tx[r], ty[r], tz[r] = _midrank(pos[r]), _midrank(neg[r]), _midrank(preds[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else np.nan
    return aucs[0], aucs[1], z, 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan


def mcnemar_test(y, p1, p2, thr=0.5):
    c1, c2 = ((p1 >= thr) == y), ((p2 >= thr) == y)
    b, c = int((c1 & ~c2).sum()), int((~c1 & c2).sum())
    pval = stats.binomtest(b, b + c, 0.5).pvalue if (b + c) > 0 else 1.0
    return b, c, pval


# --------------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------------
def build_cache(X, y, dates, f, cols):
    """Preprocess one (fold, feature set): full-train arrays and tuning-split arrays."""
    tr, te = f["train"], f["test"]
    prep = FoldPreprocessor(cols).fit(X.loc[tr])
    cache = {"tr": tr, "te": te, "ytr": y[tr], "yte": y[te], "prep": prep,
             "tr_raw": prep.transform(X.loc[tr]), "te_raw": prep.transform(X.loc[te]),
             "tr_sc": prep.transform(X.loc[tr], True), "te_sc": prep.transform(X.loc[te], True)}
    # purged chronological validation split inside the training window
    dtr = dates[tr]
    nv = max(VAL_MIN, int(VAL_FRAC * len(tr)))
    nv = min(nv, max(len(tr) // 2, 1))
    val_pos = np.arange(len(tr) - nv, len(tr))
    val_start = dtr[val_pos[0]]
    sub_pos = np.where(dtr < val_start - np.timedelta64(VAL_PURGE_DAYS, "D"))[0]
    if len(sub_pos) < 20:
        sub_pos = np.arange(0, max(len(tr) - nv, 20))
    sub_idx, val_idx = tr[sub_pos], tr[val_pos]
    prep_sub = FoldPreprocessor(cols).fit(X.loc[sub_idx])
    cache.update({"sub_raw": prep_sub.transform(X.loc[sub_idx]), "val_raw": prep_sub.transform(X.loc[val_idx]),
                  "sub_sc": prep_sub.transform(X.loc[sub_idx], True), "val_sc": prep_sub.transform(X.loc[val_idx], True),
                  "ysub": y[sub_idx], "yval": y[val_idx]})
    return cache


def run_experiment(df, X, folds, models, args):
    y = df[TARGET].to_numpy()
    dates = df[DATE_COL].to_numpy()
    sets = dict(FEATURE_SETS)
    sets.update(CONTROL_SETS)
    all_models = list(models) + ["LOGIT"]
    pred_rows, best_params, caches, fitted = [], {}, {}, {}
    t0 = time.time()
    for f in folds:
        for sname, cols in sets.items():
            cache = build_cache(X, y, dates, f, cols)
            caches[(f["year"], sname)] = cache
            for m in all_models:
                seed = stable_seed(args.seed, f["year"], sname, m)
                if m == "LOGIT":
                    prm = {}
                else:
                    prm, _ = tune(m, cache, seed, args.n_iter, args.quick, args.jobs)
                best_params[(f["year"], sname, m)] = prm
                scaled = m in SCALED_MODELS
                Xtr, Xte = (cache["tr_sc"], cache["te_sc"]) if scaled else (cache["tr_raw"], cache["te_raw"])
                try:
                    mdl = fit_model(m, prm, Xtr, cache["ytr"], seed, args.jobs)
                    p = mdl.predict_proba(Xte)[:, 1]
                    if sname == ALL_NAME and m != "LOGIT":
                        fitted[(f["year"], m)] = mdl          # kept for the SHAP report
                except Exception as e:
                    print(f"  ! {m} failed on {sname} fold {f['year']}: {e}")
                    p = np.full(len(cache["te"]), cache["ytr"].mean())
                pred_rows.append(pd.DataFrame({"set": sname, "model": m, "fold": f["year"],
                                               "symbol": df.loc[cache["te"], "symbol"].values,
                                               "ipo_date": df.loc[cache["te"], DATE_COL].values,
                                               "y": cache["yte"], "p": p}))
        print(f"  fold {f['year']} done ({time.time() - t0:.0f}s)")
    # prior-rate baseline
    for f in folds:
        pb = float(y[f["train"]].mean())
        te = f["test"]
        pred_rows.append(pd.DataFrame({"set": ALL_NAME, "model": "PRIOR", "fold": f["year"],
                                       "symbol": df.loc[te, "symbol"].values, "ipo_date": df.loc[te, DATE_COL].values,
                                       "y": y[te], "p": pb}))
    return pd.concat(pred_rows, ignore_index=True), best_params, caches, fitted


def summarize(preds, args):
    rows_fold, rows_pool = [], []
    for (s, m), g in preds.groupby(["set", "model"], sort=False):
        for fy, gf in g.groupby("fold"):
            r = metrics(gf["y"], gf["p"].to_numpy())
            r.update({"set": s, "model": m, "fold": fy, "n_test": len(gf)})
            rows_fold.append(r)
        r = metrics(g["y"], g["p"].to_numpy())
        fold_auc = [x["auc"] for x in rows_fold if x["set"] == s and x["model"] == m]
        r.update({"set": s, "model": m, "n_test": len(g), "mean_fold_auc": np.nanmean(fold_auc),
                  "std_fold_auc": np.nanstd(fold_auc, ddof=1) if len(fold_auc) > 1 else np.nan})
        rows_pool.append(r)
    fold_df, pool = pd.DataFrame(rows_fold), pd.DataFrame(rows_pool)

    # bootstrap CI + permutation p-value (labels permuted within each test fold)
    rng = np.random.default_rng(args.seed)
    ref = preds[(preds["set"] == ALL_NAME) & (preds["model"] == "PRIOR")]
    y_ref, fold_ref = ref["y"].to_numpy(), ref["fold"].to_numpy()
    Yperm = np.empty((args.n_perm, len(y_ref)))
    for fy in np.unique(fold_ref):
        idx = np.where(fold_ref == fy)[0]
        for i in range(args.n_perm):
            Yperm[i, idx] = rng.permutation(y_ref[idx])
    n1 = y_ref.sum()
    n0 = len(y_ref) - n1
    lo, hi, pv = [], [], []
    for _, r in pool.iterrows():
        g = preds[(preds["set"] == r["set"]) & (preds["model"] == r["model"])]
        p, yy = g["p"].to_numpy(), g["y"].to_numpy()
        boots = []
        for _ in range(args.n_boot):
            ix = rng.integers(0, len(yy), len(yy))
            boots.append(fast_auc(yy[ix], p[ix]))
        lo.append(np.nanpercentile(boots, 2.5))
        hi.append(np.nanpercentile(boots, 97.5))
        ranks = rankdata(p)
        perm_auc = (Yperm @ ranks - n1 * (n1 + 1) / 2) / (n1 * n0)
        pv.append((1 + np.sum(perm_auc >= r["auc"] - 1e-12)) / (args.n_perm + 1))
    pool["auc_ci_low"], pool["auc_ci_high"], pool["perm_p_value_auc"] = lo, hi, pv
    return fold_df, pool


def pairwise_tests(preds, fold_df, models):
    get = lambda s, m: preds[(preds["set"] == s) & (preds["model"] == m)]
    ms = list(models) + ["LOGIT"]
    d_models, d_pillar, mc, wx = [], [], [], []
    for i, a in enumerate(ms):
        for b in ms[i + 1:]:
            ga, gb = get(ALL_NAME, a), get(ALL_NAME, b)
            y = ga["y"].to_numpy()
            a1, a2, z, p = delong_test(y, ga["p"].to_numpy(), gb["p"].to_numpy())
            d_models.append({"set": ALL_NAME, "model_A": MODEL_LABEL[a], "model_B": MODEL_LABEL[b],
                             "auc_A": a1, "auc_B": a2, "auc_diff": a1 - a2, "z": z, "p_value": p})
            bb, cc, pm = mcnemar_test(y, ga["p"].to_numpy(), gb["p"].to_numpy())
            mc.append({"set": ALL_NAME, "model_A": MODEL_LABEL[a], "model_B": MODEL_LABEL[b],
                       "A_right_B_wrong": bb, "A_wrong_B_right": cc, "p_value": pm})
            fa = fold_df[(fold_df["set"] == ALL_NAME) & (fold_df["model"] == a)].sort_values("fold")["auc"].to_numpy()
            fb = fold_df[(fold_df["set"] == ALL_NAME) & (fold_df["model"] == b)].sort_values("fold")["auc"].to_numpy()
            try:
                pw = stats.wilcoxon(fa, fb).pvalue
            except Exception:
                pw = np.nan
            wx.append({"set": ALL_NAME, "model_A": MODEL_LABEL[a], "model_B": MODEL_LABEL[b],
                       "mean_fold_auc_A": np.nanmean(fa), "mean_fold_auc_B": np.nanmean(fb), "wilcoxon_p": pw})
    for m in ms:
        gall = get(ALL_NAME, m)
        y = gall["y"].to_numpy()
        for pn in list(PILLARS) + list(CONTROL_SETS):
            gp = get(pn, m)
            a1, a2, z, p = delong_test(y, gall["p"].to_numpy(), gp["p"].to_numpy())
            d_pillar.append({"model": MODEL_LABEL[m], "ALL_auc": a1, "other_set": pn, "other_auc": a2,
                             "auc_diff": a1 - a2, "z": z, "p_value": p})
    return (pd.DataFrame(d_models), pd.DataFrame(d_pillar), pd.DataFrame(mc), pd.DataFrame(wx))


# --------------------------------------------------------------------------
# Classical logit inference (full sample, descriptive)
# --------------------------------------------------------------------------
def logit_inference(df, X):
    y = df[TARGET].to_numpy().astype(float)
    out_coef, out_sets, out_drop = [], [], []

    def design(cols):
        prep = FoldPreprocessor(cols).fit(X)          # full-sample: descriptive inference only
        return prep.transform(X, scaled=True)

    def fit_set(cols):
        res = fit_logit(design(cols), y)
        p0 = y.mean()
        ll0 = float(np.sum(y * np.log(p0) + (1 - y) * np.log(1 - p0)))
        lr = 2 * (res["ll"] - ll0)
        k = len(cols)
        return res, ll0, lr, k, stats.chi2.sf(lr, k)

    for sname, cols in FEATURE_SETS.items():
        try:
            res, ll0, lr, k, p = fit_set(cols)
        except Exception as e:
            print(f"  ! logit inference failed for {sname}: {e}")
            continue
        out_sets.append({"set": sname, "n": res["n"], "k_features": k, "loglik": res["ll"],
                         "LR_chi2_vs_null": lr, "LR_p_value": p, "McFadden_R2": 1 - res["ll"] / ll0,
                         "AIC": 2 * res["k"] - 2 * res["ll"], "BIC": res["k"] * np.log(res["n"]) - 2 * res["ll"],
                         "possible_separation": res["separation"]})
        if sname == ALL_NAME:
            se = np.sqrt(np.clip(np.diag(res["cov"]), 1e-18, None))
            z = res["beta"] / se
            names = ["const"] + cols
            for i, nme in enumerate(names):
                out_coef.append({"feature": nme, "coef": res["beta"][i], "odds_ratio": np.exp(res["beta"][i]),
                                 "robust_se_HC3": se[i], "z": z[i], "p_value": 2 * stats.norm.sf(abs(z[i])),
                                 "OR_ci_low": np.exp(res["beta"][i] - 1.96 * se[i]),
                                 "OR_ci_high": np.exp(res["beta"][i] + 1.96 * se[i])})
            res_all = res
    # drop-one-pillar likelihood-ratio tests
    try:
        for pn, cols in PILLARS.items():
            rest = [c for c in FEATURE_SETS[ALL_NAME] if c not in cols]
            r2 = fit_logit(design(rest), y)
            lr = 2 * (res_all["ll"] - r2["ll"])
            out_drop.append({"pillar_removed": pn, "df": len(cols), "LR_chi2": lr,
                             "p_value": stats.chi2.sf(lr, len(cols))})
    except Exception as e:
        print(f"  ! drop-one-pillar test failed: {e}")
    # VIF
    try:
        Z = design(FEATURE_SETS[ALL_NAME])
        corr = np.corrcoef(Z, rowvar=False)
        vif = np.diag(np.linalg.inv(corr + np.eye(len(corr)) * 1e-10))
        vif_df = pd.DataFrame({"feature": FEATURE_SETS[ALL_NAME], "VIF": vif}).sort_values("VIF", ascending=False)
    except Exception:
        vif_df = pd.DataFrame()
    return pd.DataFrame(out_coef), pd.DataFrame(out_sets), pd.DataFrame(out_drop), vif_df


# --------------------------------------------------------------------------
# Data-leakage diagnostics
# --------------------------------------------------------------------------
def leakage_integrity(df, folds, X):
    rows = []
    for f in folds:
        tr, te, dr = df.loc[f["train"]], df.loc[f["test"]], df.loc[f["dropped"]]
        gap = (te[DATE_COL].min() - tr[DATE_COL].max()).days
        overlap_idx = set(f["train"]) & set(f["test"])
        shared_sym = len(set(tr["symbol"]) & set(te["symbol"]))
        rows.append({"fold": f["year"],
                     "train_before_test": bool(tr[DATE_COL].max() < te[DATE_COL].min()),
                     "gap_days_train_last_to_test_first": gap,
                     "embargo_respected": bool((te[DATE_COL] > pd.Timestamp(f["year"] - 1, 12, 31)
                                                + pd.Timedelta(days=EMBARGO_DAYS - 1)).all()),
                     "dropped_not_in_test": len(set(f["dropped"]) & set(f["test"])) == 0,
                     "row_overlap_train_test": len(overlap_idx),
                     "companies_in_both_train_and_test": shared_sym,
                     "target_in_features": bool(TARGET in X.columns),
                     "test_dates_after_all_train_dates": bool(te[DATE_COL].min() > tr[DATE_COL].max())})
    return pd.DataFrame(rows)


def leakage_label_permutation(df, folds, caches, best_params, models, reps, seed, n_jobs, preds):
    """Shuffle TRAIN labels, refit with the tuned parameters, score the REAL test labels."""
    out, rng = [], np.random.default_rng(seed + 7)
    for m in list(models) + ["LOGIT"]:
        scaled = m in SCALED_MODELS
        real = preds[(preds["set"] == ALL_NAME) & (preds["model"] == m)]
        real_auc = fast_auc(real["y"], real["p"])
        perm_aucs = []
        for r in range(reps):
            ys, ps = [], []
            for f in folds:
                c = caches[(f["year"], ALL_NAME)]
                ytr = rng.permutation(c["ytr"])
                Xtr, Xte = (c["tr_sc"], c["te_sc"]) if scaled else (c["tr_raw"], c["te_raw"])
                try:
                    p = fit_predict(m, best_params[(f["year"], ALL_NAME, m)], Xtr, ytr, Xte,
                                    stable_seed(seed, r, f["year"], m), n_jobs)
                except Exception:
                    p = np.full(len(c["yte"]), 0.5)
                ys.append(c["yte"])
                ps.append(p)
            perm_aucs.append(fast_auc(np.concatenate(ys), np.concatenate(ps)))
        perm_aucs = np.array(perm_aucs)
        out.append({"model": MODEL_LABEL[m], "real_pooled_auc": real_auc, "perm_auc_mean": perm_aucs.mean(),
                    "perm_auc_sd": perm_aucs.std(ddof=1) if reps > 1 else np.nan, "perm_auc_max": perm_aucs.max(),
                    "p_value_real_vs_shuffled": (1 + np.sum(perm_aucs >= real_auc)) / (reps + 1),
                    "shuffled_labels_near_0.5": bool(abs(perm_aucs.mean() - 0.5) < 0.05)})
    return pd.DataFrame(out)


def leakage_univariate(df, X):
    y = df[TARGET].to_numpy()
    rows = []
    for c in [c for c in X.columns]:
        x = X[c].fillna(X[c].median()).to_numpy()
        a = fast_auc(y, x)
        rows.append({"feature": c, "univariate_auc": a, "abs_auc_from_0.5": abs(a - 0.5),
                     "corr_with_target": float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 else np.nan,
                     "flag_suspicious(auc>0.70 or <0.30)": bool(a > 0.70 or a < 0.30)})
    return pd.DataFrame(rows).sort_values("abs_auc_from_0.5", ascending=False)


def leakage_random_vs_chrono(df, X, folds, models, args):
    """Diagnostic only: default hyper-parameters on ALL features, random 5-fold vs chronological."""
    y = df[TARGET].to_numpy()
    cols = FEATURE_SETS[ALL_NAME]
    rows = []
    for m in models:
        scaled = m in SCALED_MODELS
        # chronological (same folds, default params)
        ys, ps = [], []
        for f in folds:
            prep = FoldPreprocessor(cols).fit(X.loc[f["train"]])
            Xtr, Xte = prep.transform(X.loc[f["train"]], scaled), prep.transform(X.loc[f["test"]], scaled)
            ps.append(fit_predict(m, default_params(m), Xtr, y[f["train"]], Xte, args.seed, args.jobs))
            ys.append(y[f["test"]])
        chrono = fast_auc(np.concatenate(ys), np.concatenate(ps))
        # random stratified 5-fold
        ys, ps = [], []
        skf = StratifiedKFold(5, shuffle=True, random_state=args.seed)
        for tr, te in skf.split(X, y):
            prep = FoldPreprocessor(cols).fit(X.iloc[tr])
            Xtr, Xte = prep.transform(X.iloc[tr], scaled), prep.transform(X.iloc[te], scaled)
            ps.append(fit_predict(m, default_params(m), Xtr, y[tr], Xte, args.seed, args.jobs))
            ys.append(y[te])
        rand = fast_auc(np.concatenate(ys), np.concatenate(ps))
        rows.append({"model": MODEL_LABEL[m], "chronological_auc_default_params": chrono,
                     "random_5fold_auc_default_params": rand, "gap_random_minus_chrono": rand - chrono})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Export of the preprocessed (imputed + winsorized) data each fold used
# --------------------------------------------------------------------------
def export_processed(df, X, sector, folds, caches, path):
    cols = FEATURE_SETS[ALL_NAME]
    meta = ["company_name", "symbol", DATE_COL, "Exchange", "underwriter_reputation", "industry"]
    param_rows, summ_rows, fold_frames = [], [], {}
    for f in folds:
        c = caches[(f["year"], ALL_NAME)]
        prep, tr, te = c["prep"], c["tr"], c["te"]
        raw = {"train": X.loc[tr, cols], "test": X.loc[te, cols]}
        proc = {"train": pd.DataFrame(c["tr_raw"], columns=cols, index=tr),
                "test": pd.DataFrame(c["te_raw"], columns=cols, index=te)}
        flags = {}
        for split in ("train", "test"):
            filled = raw[split].fillna(prep.fill_) if prep.fill_ else raw[split]
            flags[split] = (raw[split].isna(), filled != proc[split])
        for col in cols:
            lo, hi = prep.bounds_.get(col, (np.nan, np.nan))
            param_rows.append({"fold": f["year"], "feature": col,
                               "type": "continuous" if col in prep.cont else "binary/dummy",
                               "fill_value_from_train": prep.fill_.get(col, np.nan),
                               "winsor_lower_from_train": lo, "winsor_upper_from_train": hi,
                               "train_cells_imputed": int(flags["train"][0][col].sum()),
                               "train_cells_winsorized": int(flags["train"][1][col].sum()),
                               "test_cells_imputed": int(flags["test"][0][col].sum()),
                               "test_cells_winsorized": int(flags["test"][1][col].sum())})
        summ_rows.append({"fold": f["year"], "n_train": len(tr), "n_test": len(te),
                          "train_cells_imputed": int(flags["train"][0].to_numpy().sum()),
                          "train_cells_winsorized": int(flags["train"][1].to_numpy().sum()),
                          "test_cells_imputed": int(flags["test"][0].to_numpy().sum()),
                          "test_cells_winsorized": int(flags["test"][1].to_numpy().sum())})
        parts = []
        for split, idx in (("train", tr), ("test", te)):
            d = df.loc[idx, meta].copy()
            d.insert(0, "split", split)
            d["sector"] = sector.loc[idx].values
            d[TARGET] = df.loc[idx, TARGET].values
            parts.append(pd.concat([d, proc[split]], axis=1))
        fold_frames[f["year"]] = pd.concat(parts, ignore_index=True)
    readme = pd.DataFrame({"note": [
        "Each fold_YYYY sheet holds the data that fold's models used: 'train' rows = IPOs before 1 Jan YYYY, "
        "'test' rows = that year's test IPOs (embargoed January IPOs excluded).",
        "Values are AFTER imputation and winsorization, UNSCALED. Fill values and winsorization bounds were fitted on the "
        "fold's TRAIN rows only and applied to its test rows (see 'parameters').",
        "Binary flags are mode-imputed; categorical columns are shown as in the raw file, the *_ dummy columns "
        "(exch_*, uw_*, sec_*) are the encoded versions (reference: Nasdaq, Tier 1, Biotech & Pharma).",
        "Logistic models additionally apply median/IQR scaling (train-fitted) to the continuous columns; not shown here.",
        "log_proceeds is not a model feature (collinear with log_offer_price + log_shares_offered).",
        ("Preprocessing was SKIPPED (--as-given): the data is exactly as supplied." if not CLEAN_IN_FOLD else
         "Preprocessing: median/mode imputation, then iterative Tukey-fence winsorization (k=1.5, repeated until no outlier; "
         "1st/99th percentile fallback when IQR=0).")]})
    with pd.ExcelWriter(path) as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        pd.DataFrame(summ_rows).to_excel(xw, sheet_name="summary", index=False)
        pd.DataFrame(param_rows).to_excel(xw, sheet_name="parameters", index=False)
        for yr, fr in fold_frames.items():
            fr.to_excel(xw, sheet_name=f"fold_{yr}", index=False)
    print(f"Saved preprocessed data by fold -> {path}")


# --------------------------------------------------------------------------
# SHAP in probability units
# --------------------------------------------------------------------------
def perm_shapley(predict_fn, bg, X, m_perm=1, seed=0):
    """Interventional permutation-Shapley estimate in the model's output space (probability).
    For each background row k and each of m_perm random feature orders (plus the reversed
    order, antithetic sampling) features are switched from the background value to the
    explained value one at a time. Contributions telescope, so the values add up exactly to
    f(x) - mean_k f(bg_k)."""
    rng = np.random.default_rng(seed)
    K, f = bg.shape
    steps = np.arange(f + 1)[None, :, None]
    phi = np.zeros((len(X), f))
    bg_idx = np.tile(np.repeat(np.arange(K), m_perm), 2)
    for i in range(len(X)):
        perms = rng.permuted(np.tile(np.arange(f), (K * m_perm, 1)), axis=1)
        perms = np.vstack([perms, perms[:, ::-1]])
        pos = np.argsort(perms, axis=1)
        H = np.where(pos[:, None, :] < steps, X[i][None, None, :], bg[bg_idx][:, None, :])
        pr = predict_fn(H.reshape(-1, f)).reshape(len(perms), f + 1)
        d = pr[:, 1:] - pr[:, :-1]
        np.add.at(phi[i], perms.ravel(), d.ravel())
        phi[i] /= len(perms)
    return phi


def shap_values_prob(name, mdl, Xbg, Xte, args, seed):
    """Return (phi[n, f] in probability units, base value, engine label)."""
    fn = lambda A: mdl.predict_proba(A)[:, 1]
    if args.shap_engine == "auto":
        try:
            import shap
            if name in ("RF", "XGB", "LGBM", "GB"):
                ex = shap.TreeExplainer(mdl, data=Xbg, feature_perturbation="interventional",
                                        model_output="probability")
                sv = ex.shap_values(Xte, check_additivity=False)
                sv = np.asarray(sv[1] if isinstance(sv, list) else sv)
                if sv.ndim == 3:
                    sv = sv[:, :, 1]
                return sv, float(np.ravel(ex.expected_value)[-1]), "shap.TreeExplainer (probability)"
            ex = shap.PermutationExplainer(fn, shap.maskers.Independent(Xbg, max_samples=len(Xbg)))
            e = ex(Xte)
            return np.asarray(e.values), float(np.mean(e.base_values)), "shap.PermutationExplainer (probability)"
        except ImportError:
            pass
        except Exception as e:
            print(f"  ! shap package failed for {name} ({e}); using built-in estimator")
    phi = perm_shapley(fn, Xbg, Xte, args.shap_perms, seed)
    return phi, float(fn(Xbg).mean()), "built-in permutation Shapley (probability)"


def _plot_bar(imp, label, path, top=15):
    import matplotlib.pyplot as plt
    d = imp.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(d["feature"], d["mean_abs_shap_pp"], color="#3b6fb6")
    ax.set_xlabel("Mean |SHAP| (percentage points of predicted probability)")
    ax.set_title(f"{label}: top features")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_beeswarm(phi, vals, feats, label, path, seed):
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for row, c in enumerate(feats[::-1]):
        v = vals[c].to_numpy()
        col = rankdata(v) / len(v)
        ax.scatter(phi[c].to_numpy() * 100, row + rng.normal(0, 0.09, len(v)), c=col, cmap="coolwarm",
                   s=9, alpha=0.75, linewidths=0)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats[::-1], fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("SHAP value (percentage-point change in predicted probability of underpricing)")
    ax.set_title(f"{label}: SHAP summary (red = high feature value, blue = low)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_pillars(pill, path):
    import matplotlib.pyplot as plt
    wide = pill.pivot(index="pillar", columns="model", values="mean_abs_shap_pp").reindex(list(PILLARS))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    wide.plot.barh(ax=ax, width=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |pillar SHAP| (percentage points of predicted probability)")
    ax.set_title("Contribution of each pillar to predictions (out-of-sample)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_shap_report(df, folds, caches, fitted, models, args, best=None):
    import base64
    cols = FEATURE_SETS[ALL_NAME]
    pillar_of = {c: p for p, cs in PILLARS.items() for c in cs}
    mlist = [m for m in ALL_MODEL_CODES if m in models and any((f["year"], m) in fitted for f in folds)]
    long, engines = {}, {}
    for m in mlist:
        ph, mt, vl = [], [], []
        for f in folds:
            mdl = fitted.get((f["year"], m))
            if mdl is None:
                continue
            c = caches[(f["year"], ALL_NAME)]
            Xtr, Xte = (c["tr_sc"], c["te_sc"]) if m in SCALED_MODELS else (c["tr_raw"], c["te_raw"])
            seed = stable_seed(args.seed, "shap", f["year"], m)
            bg = Xtr[np.random.default_rng(seed).choice(len(Xtr), min(args.shap_bg, len(Xtr)), replace=False)]
            phi, base, eng = shap_values_prob(m, mdl, bg, Xte, args, seed)
            engines[m] = eng
            ph.append(pd.DataFrame(phi, columns=cols))
            vl.append(pd.DataFrame(c["te_raw"], columns=cols))
            mt.append(pd.DataFrame({"model": MODEL_LABEL[m], "fold": f["year"],
                                    "symbol": df.loc[c["te"], "symbol"].values,
                                    "ipo_date": df.loc[c["te"], DATE_COL].values, "underpriced": c["yte"],
                                    "predicted_prob": mdl.predict_proba(Xte)[:, 1], "base_value": base}))
        long[m] = (pd.concat(mt, ignore_index=True), pd.concat(ph, ignore_index=True), pd.concat(vl, ignore_index=True))
        print(f"  SHAP {MODEL_LABEL[m]}: {engines[m]}")

    glob, pill, year, add, drv = [], [], [], [], []
    for m, (meta, phi, vals) in long.items():
        label = MODEL_LABEL[m]
        mabs = phi.abs().mean() * 100
        for c in cols:
            v, sv = vals[c].to_numpy(), phi[c].to_numpy()
            corr = stats.spearmanr(v, sv).statistic if np.std(v) > 0 and np.std(sv) > 0 else np.nan
            glob.append({"model": label, "feature": c, "pillar": pillar_of[c], "mean_abs_shap_pp": mabs[c],
                         "share_of_total_pct": 100 * mabs[c] / mabs.sum(), "mean_signed_shap_pp": sv.mean() * 100,
                         "direction_spearman(value,shap)": corr})
        pabs = {p: phi[cs].sum(axis=1).abs().mean() * 100 for p, cs in PILLARS.items()}
        for p, v in pabs.items():
            pill.append({"model": label, "pillar": p, "mean_abs_shap_pp": v,
                         "share_of_total_pct": 100 * v / sum(pabs.values())})
        for fy in meta["fold"].unique():
            mk = (meta["fold"] == fy).to_numpy()
            for c, v in (phi[mk].abs().mean() * 100).items():
                year.append({"model": label, "fold": fy, "feature": c, "mean_abs_shap_pp": v})
        err = (meta["base_value"] + phi.sum(axis=1) - meta["predicted_prob"]).abs()
        add.append({"model": label, "engine": engines[m], "n_explained": len(meta),
                    "max_abs_additivity_error": err.max(), "mean_abs_additivity_error": err.mean()})
        arr = phi.to_numpy()
        for i in range(len(meta)):
            o = np.argsort(arr[i])
            neg = [f"{cols[j]} ({arr[i, j] * 100:+.1f}pp)" for j in o[:3] if arr[i, j] < 0]
            pos = [f"{cols[j]} ({arr[i, j] * 100:+.1f}pp)" for j in o[::-1][:3] if arr[i, j] > 0]
            drv.append({"model": label, "fold": meta.at[i, "fold"], "symbol": meta.at[i, "symbol"],
                        "ipo_date": meta.at[i, "ipo_date"], "underpriced": meta.at[i, "underpriced"],
                        "predicted_prob": meta.at[i, "predicted_prob"], "base_value": meta.at[i, "base_value"],
                        "top_pushing_up": "; ".join(pos), "top_pushing_down": "; ".join(neg)})
    glob, pill, year = pd.DataFrame(glob), pd.DataFrame(pill), pd.DataFrame(year)
    add, drv = pd.DataFrame(add), pd.DataFrame(drv)

    xlsx = os.path.join(args.outdir, "shap_values_probability.xlsx")
    readme = pd.DataFrame({"note": [
        "SHAP values are in PROBABILITY units: each IPO's values add up to (predicted probability - base_value). "
        "'pp' = percentage points.",
        "Computed OUT-OF-SAMPLE: each fold's fitted model (ALL-pillars features) explains that fold's test IPOs only.",
        "The models are class-weighted (balanced), so predicted probabilities and base values sit near 0.5 rather "
        "than the raw 61% underpricing rate.",
        "Mean |SHAP| = how much a feature moves predictions on average; direction_spearman = correlation between "
        "feature value and its SHAP value (+ = higher value raises the predicted chance of underpricing).",
        "SHAP describes what the MODEL relies on. It is not evidence of causation, and correlated features share credit.",
        "Engines: " + "; ".join(f"{MODEL_LABEL[m]} = {e}" for m, e in engines.items())]})
    with pd.ExcelWriter(xlsx) as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        glob.sort_values(["model", "mean_abs_shap_pp"], ascending=[True, False]).to_excel(
            xw, sheet_name="global_importance", index=False)
        pill.to_excel(xw, sheet_name="pillar_importance", index=False)
        year.to_excel(xw, sheet_name="yearly_importance", index=False)
        add.to_excel(xw, sheet_name="additivity_check", index=False)
        drv.to_excel(xw, sheet_name="top_drivers_per_ipo", index=False)
        for m, (meta, phi, vals) in long.items():
            pd.concat([meta, phi], axis=1).to_excel(xw, sheet_name=f"shap_{m}", index=False)

    # figures + HTML report
    try:
        import matplotlib
        matplotlib.use("Agg")
    except Exception:
        print("  matplotlib missing: skipping SHAP figures/HTML")
        return {"glob": glob, "pill": pill, "year": year, "add": add, "engines": engines, "imgs": {},
                "top_feats": [], "xlsx": xlsx}
    imgs = {}
    for m, (meta, phi, vals) in long.items():
        label = MODEL_LABEL[m]
        imp = glob[glob["model"] == label].sort_values("mean_abs_shap_pp", ascending=False)
        p1 = os.path.join(args.outdir, f"shap_{m}_bar.png")
        p2 = os.path.join(args.outdir, f"shap_{m}_beeswarm.png")
        _plot_bar(imp, label, p1)
        _plot_beeswarm(phi, vals, list(imp["feature"].head(12)), label, p2, args.seed)
        imgs[m] = (p1, p2)
        print(f"  {label} top features (pp): " + ", ".join(
            f"{r.feature} {r.mean_abs_shap_pp:.2f}" for r in imp.head(6).itertuples()))
    p3 = os.path.join(args.outdir, "shap_pillars.png")
    _plot_pillars(pill, p3)

    # extra figures for the best model: dependence plots + stability across test years
    import matplotlib.pyplot as plt
    bm = best if best in long else list(long)[0]
    _, phi_b, vals_b = long[bm]
    blab = MODEL_LABEL[bm]
    imp_b = glob[glob["model"] == blab].sort_values("mean_abs_shap_pp", ascending=False)
    top6 = list(imp_b["feature"].head(6))
    dep_png = os.path.join(args.outdir, f"shap_{bm}_dependence.png")
    try:
        def _dep(ax, col):
            ax.scatter(vals_b[col], phi_b[col] * 100, s=8, alpha=0.6, color="#3b6fb6", linewidths=0)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_title(col if len(col) < 34 else col[:33] + "…", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_ylabel("SHAP (pp)", fontsize=7)
        _plot_panels(top6, dep_png, 3, _dep, suptitle=f"{blab}: SHAP dependence (feature value vs effect on predicted probability, pp)")
    except Exception as e:
        print("  dependence plot skipped:", e)
        dep_png = None
    yearly_png = os.path.join(args.outdir, f"shap_{bm}_yearly.png")
    try:
        yb = year[year["model"] == blab]
        piv = yb.pivot(index="feature", columns="fold", values="mean_abs_shap_pp").reindex(list(imp_b["feature"].head(10)))
        fig, ax = plt.subplots(figsize=(9, 4.8))
        im = ax.imshow(piv.to_numpy(), cmap="Blues", aspect="auto")
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels([str(c_) for c_ in piv.columns], fontsize=8)
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels([c_ if len(c_) < 34 else c_[:33] + "…" for c_ in piv.index], fontsize=8)
        for i_ in range(piv.shape[0]):
            for j_ in range(piv.shape[1]):
                ax.text(j_, i_, f"{piv.iat[i_, j_]:.1f}", ha="center", va="center", fontsize=6.5)
        fig.colorbar(im, ax=ax, fraction=0.03).set_label("mean |SHAP| (pp)")
        ax.set_title(f"{blab}: feature importance by test year", fontsize=10)
        fig.tight_layout()
        fig.savefig(yearly_png, dpi=160)
        plt.close(fig)
    except Exception as e:
        print("  yearly SHAP plot skipped:", e)
        yearly_png = None

    b64 = lambda path: base64.b64encode(open(path, "rb").read()).decode()
    html = ["<html><head><meta charset='utf-8'><title>SHAP report - IPO underpricing</title>",
            "<style>body{font-family:Arial;max-width:1000px;margin:auto;padding:16px}table{border-collapse:collapse;"
            "font-size:13px}td,th{border:1px solid #ccc;padding:4px 8px}img{max-width:100%}</style></head><body>",
            "<h1>SHAP explanation report: U.S. IPO underpricing</h1>",
            "<p>Every value is in <b>probability units</b> (percentage points, pp, of the predicted probability that an IPO "
            "is underpriced). Values are out-of-sample: each fold's model explains that fold's test IPOs (expanding window, "
            "ALL-pillars feature set). For each IPO, base value + sum of SHAP values = predicted probability.</p>",
            "<p><b>Reading the charts.</b> Bars show the average absolute effect of each feature. In the dot plots every dot is an "
            "IPO: position = how much the feature moved that IPO's predicted probability, colour = the feature's value "
            "(red high, blue low).</p>",
            "<p><b>Caveats.</b> SHAP shows what the model relies on, not what causes underpricing; correlated features share "
            "credit; the models are class-weighted so base values sit near 0.5.</p>"]
    for m, (p1, p2) in imgs.items():
        label = MODEL_LABEL[m]
        top = glob[glob["model"] == label].sort_values("mean_abs_shap_pp", ascending=False).head(10)[
            ["feature", "pillar", "mean_abs_shap_pp", "share_of_total_pct", "direction_spearman(value,shap)"]]
        html.append(f"<h2>{label}</h2><p>Engine: {engines[m]}</p>")
        html.append(top.round(3).to_html(index=False))
        html.append(f"<img src='data:image/png;base64,{b64(p1)}'/><img src='data:image/png;base64,{b64(p2)}'/>")
    html.append("<h2>Pillar contributions</h2>")
    html.append(f"<img src='data:image/png;base64,{b64(p3)}'/>")
    html.append(pill.pivot(index="pillar", columns="model", values="mean_abs_shap_pp").reindex(list(PILLARS))
                .round(3).to_html())
    html.append("<h2>Additivity check</h2>" + add.round(5).to_html(index=False))
    html.append("</body></html>")
    path = os.path.join(args.outdir, "shap_report.html")
    open(path, "w", encoding="utf-8").write("\n".join(html))
    print(f"Saved SHAP report -> {path}\n                     {xlsx}")
    return {"glob": glob, "pill": pill, "year": year, "add": add, "engines": engines, "imgs": imgs,
            "pillars_png": p3, "dep_png": dep_png, "yearly_png": yearly_png, "best": bm, "xlsx": xlsx,
            "top_feats": [(r.feature, r.mean_abs_shap_pp) for r in imp_b.head(10).itertuples()]}


# --------------------------------------------------------------------------
# DESCRIPTIVE STATISTICS  (TRAINING rows of ONE fold only -> no test-set leakage)
# --------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    """Wilson 95% interval for a proportion."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def rate_table(groups, y, var, order=None):
    """Underpricing rate (with Wilson CI) per group + chi-square test of independence."""
    d = pd.DataFrame({"g": pd.Series(groups).astype(str).values, "y": np.asarray(y)})
    order = list(order) if order is not None else sorted(d["g"].unique())
    rows = []
    for g in order:
        s = d.loc[d["g"] == g, "y"]
        if len(s) == 0:
            continue
        lo, hi = wilson(int(s.sum()), len(s))
        rows.append({"variable": var, "group": g, "n": len(s), "n_underpriced": int(s.sum()),
                     "underpricing_rate_pct": 100 * s.mean(), "ci_low_pct": 100 * lo, "ci_high_pct": 100 * hi,
                     "share_of_sample_pct": 100 * len(s) / len(d)})
    tab = pd.DataFrame(rows)
    try:
        ct = pd.crosstab(d["g"], d["y"])
        chi = stats.chi2_contingency(ct)
        tab.attrs["chi2_p"] = float(chi[1])
        tab.attrs["min_expected"] = float(chi[3].min())
    except Exception:
        tab.attrs["chi2_p"], tab.attrs["min_expected"] = np.nan, np.nan
    return tab


def _short(s, n=26):
    return s if len(s) <= n else s[:n - 1] + "…"


def _plot_rate(tab, title, path, overall, horizontal=False, figsize=(8, 4.2), label_len=26):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(tab))
    r = tab["underpricing_rate_pct"].to_numpy()
    err = np.vstack([r - tab["ci_low_pct"].to_numpy(), tab["ci_high_pct"].to_numpy() - r])
    labels = [_short(str(g), label_len) for g in tab["group"]]
    if horizontal:
        ax.barh(x, r, xerr=err, color="#3b6fb6", capsize=2, alpha=0.9)
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(overall, color="k", ls="--", lw=1)
        ax.set_xlabel("Underpricing rate (% of IPOs, 95% Wilson CI)")
        ax.set_xlim(0, 105)
        for i, n in enumerate(tab["n"]):
            ax.text(101, i, f"n={n}", va="center", ha="right", fontsize=7, color="#444")
    else:
        ax.bar(x, r, yerr=err, color="#3b6fb6", capsize=2, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=45 if len(tab) > 6 else 0, ha="right" if len(tab) > 6 else "center")
        ax.axhline(overall, color="k", ls="--", lw=1)
        ax.set_ylabel("Underpricing rate (% of IPOs, 95% Wilson CI)")
        ax.set_ylim(0, 105)
        for i, n in enumerate(tab["n"]):
            ax.text(i, 101, f"n={n}", ha="center", va="top", fontsize=7, color="#444")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_panels(items, path, ncols, draw, panel_h=2.5, panel_w=3.6, suptitle=None):
    import matplotlib.pyplot as plt
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, panel_h * nrows + (0.3 if suptitle else 0)))
    axes = np.atleast_1d(axes).ravel()
    for ax, it in zip(axes, items):
        draw(ax, it)
    for ax in axes[len(items):]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def descriptive_analysis(df, X, sector, folds, caches, args):
    """All descriptive statistics and charts use ONLY the training rows of one fold (first or last).
    Continuous features are shown after that fold's train-fitted imputation + winsorization (what the models saw);
    categorical variables come from the raw workbook."""
    which = folds[-1] if args.desc_fold == "last" else folds[0]
    fy, tr = which["year"], which["train"]
    cols = FEATURE_SETS[ALL_NAME]
    c = caches[(fy, ALL_NAME)]
    P = pd.DataFrame(c["tr_raw"], columns=cols, index=tr)
    D = df.loc[tr].copy()
    D["sector"] = sector.loc[tr].values
    D["year"] = D[DATE_COL].dt.year
    D["underwriter_tier"] = D["underwriter_reputation"].fillna("Unknown")
    D["Exchange"] = D["Exchange"].fillna("exch_nasdaq")
    y = D[TARGET].to_numpy().astype(int)
    overall = 100 * y.mean()
    pillar_of = {c_: p for p, cs in PILLARS.items() for c_ in cs}
    cont = [c_ for c_ in cols if c_ in CONT_ALL]
    out = {"fold_year": fy, "which": args.desc_fold, "n_train": len(tr), "overall_rate": overall,
           "date_min": D[DATE_COL].min(), "date_max": D[DATE_COL].max(), "tables": {}, "figs": {}, "notes": {}}
    T, F = out["tables"], out["figs"]
    tag = f"fold{fy}"

    # ---- continuous features: summary + class comparison (training rows only)
    rows = []
    for col in cont:
        v, raw = P[col].to_numpy(), X.loc[tr, col]
        g1, g0 = v[y == 1], v[y == 0]
        try:
            mw = stats.mannwhitneyu(g1, g0, alternative="two-sided").pvalue
        except Exception:
            mw = np.nan
        rows.append({"feature": col, "pillar": pillar_of[col], "n_missing_raw": int(raw.isna().sum()),
                     "mean": v.mean(), "sd": v.std(ddof=1), "min": v.min(), "p25": np.percentile(v, 25),
                     "median": np.median(v), "p75": np.percentile(v, 75), "max": v.max(),
                     "skew": float(stats.skew(v)), "raw_min": raw.min(), "raw_max": raw.max(),
                     "mean_underpriced": g1.mean(), "mean_not_underpriced": g0.mean(),
                     "median_underpriced": np.median(g1), "median_not_underpriced": np.median(g0),
                     "mannwhitney_p": mw, "univariate_auc": fast_auc(y, v),
                     "spearman_with_target": float(stats.spearmanr(v, y).statistic)})
    T["continuous_summary"] = pd.DataFrame(rows)

    # ---- binary flags / dummies
    flag_cols = BINARY
    rows, rate_flag = [], []
    for col in flag_cols:
        t = rate_table(P[col].astype(int).map({1: "Yes (1)", 0: "No (0)"}), y, col, ["No (0)", "Yes (1)"])
        rate_flag.append(t)
        rows.append({"feature": col, "share_equal_1_pct": 100 * P[col].mean(),
                     "rate_if_1_pct": t.loc[t["group"] == "Yes (1)", "underpricing_rate_pct"].squeeze() if (t["group"] == "Yes (1)").any() else np.nan,
                     "rate_if_0_pct": t.loc[t["group"] == "No (0)", "underpricing_rate_pct"].squeeze() if (t["group"] == "No (0)").any() else np.nan,
                     "chi2_p": t.attrs["chi2_p"]})
    T["binary_summary"] = pd.DataFrame(rows)
    T["by_binary_flags"] = pd.concat(rate_flag, ignore_index=True)

    # ---- categorical breakdowns
    years = sorted(D["year"].unique())
    T["by_year"] = rate_table(D["year"], y, "ipo_year", [str(v) for v in years])
    T["by_sector"] = rate_table(D["sector"], y, "sector", SECTORS)
    ind_n = D["industry"].fillna("Unknown").value_counts()
    top = list(ind_n.index[:15])
    ind = D["industry"].fillna("Unknown").where(D["industry"].fillna("Unknown").isin(top), "All other industries")
    T["by_industry"] = rate_table(ind, y, "industry (top 15 by n)", top + ["All other industries"])
    tiers = [t for t in ["Tier 1", "Tier 2", "Other", "Unknown"] if t in set(D["underwriter_tier"])]
    T["by_underwriter"] = rate_table(D["underwriter_tier"], y, "underwriter_reputation", tiers)
    T["by_exchange"] = rate_table(D["Exchange"], y, "Exchange", [e for e in ["exch_nasdaq", "exch_nyse", "exch_nyse_american"]
                                                                  if e in set(D["Exchange"])])
    chi_rows = []
    for k in ("by_year", "by_sector", "by_industry", "by_underwriter", "by_exchange"):
        chi_rows.append({"breakdown": T[k]["variable"].iloc[0], "n_groups": len(T[k]),
                         "chi2_p_value": T[k].attrs["chi2_p"], "min_expected_count": T[k].attrs["min_expected"]})
    for t in rate_flag:
        chi_rows.append({"breakdown": t["variable"].iloc[0], "n_groups": 2, "chi2_p_value": t.attrs["chi2_p"],
                         "min_expected_count": t.attrs["min_expected"]})
    T["independence_tests"] = pd.DataFrame(chi_rows)

    # ---- quintiles of every continuous feature
    qrows = []
    for col in cont:
        s = P[col]
        if s.nunique() < 3:
            continue
        q = pd.qcut(s, 5, duplicates="drop")
        cats = list(q.cat.categories)
        lab = {cat: f"Q{i + 1}: {cat.left:.3g} to {cat.right:.3g}" for i, cat in enumerate(cats)}
        t = rate_table(q.map(lab), y, col, list(lab.values()))
        t["pillar"] = pillar_of[col]
        t["chi2_p"] = t.attrs["chi2_p"]
        qrows.append(t)
    T["by_quintile"] = pd.concat(qrows, ignore_index=True)

    # ---- yearly feature means (the training IPOs only)
    ym = P[cont].groupby(D["year"].values).mean()
    ym["n_ipos"] = D.groupby("year").size()
    ym["underpricing_rate_pct"] = D.groupby("year")[TARGET].mean() * 100
    T["yearly_feature_means"] = ym.reset_index().rename(columns={"index": "year"})

    # ---- correlation
    corr = P[cont].corr(method="spearman")
    T["spearman_corr"] = corr.reset_index().rename(columns={"index": "feature"})
    pairs = [{"feature_A": a, "feature_B": b, "spearman": corr.loc[a, b]}
             for i, a in enumerate(cont) for b in cont[i + 1:] if abs(corr.loc[a, b]) >= 0.5]
    T["high_corr_pairs"] = (pd.DataFrame(pairs).assign(abs_rho=lambda d_: d_["spearman"].abs())
                            .sort_values("abs_rho", ascending=False).drop(columns="abs_rho")
                            if pairs else pd.DataFrame(columns=["feature_A", "feature_B", "spearman"]))

    # ---- charts
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  matplotlib missing: descriptive charts skipped")
        return out
    od = args.outdir
    p = os.path.join(od, f"desc_{tag}_by_year.png")
    ty = T["by_year"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(ty))
    r = ty["underpricing_rate_pct"].to_numpy()
    ax.bar(x, r, yerr=np.vstack([r - ty["ci_low_pct"], ty["ci_high_pct"] - r]), color="#3b6fb6", capsize=2, alpha=0.9)
    ax.axhline(overall, color="k", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(ty["group"], fontsize=8)
    ax.set_ylabel("Underpricing rate (%, 95% Wilson CI)")
    ax.set_ylim(0, 105)
    ax2 = ax.twinx()
    ax2.plot(x, ty["n"], color="#d1495b", marker="o", lw=1.5)
    ax2.set_ylabel("Number of IPOs", color="#d1495b")
    ax.set_title(f"Underpricing by IPO year (training rows of fold {fy}; dashed = overall {overall:.1f}%)", fontsize=10)
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    F["by_year"] = p
    for key, title, horiz, size in (("by_sector", "Underpricing by sector", True, (8, 4.2)),
                                    ("by_industry", "Underpricing by industry (15 largest)", True, (8.5, 6)),
                                    ("by_underwriter", "Underpricing by underwriter reputation", False, (6.5, 4)),
                                    ("by_exchange", "Underpricing by exchange", False, (6, 4)),
                                    ("by_binary_flags", "Underpricing by governance / certification flags", True, (8, 3.6))):
        t = T[key].copy()
        if key == "by_binary_flags":
            t["group"] = t["variable"].str.replace("_", " ") + ": " + t["group"]
        p = os.path.join(od, f"desc_{tag}_{key}.png")
        _plot_rate(t, f"{title} (training rows of fold {fy})", p, overall, horizontal=horiz, figsize=size,
                   label_len=42 if key in ("by_binary_flags", "by_industry") else 26)
        F[key] = p
    # quintile panels
    qf = [c_ for c_ in cont if (T["by_quintile"]["variable"] == c_).any()]
    qpaths = []
    for i in range(0, len(qf), 6):
        chunk = qf[i:i + 6]
        pth = os.path.join(od, f"desc_{tag}_quintiles_{i // 6 + 1}.png")

        def dq(ax, col):
            t = T["by_quintile"][T["by_quintile"]["variable"] == col]
            r_ = t["underpricing_rate_pct"].to_numpy()
            ax.bar(np.arange(len(t)), r_, yerr=np.vstack([r_ - t["ci_low_pct"], t["ci_high_pct"] - r_]),
                   color="#3b6fb6", capsize=1.5, alpha=0.9)
            ax.axhline(overall, color="k", ls="--", lw=0.8)
            ax.set_xticks(np.arange(len(t)))
            ax.set_xticklabels([f"Q{j + 1}" for j in range(len(t))], fontsize=7)
            ax.set_ylim(0, 105)
            ax.set_title(_short(col, 34), fontsize=8)
            ax.tick_params(axis="y", labelsize=7)
        _plot_panels(chunk, pth, 3, dq, suptitle="Underpricing rate (%) by quintile of each feature (Q1 = lowest, training rows only)")
        qpaths.append(pth)
    F["quintiles"] = qpaths
    # distributions by class
    hpaths = []
    for i in range(0, len(cont), 6):
        chunk = cont[i:i + 6]
        pth = os.path.join(od, f"desc_{tag}_distributions_{i // 6 + 1}.png")

        def dh(ax, col):
            v = P[col].to_numpy()
            bins = np.histogram_bin_edges(v, bins=20)
            ax.hist(v[y == 1], bins=bins, alpha=0.6, density=True, color="#3b6fb6", label="underpriced")
            ax.hist(v[y == 0], bins=bins, alpha=0.6, density=True, color="#d1495b", label="not underpriced")
            ax.set_title(_short(col, 34), fontsize=8)
            ax.tick_params(labelsize=7)
        _plot_panels(chunk, pth, 3, dh, suptitle="Distribution by outcome (blue = underpriced, red = not; training rows, processed values)")
        hpaths.append(pth)
    F["distributions"] = hpaths
    # correlation heat-map
    p = os.path.join(od, f"desc_{tag}_correlation.png")
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cont)))
    ax.set_yticks(range(len(cont)))
    ax.set_xticklabels([_short(c_, 22) for c_ in cont], rotation=90, fontsize=7)
    ax.set_yticklabels([_short(c_, 22) for c_ in cont], fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046).set_label("Spearman rho")
    ax.set_title(f"Spearman correlation of continuous features (training rows, fold {fy})", fontsize=10)
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    F["correlation"] = p
    # year x feature heat-map (z-scored year means)
    p = os.path.join(od, f"desc_{tag}_yearly_features.png")
    z = (ym[cont] - ym[cont].mean()) / ym[cont].std(ddof=0).replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(z.T.to_numpy(), cmap="coolwarm", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_xticks(range(len(ym)))
    ax.set_xticklabels([str(v) for v in ym.index], fontsize=8)
    ax.set_yticks(range(len(cont)))
    ax.set_yticklabels([_short(c_, 30) for c_ in cont], fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.03).set_label("z-score of the year's mean")
    ax.set_title("Mean of each feature by IPO year (standardised across years; training rows only)", fontsize=10)
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    F["yearly_features"] = p
    # univariate AUC bar
    p = os.path.join(od, f"desc_{tag}_univariate_auc.png")
    u = T["continuous_summary"].sort_values("univariate_auc")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh([_short(v, 34) for v in u["feature"]], u["univariate_auc"] - 0.5, left=0.5,
            color=["#3b6fb6" if v >= 0.5 else "#d1495b" for v in u["univariate_auc"]])
    ax.axvline(0.5, color="k", lw=1)
    ax.set_xlabel("Univariate AUC for 'underpriced' (training rows; 0.5 = no signal)")
    ax.set_title("Single-feature discrimination", fontsize=10)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    F["univariate_auc"] = p

    xlsx = os.path.join(od, f"descriptive_statistics_train_fold{fy}.xlsx")
    readme = pd.DataFrame({"note": [
        f"ALL statistics use ONLY the TRAINING rows of the {args.desc_fold.upper()} fold (test year {fy}): "
        f"{len(tr)} IPOs dated {out['date_min']:%Y-%m-%d} to {out['date_max']:%Y-%m-%d}. No test-year IPO is used, so nothing "
        "learnt here can leak into evaluation.",
        "Continuous features are shown after that fold's TRAIN-fitted imputation and iterative winsorization "
        "(raw_min/raw_max show the unwinsorized range). Categorical variables are taken from the raw workbook.",
        "Underpricing rate = share of IPOs with underpriced = 1. 95% intervals are Wilson intervals; chi-square tests "
        "check independence between each grouping and the outcome (see 'independence_tests'; "
        "treat cells with expected counts < 5 with caution).",
        "The workbook holds the binary outcome only (no first-day return), so magnitude of underpricing is not analysed.",
        "Descriptive associations are not causal and are not corrected for multiple testing."]})
    with pd.ExcelWriter(xlsx) as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        for k, t in T.items():
            t.to_excel(xw, sheet_name=k[:31], index=False)
    out["xlsx"] = xlsx
    print(f"Saved descriptive statistics (training rows of fold {fy}) -> {xlsx}")
    return out


# --------------------------------------------------------------------------
# ABLATION: how much does each common validation flaw inflate AUC?
# (integrated from ablation_flaws.py; same data, features, models, FIXED default hyper-parameters)
# --------------------------------------------------------------------------
ABL_SCENARIOS = {
    "S0 Our design (chrono + embargo + in-fold prep)": dict(kind="chrono", embargo=True, prep="infold", smote=False),
    "S1 No embargo": dict(kind="chrono", embargo=False, prep="infold", smote=False),
    "S2 Global preprocessing": dict(kind="chrono", embargo=True, prep="global", smote=False),
    "S3 Random 5-fold CV": dict(kind="random", prep="infold", smote=False),
    "S4 Random CV + global preprocessing": dict(kind="random", prep="global", smote=False),
    "S5 Random CV + global prep + SMOTE before split": dict(kind="random", prep="global", smote=True),
}
ABL_BASE = list(ABL_SCENARIOS)[0]


def abl_smote_pairs(Zs, y, k, rng):
    minority = 0 if (y == 0).sum() <= (y == 1).sum() else 1
    mi = np.where(y == minority)[0]
    n_new = int((y != minority).sum() - len(mi))
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(mi))).fit(Zs[mi])
    nb = nn.kneighbors(Zs[mi], return_distance=False)[:, 1:]
    pick = rng.integers(0, len(mi), n_new)
    nbr = nb[pick, rng.integers(0, nb.shape[1], n_new)]
    return mi[pick], mi[nbr], rng.random(n_new), minority


def abl_interpolate(Z, a, b, lam):
    return Z[a] + lam[:, None] * (Z[b] - Z[a])


def abl_prep_global(X, cols):
    prep = FoldPreprocessor(cols).fit(X)
    return prep.transform(X), prep.transform(X, True)


def abl_fitpred(m, Xtr, ytr, Xte, seed, jobs):
    return fit_predict(m, default_params(m), Xtr, ytr, Xte, seed, jobs)


def abl_run_chrono(X, y, cols, folds, m, prep_mode, seed, jobs):
    p = np.full(len(y), np.nan)
    scaled = m in SCALED_MODELS
    if prep_mode == "global":
        Zr, Zs = abl_prep_global(X, cols)
        Z = Zs if scaled else Zr
    for f in folds:
        tr, te = f["train"], f["test"]
        if prep_mode == "infold":
            prep = FoldPreprocessor(cols).fit(X.loc[tr])
            Xtr, Xte = prep.transform(X.loc[tr], scaled), prep.transform(X.loc[te], scaled)
        else:
            Xtr, Xte = Z[tr], Z[te]
        p[te] = abl_fitpred(m, Xtr, y[tr], Xte, stable_seed(seed, f["year"], m), jobs)
    return p[None, :], [np.nan]


def abl_run_random(X, y, cols, m, prep_mode, use_smote, seed, jobs, repeats):
    n = len(y)
    scaled = m in SCALED_MODELS
    P, own = np.full((repeats, n), np.nan), []
    if prep_mode == "global":
        Zr, Zs = abl_prep_global(X, cols)
    for r in range(repeats):
        rs = stable_seed(seed, "random", r)
        if use_smote:                               # SMOTE on the WHOLE dataset, then split
            rng = np.random.default_rng(rs)
            a, b, lam, mino = abl_smote_pairs(Zs, y, 5, rng)
            base = Zs if scaled else Zr
            Z = np.vstack([base, abl_interpolate(base, a, b, lam)])
            ya = np.concatenate([y, np.full(len(a), mino)])
            skf = StratifiedKFold(5, shuffle=True, random_state=rs % (2 ** 31))
            oof = np.full(len(ya), np.nan)
            for tr, te in skf.split(Z, ya):
                oof[te] = abl_fitpred(m, Z[tr], ya[tr], Z[te], stable_seed(seed, r, m), jobs)
            P[r] = oof[:n]
            own.append(fast_auc(ya, oof))           # includes synthetic rows (what such papers report)
        else:
            skf = StratifiedKFold(5, shuffle=True, random_state=rs % (2 ** 31))
            for tr, te in skf.split(np.zeros(n), y):
                if prep_mode == "infold":
                    prep = FoldPreprocessor(cols).fit(X.iloc[tr])
                    Xtr, Xte = prep.transform(X.iloc[tr], scaled), prep.transform(X.iloc[te], scaled)
                else:
                    Z = Zs if scaled else Zr
                    Xtr, Xte = Z[tr], Z[te]
                P[r, te] = abl_fitpred(m, Xtr, y[tr], Xte, stable_seed(seed, r, m), jobs)
            own.append(fast_auc(y, P[r]))
    return P, own


def abl_auc_rep(P, y, idx):
    return float(np.nanmean([fast_auc(y[idx], P[r, idx]) for r in range(P.shape[0])]))


def ablation_study(df, X, models, folds_emb, args):
    cols = FEATURE_SETS[ALL_NAME]
    y = df[TARGET].to_numpy()
    folds_noemb = make_folds(df, embargo_days=1)           # nothing dropped
    common = np.sort(np.concatenate([f["test"] for f in folds_emb]))
    n_noemb = sum(len(f["test"]) for f in folds_noemb)
    print(f"  common evaluation set = {len(common)} real test IPOs of the embargoed design "
          f"({n_noemb} test IPOs without embargo)")
    preds, own, t0 = {}, {}, time.time()
    for sname, sc in ABL_SCENARIOS.items():
        for m in models:
            if sc["kind"] == "chrono":
                fl = folds_emb if sc["embargo"] else folds_noemb
                P, o = abl_run_chrono(X, y, cols, fl, m, sc["prep"], args.seed, args.jobs)
            else:
                P, o = abl_run_random(X, y, cols, m, sc["prep"], sc["smote"], args.seed, args.jobs, args.abl_repeats)
            preds[(sname, m)], own[(sname, m)] = P, float(np.nanmean(o))
        print(f"  {sname} done ({time.time() - t0:.0f}s)")
    rng = np.random.default_rng(args.seed)
    boot_idx = [rng.integers(0, len(common), len(common)) for _ in range(args.abl_boot)]
    rows = []
    for m in models:
        base = preds[(ABL_BASE, m)]
        a0 = abl_auc_rep(base, y, common)
        for sname in ABL_SCENARIOS:
            P = preds[(sname, m)]
            a = abl_auc_rep(P, y, common)
            d = [abl_auc_rep(P, y, common[bi]) - abl_auc_rep(base, y, common[bi]) for bi in boot_idx]
            mask = ~np.isnan(P[0])
            rows.append({"model": MODEL_LABEL[m], "scenario": sname, "auc_common_real_ipos": a,
                         "auc_own_real_test_set": abl_auc_rep(P, y, np.where(mask)[0]), "n_own_test_set": int(mask.sum()),
                         "delta_vs_S0": a - a0, "delta_ci_low": np.percentile(d, 2.5), "delta_ci_high": np.percentile(d, 97.5),
                         "auc_own_incl_synthetic": (own[(sname, m)] if ABL_SCENARIOS[sname]["smote"] else np.nan),
                         "n_scored_common": len(common)})
    res = pd.DataFrame(rows)
    order = list(ABL_SCENARIOS)
    mat = res.pivot(index="scenario", columns="model", values="auc_common_real_ipos").reindex(order)
    mat["mean over models"] = mat.mean(axis=1)
    dmat = res.pivot(index="scenario", columns="model", values="delta_vs_S0").reindex(order)
    dmat["mean over models"] = dmat.mean(axis=1)
    own_mat = res.pivot(index="scenario", columns="model", values="auc_own_real_test_set").reindex(order)
    own_mat["mean over models"] = own_mat.mean(axis=1)
    smote_own = res[res["auc_own_incl_synthetic"].notna()][["model", "auc_own_incl_synthetic"]]
    readme = pd.DataFrame({"note": [
        "Same data, features (ALL pillars), models and FIXED default hyper-parameters in every scenario.",
        "auc_common_real_ipos: AUC on the same real IPOs for every scenario (the embargoed test IPOs of S0). "
        "Random designs: mean over repeats of out-of-fold AUC.",
        "delta_vs_S0 with 95% paired bootstrap CI (resampling IPOs). Positive = design inflates AUC.",
        "S5: SMOTE (k=5) is applied to the full dataset before the split; 'auc_own_incl_synthetic' is what a paper using "
        "this pipeline would report (synthetic rows in the test folds).",
        "auc_own_real_test_set: AUC on each scenario's own real test IPOs. The embargo only drops test IPOs, so on the "
        "common set S1 equals S0 by construction.",
        "Inflation is specific to this dataset and these default-parameter models."]})
    xlsx = os.path.join(args.outdir, "ablation_results.xlsx")
    with pd.ExcelWriter(xlsx) as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        mat.to_excel(xw, sheet_name="auc_matrix")
        dmat.to_excel(xw, sheet_name="delta_vs_S0_matrix")
        own_mat.to_excel(xw, sheet_name="auc_own_test_set")
        res.to_excel(xw, sheet_name="long_results", index=False)
        smote_own.to_excel(xw, sheet_name="S5_auc_with_synthetic", index=False)
    figs = {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        ax.barh([s.split(" ", 1)[0] + " " + s.split(" ", 1)[1][:38] for s in mat.index][::-1],
                mat["mean over models"].to_numpy()[::-1], color="#3b6fb6")
        ax.axvline(0.5, color="k", ls="--", lw=1)
        ax.set_xlim(0.45, max(0.7, float(mat["mean over models"].max()) + 0.03))
        ax.set_xlabel("AUC on common real test IPOs (mean over models)")
        ax.set_title("Cost of common validation flaws")
        fig.tight_layout()
        p = os.path.join(args.outdir, "ablation_auc.png")
        fig.savefig(p, dpi=200)
        plt.close(fig)
        figs["mean"] = p
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ms = [c_ for c_ in mat.columns if c_ != "mean over models"]
        w = 0.8 / len(ms)
        for j, mm in enumerate(ms):
            ax.bar(np.arange(len(order)) + j * w, mat[mm].to_numpy(), w, label=mm)
        ax.axhline(0.5, color="k", ls="--", lw=1)
        ax.set_xticks(np.arange(len(order)) + w * (len(ms) - 1) / 2)
        ax.set_xticklabels([s.split(" ", 1)[0] for s in order])
        ax.set_ylim(0.45, max(0.7, float(np.nanmax(mat[ms].to_numpy())) + 0.03))
        ax.set_ylabel("AUC on common real test IPOs")
        ax.set_title("Validation design (S0 = ours) by model")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(args.outdir, "ablation_auc_by_model.png")
        fig.savefig(p, dpi=200)
        plt.close(fig)
        figs["by_model"] = p
    except Exception as e:
        print("  ablation chart skipped:", e)
    print(f"Saved ablation -> {xlsx}")
    return {"res": res, "mat": mat, "dmat": dmat, "own_mat": own_mat, "smote_own": smote_own, "figs": figs,
            "n_common": len(common), "n_noemb": n_noemb, "repeats": args.abl_repeats, "n_boot": args.abl_boot}


# --------------------------------------------------------------------------
# WORD REPORT  (python-docx)
# --------------------------------------------------------------------------
def _fmt_val(v, col=""):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}" if abs(v) >= 10000 else str(v)
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "–"
        lc = col.lower()
        if "p_value" in lc or lc.endswith("_p") or lc in ("p", "p-value") or "chi2_p" in lc or "mannwhitney_p" in lc or "wilcoxon_p" in lc:
            return "<0.001" if v < 0.001 else f"{v:.3f}"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 100:
            return f"{v:.1f}"
        if abs(v) < 0.01 and v != 0:
            return f"{v:.4f}"
        return f"{v:.3f}"
    if isinstance(v, pd.Timestamp):
        return f"{v:%Y-%m-%d}"
    return str(v)


def build_word_report(ctx, path):
    try:
        from docx import Document
        from docx.enum.section import WD_ORIENT
        from docx.enum.table import WD_TABLE_ALIGNMENT
        from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        from docx.shared import Cm, Pt, RGBColor
    except ImportError:
        print("  python-docx not installed -> Word report skipped (pip install python-docx)")
        return None

    args, df, folds = ctx["args"], ctx["df"], ctx["folds"]
    pool, models = ctx["pool"], ctx["models"]
    desc, shp, abl = ctx.get("desc"), ctx.get("shap"), ctx.get("abl")
    best = ctx["best_model"]
    blabel = MODEL_LABEL[best]
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    for side in ("left_margin", "right_margin"):
        setattr(sec, side, Cm(2.0))
    sec.top_margin, sec.bottom_margin = Cm(2.0), Cm(2.0)
    TEXT_W = 17.0
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Calibri", Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    st.paragraph_format.space_after = Pt(6)
    for name, size, color in (("Heading 1", 16, "1F3864"), ("Heading 2", 13, "2F5496"), ("Heading 3", 11, "2F5496")):
        h = doc.styles[name]
        h.font.name, h.font.size, h.font.bold = "Calibri", Pt(size), True
        h.font.color.rgb = RGBColor.from_string(color)
        h.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    fig_no, tab_no = [0], [0]

    def P(text="", bold=False, italic=False, size=None, align=None, style=None, space_after=None):
        p = doc.add_paragraph(style=style)
        r = p.add_run(text)
        r.bold, r.italic = bold, italic
        if size:
            r.font.size = Pt(size)
        if align:
            p.alignment = align
        if space_after is not None:
            p.paragraph_format.space_after = Pt(space_after)
        return p

    def LP(lead, text, style=None):
        p = doc.add_paragraph(style=style)
        p.add_run(lead).bold = True
        p.add_run(text)
        return p

    def B(text, lead=None):
        p = doc.add_paragraph(style="List Bullet")
        if lead:
            p.add_run(lead).bold = True
        p.add_run(text)
        p.paragraph_format.space_after = Pt(2)
        return p

    def H(text, lvl=1):
        return doc.add_heading(text, lvl)

    def shade(cell, fill):
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), fill)
        tcPr.append(shd)

    def T(frame, title=None, widths=None, size=8, index=False, note=None, max_rows=None):
        frame = frame.copy()
        if index:
            frame = frame.reset_index()
        if max_rows:
            frame = frame.head(max_rows)
        if title:
            tab_no[0] += 1
            c = P(f"Table {tab_no[0]}. {title}", bold=True, size=9, space_after=2)
            c.paragraph_format.keep_with_next = True
        cols = list(frame.columns)
        t = doc.add_table(rows=1, cols=len(cols))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        if widths is None:
            first = None
            if len(cols) > 3 and frame[cols[0]].map(lambda v: isinstance(v, str)).all():
                first = float(min(7.0, max(3.2, 0.16 * frame[cols[0]].map(len).max())))
            if first and len(cols) > 1:
                rest = (TEXT_W - first) / (len(cols) - 1)
                widths = [first] + [rest] * (len(cols) - 1)
            else:
                widths = [TEXT_W / len(cols)] * len(cols)
        else:
            tot = sum(widths)
            widths = [w * TEXT_W / tot for w in widths]
        for i, w_ in enumerate(widths):
            t.columns[i].width = Cm(w_)          # table grid (LibreOffice / Google Docs) as well as cell widths (Word)
        for i, c_ in enumerate(cols):
            cell = t.rows[0].cells[i]
            cell.width = Cm(widths[i])
            cell.text = ""
            r = cell.paragraphs[0].add_run(str(c_).replace("_", " "))
            r.bold, r.font.size = True, Pt(size)
            r.font.color.rgb = RGBColor(255, 255, 255)
            cell.paragraphs[0].paragraph_format.space_after = Pt(0)
            shade(cell, "2F5496")
        trPr = t.rows[0]._tr.get_or_add_trPr()
        th = OxmlElement("w:tblHeader")
        th.set(qn("w:val"), "true")
        trPr.append(th)
        for ri, (_, row) in enumerate(frame.iterrows()):
            cells = t.add_row().cells
            for i, c_ in enumerate(cols):
                cells[i].width = Cm(widths[i])
                cells[i].text = ""
                pr = cells[i].paragraphs[0]
                pr.paragraph_format.space_after = Pt(0)
                v = row[c_]
                rr = pr.add_run(_fmt_val(v, str(c_)))
                rr.font.size = Pt(size)
                if i > 0 and isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                    pr.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                if ri % 2 == 1:
                    shade(cells[i], "EEF2F9")
        if note:
            P(note, italic=True, size=8)
        else:
            doc.add_paragraph().paragraph_format.space_after = Pt(2)
        return t

    def FIG(path_, caption, width=16.0):
        if not path_ or not os.path.exists(path_):
            return
        fig_no[0] += 1
        doc.add_picture(path_, width=Cm(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.paragraphs[-1].paragraph_format.keep_with_next = True
        c = P(f"Figure {fig_no[0]}. {caption}", italic=True, size=8.5, align=WD_ALIGN_PARAGRAPH.CENTER)

    def PB():
        doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # footer with page numbers
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = fp.add_run("IPO underpricing – modelling report   |   page ")
    r.font.size = Pt(8)
    for typ, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        rr = fp.add_run()
        rr.font.size = Pt(8)
        if typ:
            fc = OxmlElement("w:fldChar")
            fc.set(qn("w:fldCharType"), typ)
            rr._r.append(fc)
        else:
            it = OxmlElement("w:instrText")
            it.set(qn("xml:space"), "preserve")
            it.text = txt
            rr._r.append(it)

    # ------------------------------------------------------------------ numbers used in the text
    ALL_ = pool[pool["set"] == ALL_NAME]
    brow = ALL_[ALL_["model"] == best].iloc[0]
    ctrl = pool[(pool["set"] == CONTROL_NAME) & (pool["model"] == best)]
    n_ipo, y_all = len(df), df[TARGET]
    n_test = int(brow["n_test"])
    pillar_auc = pool[(pool["model"] == best) & (pool["set"].isin(list(PILLARS)))].sort_values("auc", ascending=False)

    # ------------------------------------------------------------------ title
    for _ in range(5):
        P("")
    P("Predicting U.S. IPO Underpricing", bold=True, size=26, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    P("Leak-free expanding-window machine-learning study with SHAP explanations,", size=13, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=0)
    P("descriptive analysis and a validation-flaw ablation", size=13, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=18)
    P(f"Sample: {n_ipo} IPOs, {df[DATE_COL].min():%d %b %Y} – {df[DATE_COL].max():%d %b %Y}", align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    P(f"Report generated {pd.Timestamp.now():%d %B %Y} by predict_underpricing.py", align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    mode = "QUICK smoke-test settings (small search, few resamples)" if args.quick else "full settings"
    P(f"Run configuration: {mode}; seed {args.seed}", italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)
    PB()
    H("Contents", 1)
    toc = ["1  Executive summary", "2  Data and descriptive analysis (training rows of one fold only)",
           "3  Methodology: the full modelling process", "4  Predictive results", "5  Statistical confirmation",
           "6  Data-leakage diagnostics", "7  Explainability: SHAP values in probability units",
           "8  Methodology comparison: our design versus common validation flaws",
           "9  Limitations and interpretation", "Appendix A  Tuned hyper-parameters   |   Appendix B  Output files"]
    for t_ in toc:
        P(t_, space_after=1)
    PB()

    # ------------------------------------------------------------------ 1 executive summary
    H("1  Executive summary", 1)
    P(f"This report documents a leak-free evaluation of how well pre-IPO information predicts whether a U.S. "
      f"IPO is underpriced (target = 1). The sample has {n_ipo} IPOs, of which {100 * y_all.mean():.1f}% are underpriced. Models are "
      f"trained on an expanding chronological window and tested on each calendar year from {folds[0]['year']} to {folds[-1]['year']}, "
      f"giving {n_test} pooled out-of-sample predictions per model. All numbers below come from that run.")
    B(f" {blabel} is the best model on all eight pillars combined, with a pooled out-of-sample AUC of {brow['auc']:.3f} "
      f"(95% bootstrap CI {brow['auc_ci_low']:.3f}–{brow['auc_ci_high']:.3f}; label-permutation p = {_fmt_val(brow['perm_p_value_auc'], 'p_value')}); "
      f"accuracy {brow['accuracy']:.3f} and balanced accuracy {brow['balanced_accuracy']:.3f} at a 0.5 threshold. An AUC of 0.5 is chance.",
      "Predictive power.")
    if len(ctrl):
        B(f" a model that only knows the calendar date (ipo_year) reaches AUC {ctrl.iloc[0]['auc']:.3f}, compared with "
          f"{brow['auc']:.3f} for the same model using all pillars; the gap shows how much of the result goes beyond the passage of time.",
          "Time-only control:")
    if len(pillar_auc):
        B(f" for {blabel} the most informative single pillar is “{pillar_auc.iloc[0]['set']}” (AUC {pillar_auc.iloc[0]['auc']:.3f}) "
          f"and the least informative is “{pillar_auc.iloc[-1]['set']}” (AUC {pillar_auc.iloc[-1]['auc']:.3f}).", "Pillars:")
    if shp and shp.get("top_feats"):
        tf = shp["top_feats"]
        B(" the three features with the largest mean absolute SHAP value (probability units) are "
          + ", ".join(f"{a} ({b:.2f} pp)" for a, b in tf[:3]) + f" for {blabel}. SHAP describes what the model relies on, not causation.",
          "Explainability:")
    ig = ctx["integrity_ok"]
    pm = ctx["perm"]
    B(f" temporal-integrity checks {'all passed' if ig else 'FAILED – inspect leak_integrity'}; refitting on shuffled training labels gave AUC "
      f"{pm['perm_auc_mean'].mean():.3f} on average across models (honest pipelines return ≈0.5).", "Leakage:")
    if abl:
        m_ = abl["dmat"]["mean over models"]
        B(f" holding data, features and models fixed, random 5-fold CV changes the mean AUC by {m_.iloc[3]:+.3f}, random CV with global preprocessing "
          f"{m_.iloc[4]:+.3f}, and random CV with global preprocessing plus SMOTE before the split {m_.iloc[5]:+.3f} relative to our design (S0), "
          "on identical real IPOs. "
          f"See Section 8 for confidence intervals.", "Design comparison:")
    if desc:
        ty = desc["tables"]["by_year"]
        ty = ty[ty["n"] >= 10]
        if len(ty):
            hi, lo = ty.loc[ty["underpricing_rate_pct"].idxmax()], ty.loc[ty["underpricing_rate_pct"].idxmin()]
            B(f" in the training rows of fold {desc['fold_year']} the underpricing rate ranges from {lo['underpricing_rate_pct']:.0f}% "
              f"({lo['group']}) to {hi['underpricing_rate_pct']:.0f}% ({hi['group']}) across years with at least 10 IPOs.", "Descriptive pattern:")
    P("Take-away: a result is only as credible as the design that produced it. The AUC reported here is the one obtained under the strictest design; "
      "Section 8 quantifies how much weaker designs would have overstated it.", italic=True)
    PB()

    # ------------------------------------------------------------------ 2 descriptive
    H("2  Data and descriptive analysis (training rows of one fold only)", 1)
    if desc:
        T_, F_ = desc["tables"], desc["figs"]
        which = "last" if desc["which"] == "last" else "first"
        P(f"To prevent data leakage, every descriptive statistic in this section is computed on the TRAINING rows of the {which} fold "
          f"(test year {desc['fold_year']}) only: {desc['n_train']} IPOs dated {desc['date_min']:%d %b %Y} to {desc['date_max']:%d %b %Y}. "
          f"No IPO from the test years is used, so nothing learnt here can influence model evaluation. "
          f"The overall underpricing rate in these rows is {desc['overall_rate']:.1f}%. "
          "Continuous features are shown after the fold's own training-fitted imputation and winsorization (the values the models actually saw); "
          "categorical groupings come from the raw workbook. The workbook records only the binary outcome (no first-day return), so the analysis "
          "is of the frequency, not the magnitude, of underpricing. Associations are descriptive, uncorrected for multiple testing and not causal.")
        H("2.1  Underpricing by IPO year", 2)
        FIG(F_.get("by_year"), "Underpricing rate and number of IPOs by year (training rows).")
        T(T_["by_year"][["group", "n", "n_underpriced", "underpricing_rate_pct", "ci_low_pct", "ci_high_pct"]].rename(columns={"group": "year"}),
          "Underpricing by year (training rows)")
        H("2.2  Underpricing by sector and industry", 2)
        FIG(F_.get("by_sector"), "Underpricing rate by sector group (sector mapping defined in Section 3.2).")
        T(T_["by_sector"][["group", "n", "n_underpriced", "underpricing_rate_pct", "ci_low_pct", "ci_high_pct"]].rename(columns={"group": "sector"}),
          "Underpricing by sector")
        FIG(F_.get("by_industry"), "Underpricing rate for the 15 largest raw industries.", 15.5)
        T(T_["by_industry"][["group", "n", "n_underpriced", "underpricing_rate_pct", "ci_low_pct", "ci_high_pct"]].rename(columns={"group": "industry"}),
          "Underpricing by industry (15 largest)")
        H("2.3  Underpricing by underwriter quality, exchange and governance", 2)
        FIG(F_.get("by_underwriter"), "Underpricing by underwriter reputation tier.", 12)
        T(T_["by_underwriter"][["group", "n", "n_underpriced", "underpricing_rate_pct", "ci_low_pct", "ci_high_pct"]].rename(columns={"group": "underwriter tier"}),
          "Underpricing by underwriter reputation")
        FIG(F_.get("by_exchange"), "Underpricing by listing exchange.", 11)
        FIG(F_.get("by_binary_flags"), "Underpricing by institutional backing, Big-4 auditor and dual-class structure.")
        T(T_["binary_summary"], "Binary certification / governance flags")
        T(T_["independence_tests"], "Chi-square tests of independence between each grouping and the outcome",
          note="Small p-values indicate that the underpricing rate differs across groups in the training sample; groups with small counts are unreliable.")
        H("2.4  All continuous features: summary statistics", 2)
        cs = T_["continuous_summary"]
        T(cs[["feature", "n_missing_raw", "mean", "sd", "min", "p25", "median", "p75", "max", "skew"]],
          "Summary statistics of continuous features (training rows, after train-fitted imputation and winsorization)", size=7)
        T(cs[["feature", "pillar", "mean_underpriced", "mean_not_underpriced", "median_underpriced", "median_not_underpriced",
              "mannwhitney_p", "univariate_auc", "spearman_with_target"]],
          "Continuous features by outcome: group means / medians, Mann–Whitney test and univariate AUC", size=7,
          widths=[3.6, 3.2, 1.5, 1.5, 1.5, 1.5, 1.4, 1.4, 1.4],
          note="Univariate AUC > 0.5 means higher values go with underpricing; < 0.5 the opposite.")
        FIG(F_.get("univariate_auc"), "Univariate AUC of each continuous feature.", 13)
        H("2.5  Underpricing across the range of every feature", 2)
        P("Each feature is split into quintiles using training-row cut-points. Bars show the underpricing rate in each quintile (dashed = overall rate).")
        for i, q in enumerate(F_.get("quintiles", []), 1):
            FIG(q, f"Underpricing rate by feature quintile ({i} of {len(F_['quintiles'])}).", 16.5)
        H("2.6  Distributions by outcome", 2)
        for i, q in enumerate(F_.get("distributions", []), 1):
            FIG(q, f"Feature distributions for underpriced vs not underpriced IPOs ({i} of {len(F_['distributions'])}).", 16.5)
        H("2.7  Features over time and feature correlations", 2)
        FIG(F_.get("yearly_features"), "Yearly mean of each continuous feature, standardised across years.")
        FIG(F_.get("correlation"), "Spearman correlation matrix of continuous features.", 14.5)
        hp = T_["high_corr_pairs"]
        if len(hp):
            T(hp, "Feature pairs with |Spearman rho| ≥ 0.5", note="Strongly correlated features share credit in SHAP and make unpenalized-logit coefficients less stable (see VIF in Section 5).")
        ym = T_["yearly_feature_means"]
        keep = [c_ for c_ in ["year", "n_ipos", "underpricing_rate_pct", "vix", "nasdaq_composite", "ipo_count_30d", "prime_rate",
                              "yield_curve_spread", "us_gdp", "us_inflation_pct"] if c_ in ym.columns]
        T(ym[keep], "Yearly means of market and macro features with the underpricing rate (training rows)", size=7.5)
    else:
        P("Descriptive analysis was not run (--no-desc).")
    PB()

    # ------------------------------------------------------------------ 3 methodology
    H("3  Methodology: the full modelling process", 1)
    P("This section explains every step from raw workbook to final statistic, in the order the code executes it. Each design choice is "
      "made to answer one question: could the number reported have been influenced by information that would not have been available "
      "when the IPO was priced?")
    H("3.1  Research question and target", 2)
    P(f"The task is binary classification: predict whether an IPO is underpriced (underpriced = 1, otherwise 0) from information available "
      f"at or before pricing. The workbook contains {n_ipo} U.S. IPOs from {df[DATE_COL].min():%B %Y} to {df[DATE_COL].max():%B %Y}; "
      f"{int(y_all.sum())} ({100 * y_all.mean():.1f}%) are underpriced. Because the classes are imbalanced, performance is summarised mainly by "
      "the area under the ROC curve (AUC), which does not depend on a classification threshold, and secondarily by threshold-based metrics.")
    H("3.2  Features: eight pillars", 2)
    P("Features are grouped into eight conceptual pillars. Each pillar is modelled separately (to see which information matters) and all are "
      "combined in the ‘ALL pillars’ set. A control set with only the IPO date checks how much time alone predicts.")
    rows = [{"pillar": p_, "n_features": len(cs_), "features": ", ".join(cs_)} for p_, cs_ in PILLARS.items()]
    T(pd.DataFrame(rows), "Pillars and their features", widths=[3.5, 1.3, 12.2], size=7.5)
    B(" the exchange (reference: Nasdaq), underwriter reputation (reference: Tier 1; missing → ‘Unknown’) and nine sector groups "
      "(reference: Biotech & Pharma) are one-hot encoded with one level dropped to avoid the dummy-variable trap. Sector groups are built by "
      "ordered regular-expression rules applied to the raw industry label; anything unmatched is ‘Other’.", "Encoding:")
    B(" log_proceeds is removed because it is almost exactly log_offer_price + log_shares_offered (VIF ≈ 85), which made coefficients of the "
      "unpenalized logit unstable. Columns that are constant (e.g. an empty sector bucket) are dropped automatically.", "Collinearity:")
    B(" ipo_year (fractional year) is used only in the control set, never in the eight pillars.", "Time control:")
    H("3.3  Validation design: expanding window with an embargo", 2)
    P(f"Fold Y trains on every IPO dated before 1 January of year Y and tests on the IPOs of year Y. Test years run from {FIRST_TEST_YEAR} to "
      f"{folds[-1]['year']}. Earlier test and dropped IPOs re-enter later training sets, so the training window expands as time passes. "
      f"IPOs in the first {EMBARGO_DAYS - 1} days of a test year (on or before 29 January) are dropped from that fold’s test set (embargo) "
      "because their market conditions and information overlap with the end of the training window. The chronological order guarantees the model "
      "never sees the future; the embargo further reduces serial-dependence spillover. Pooled metrics concatenate all out-of-sample predictions.")
    ftab = ctx["ft"][["fold", "n_train", "n_test", "n_dropped", "train_last", "test_first"]]
    T(ftab, "Fold table (n_dropped = embargoed January IPOs)", size=7.5, widths=[1.2, 1.5, 1.5, 1.6, 5.6, 5.6])
    H("3.4  Preprocessing fitted on training rows only", 2)
    if CLEAN_IN_FOLD:
        P("For each fold, the following steps are fitted on that fold’s training rows and then applied unchanged to its test rows:")
        B(" missing continuous values are replaced by the training median; missing binary/dummy values by the training mode.", "Imputation:")
        B(f" bounds are Tukey fences (Q1 − {WINSOR_K}·IQR, Q3 + {WINSOR_K}·IQR) computed on the training rows, applied, and recomputed repeatedly "
          "until no value lies outside (iterative winsorization); if the IQR is zero the 1st/99th percentiles are used. Test values are clipped to the "
          "training bounds.", "Winsorization:")
        sc_ = [MODEL_LABEL[m_] for m_ in models if m_ in SCALED_MODELS] + ["Logit (unpenalized)"]
        un_ = [MODEL_LABEL[m_] for m_ in models if m_ not in SCALED_MODELS]
        B(f" median/IQR (robust) scaling for the models that need it ({', '.join(sc_)}); "
          + (f"{', '.join(un_)} use unscaled values." if un_ else "no model uses unscaled inputs."), "Scaling:")
    else:
        P("The run used --as-given: imputation and winsorization were skipped because the workbook is complete and clean. "
          "Median/IQR scaling (scale-sensitive models) is still fitted on training rows only.")
    P("Consequently no future IPO can influence a fill value, a clipping bound or a scaling constant. Section 8 shows what happens when this rule is broken.")
    H("3.5  Models and class imbalance", 2)
    mdl_rows = [{"model": MODEL_LABEL[m_], "role": "machine-learning model", "imbalance handling":
                 {"LR": "class_weight = balanced (L2 penalty; C tuned)", "LGBM": "class_weight = balanced",
                  "RF": "class_weight = balanced_subsample", "XGB": "scale_pos_weight = n_negative / n_positive",
                  "GB": "balanced sample weights", "SVM": "class_weight = balanced; Platt probabilities",
                  "NN": "random oversampling inside the training rows"}.get(m_, "")} for m_ in models]
    mdl_rows.append({"model": "Logit (unpenalized)", "role": "confirmatory classical benchmark (own IRLS code, HC3 robust SEs)", "imbalance handling": "none (inference model)"})
    mdl_rows.append({"model": "Baseline: train prior", "role": "predicts the training underpricing rate for every IPO (AUC 0.5)", "imbalance handling": "–"})
    T(pd.DataFrame(mdl_rows), "Models", widths=[3.2, 7.5, 6.3], size=8)
    P("Class imbalance is handled with class weights rather than synthetic oversampling, so no artificial observation is ever created. "
      "Because of the weighting, predicted probabilities are centred near 0.5, not near the raw underpricing rate; they rank IPOs well but are "
      "not calibrated frequencies (see Section 9).")
    H("3.6  Hyper-parameter tuning without peeking", 2)
    P(f"Within each fold’s training window, hyper-parameters are chosen on a purged chronological validation block: the last "
      f"{int(VAL_FRAC * 100)}% of training IPOs (at least {VAL_MIN}) form the validation set, and training rows within {VAL_PURGE_DAYS} days before it are removed "
      f"(purge). Preprocessing is re-fitted on the remaining sub-training rows. "
      + ("The logistic penalty C is chosen from a grid; " if "LR" in models else "")
      + f"{', '.join(MODEL_LABEL[m_] for m_ in models if m_ != 'LR')} use a random search of {args.n_iter} draws from their hyper-parameter grids. "
      f"The criterion is validation AUC. The winning configuration is then refitted on the "
      "full training window and used once on the test year. The test year is never used for tuning.")
    H("3.7  Evaluation metrics and statistical tests", 2)
    B(f" pooled AUC over all test years, with a {args.n_boot}-resample bootstrap 95% CI and a one-sided label-permutation p-value "
      f"({args.n_perm} permutations, labels shuffled within each test year).", "Discrimination:")
    B(" accuracy, balanced accuracy, F1 at the 0.5 threshold, Brier score and log-loss (pooled and per fold).", "Threshold and probability metrics:")
    B(" DeLong tests for paired AUC differences (model vs model on ALL; ALL vs each pillar and the control), McNemar tests on 0.5-threshold "
      "predictions, and Wilcoxon signed-rank tests on fold AUCs.", "Comparisons:")
    B(" an unpenalized logit on the full sample gives odds ratios with HC3 robust standard errors, a likelihood-ratio test per pillar "
      "set, drop-one-pillar tests, McFadden R², AIC/BIC and VIF. These are descriptive (full sample), not part of the out-of-sample evidence.", "Classical inference:")
    H("3.8  Data-leakage safeguards", 2)
    B(" for every fold: training dates precede test dates, the embargo is respected, no row or dropped IPO appears in test, and the target is not a feature.", "Temporal integrity:")
    B(" training labels are shuffled, models are refitted with the tuned parameters and scored on the real test labels; an honest pipeline returns AUC ≈ 0.5.", "Label-permutation refit:")
    B(" every feature’s single-variable AUC is screened; values far from 0.5 would signal a leaked variable.", "Univariate screen:")
    B(" random 5-fold versus chronological CV with default parameters, reported as a diagnostic only.", "Random vs chronological gap:")
    H("3.9  Explainability with SHAP in probability units", 2)
    eng = "; ".join(f"{MODEL_LABEL[k]}: {v}" for k, v in (shp or {}).get("engines", {}).items()) or "not run"
    P("SHAP (Shapley additive explanations) splits each prediction into feature contributions. Here they are expressed directly in probability units "
      "(percentage points, pp): for every IPO, base value + the sum of SHAP values = the model’s predicted probability. Explanations are strictly "
      "out-of-sample: each fold’s fitted ALL-pillars model explains that fold’s test IPOs, using a background sample of "
      f"{args.shap_bg} training rows as the reference distribution (interventional SHAP). Engines used – {eng}. "
      "Global importance is the mean absolute SHAP value; direction is the Spearman correlation between a feature’s value and its SHAP value; "
      "pillar importance sums SHAP values of the pillar’s features per IPO before taking absolute values, so offsetting features inside a pillar net out.")
    H("3.10  Descriptive analysis and the validation-flaw ablation", 2)
    P("Descriptive statistics (Section 2) use only the training rows of one fold. The ablation (Section 8) re-runs the same data, features and models with "
      "fixed default hyper-parameters under six designs, from ours (S0) to a typical flawed pipeline (S5), and scores every design on one common set of real IPOs "
      "so that differences are purely due to the design.")
    PB()

    # ------------------------------------------------------------------ 4 results
    H("4  Predictive results", 1)
    wide = ctx["wide"]
    P("Table below gives pooled out-of-sample AUC for every model on every feature set; Figure shows the same with 95% bootstrap intervals.")
    T(wide.round(3), "Pooled out-of-sample AUC by feature set and model", index=True, size=7.5)
    FIG(ctx.get("auc_png"), "Pooled AUC by pillar and model (95% bootstrap CI; dashed line = chance).")
    cols_ = [c_ for c_ in ["model_name", "auc", "auc_ci_low", "auc_ci_high", "perm_p_value_auc", "accuracy", "balanced_accuracy", "f1",
                           "brier", "log_loss", "mean_fold_auc", "std_fold_auc"] if c_ in ALL_.columns]
    T(ALL_[cols_].rename(columns={"model_name": "model"}), "All pillars combined: pooled metrics", size=7.5)
    fw = ctx["fold_wide"].copy()
    fw.columns = [MODEL_LABEL.get(c_, c_) for c_ in fw.columns]
    T(fw.round(3), "Out-of-sample AUC by test year (ALL pillars)", index=True, size=7.5)
    P("Fold-level AUCs are noisy because individual test years contain few IPOs (see the fold table); the pooled estimate and its interval are the main evidence.")

    # ------------------------------------------------------------------ 5 stats
    H("5  Statistical confirmation", 1)
    dm = ctx["d_models"]
    T(dm.rename(columns={"set": "feature set"}).drop(columns=["feature set"]), "DeLong tests: model vs model (ALL pillars)", size=7.5)
    dp = ctx["d_pillar"]
    dpb = dp[dp["model"] == blabel]
    T(dpb.drop(columns=["model"]), f"DeLong tests: ALL pillars vs each pillar / control ({blabel})", size=7.5)
    T(ctx["mcn"].drop(columns=["set"]), "McNemar tests on 0.5-threshold predictions", size=7.5)
    T(ctx["wil"].drop(columns=["set"]), "Wilcoxon signed-rank tests on fold AUCs", size=7.5)
    H("5.1  Classical logit inference (full sample, descriptive)", 2)
    T(ctx["set_stats"], "Unpenalized logit fit by feature set", size=7)
    T(ctx["drop_lr"], "Drop-one-pillar likelihood-ratio tests (ALL pillars model)", size=7.5)
    co = ctx["coef"]
    co = co[co["feature"] != "const"].assign(abs_z=lambda d_: d_["z"].abs()).sort_values("abs_z", ascending=False).drop(columns="abs_z")
    T(co[["feature", "odds_ratio", "OR_ci_low", "OR_ci_high", "z", "p_value"]], "Odds ratios (HC3 robust; continuous features median/IQR scaled), ordered by |z|", size=7.5, max_rows=20)
    vf = ctx["vif"]
    if len(vf):
        T(vf.head(10), "Largest variance-inflation factors", size=7.5)
    PB()

    # ------------------------------------------------------------------ 6 leakage
    H("6  Data-leakage diagnostics", 1)
    ig_ = ctx["integrity"]
    P(f"Temporal-integrity result: {'PASS' if ctx['integrity_ok'] else 'FAIL'} (training precedes embargo precedes test in every fold; no row overlap).")
    T(ig_, "Temporal integrity by fold", size=6.5)
    T(pm, "Label-permutation refit (shuffled training labels, real test labels)", size=7.5,
      note="Honest pipelines return AUC ≈ 0.5 with shuffled labels; a high shuffled AUC would indicate leakage.")
    T(ctx["uni"].head(10), "Univariate AUC screen of every feature (top 10 by distance from 0.5)", size=7.5)
    T(ctx["rvc"], "Random 5-fold versus chronological CV (default parameters, diagnostic)", size=8)
    PB()

    # ------------------------------------------------------------------ 7 SHAP
    H("7  Explainability: SHAP values in probability units", 1)
    if shp:
        P("All values below are in percentage points (pp) of predicted probability and are out-of-sample. Figures use the ALL-pillars models. "
          "Mean |SHAP| measures how much a feature moves predictions on average; the beeswarm shows each IPO (red = high feature value, blue = low). "
          "SHAP reflects what the fitted models use, not causal effects, and correlated features share credit.")
        glob_, pill_ = shp["glob"], shp["pill"]
        for m_, (p1, p2) in shp["imgs"].items():
            lab = MODEL_LABEL[m_]
            H(f"7.{list(shp['imgs']).index(m_) + 1}  {lab}", 2)
            tp = glob_[glob_["model"] == lab].sort_values("mean_abs_shap_pp", ascending=False).head(12)
            T(tp[["feature", "pillar", "mean_abs_shap_pp", "share_of_total_pct", "mean_signed_shap_pp", "direction_spearman(value,shap)"]]
              .rename(columns={"direction_spearman(value,shap)": "direction (Spearman)"}), f"{lab}: top features by mean |SHAP| (pp)", size=7.5,
              widths=[4.5, 4.2, 1.8, 1.8, 1.8, 2.0])
            FIG(p1, f"{lab}: mean |SHAP| of the top features.", 12.5)
            FIG(p2, f"{lab}: SHAP summary (beeswarm).", 14)
        H(f"7.{len(shp['imgs']) + 1}  Pillar contributions", 2)
        FIG(shp.get("pillars_png"), "Mean absolute pillar-level SHAP.", 15)
        T(pill_.pivot(index="pillar", columns="model", values="mean_abs_shap_pp").reindex(list(PILLARS)).round(3),
          "Mean |pillar SHAP| (pp)", index=True, size=8)
        if shp.get("dep_png"):
            H(f"7.{len(shp['imgs']) + 2}  How the best model’s predictions respond to its top features", 2)
            FIG(shp["dep_png"], f"{blabel}: SHAP dependence of the top features (value on x-axis, effect on probability in pp on y-axis).", 16.5)
        if shp.get("yearly_png"):
            H(f"7.{len(shp['imgs']) + 3}  Stability of importance across test years", 2)
            FIG(shp["yearly_png"], f"{blabel}: mean |SHAP| (pp) of the top features by test year.", 14.5)
        H(f"7.{len(shp['imgs']) + 4}  Additivity check", 2)
        T(shp["add"], "SHAP additivity: base value + Σ SHAP − predicted probability (should be ≈ 0)", size=7.5)
        P("The full per-IPO SHAP values, per-IPO top drivers and yearly importance are in shap_values_probability.xlsx.")
    else:
        P("SHAP report was not run (--no-shap).")
    PB()

    # ------------------------------------------------------------------ 8 ablation
    H("8  Methodology comparison: our design versus common validation flaws", 1)
    if abl:
        P("Many published IPO and finance machine-learning results rely on designs that let future information reach the model. The ablation below holds "
          "the data, the ALL-pillars features, the models and FIXED default hyper-parameters constant (so tuning cannot explain any difference) and changes only "
          "the validation design. Every design is scored on the same set of real IPOs (the embargoed test IPOs of S0), so differences are pure design effects.")
        sd = pd.DataFrame({"scenario": list(ABL_SCENARIOS), "what changes relative to S0": [
            "Baseline: chronological expanding window, 30-day embargo, preprocessing fitted in-fold.",
            "January IPOs of each test year are NOT dropped (embargo removed).",
            "Imputation, winsorization and scaling are fitted on ALL IPOs, including the test IPOs.",
            "Stratified random 5-fold CV instead of chronological folds (preprocessing in-fold); future IPOs train the model.",
            "Random CV and global preprocessing combined.",
            "S4 plus SMOTE applied to the whole dataset BEFORE the split, so synthetic neighbours of test IPOs sit in training."]})
        T(sd, "The six validation designs", widths=[6, 11], size=8)
        P(f"Random-split designs average {abl['repeats']} random seeds; {abl['n_boot']} bootstrap resamples of IPOs give 95% intervals for each difference versus S0.")
        T(abl["mat"].round(3), "AUC on the common set of real test IPOs", index=True, size=7.5)
        T(abl["dmat"].round(3), "Inflation versus our design S0 (AUC points; positive = the flawed design looks better)", index=True, size=7.5)
        FIG(abl["figs"].get("mean"), "Mean AUC over models on the common set, by validation design.", 13)
        FIG(abl["figs"].get("by_model"), "AUC by validation design and model.", 14)
        res = abl["res"]
        ci = res[res["scenario"] != ABL_BASE][["model", "scenario", "delta_vs_S0", "delta_ci_low", "delta_ci_high"]]
        T(ci, "Inflation versus S0 with 95% paired bootstrap interval", size=7, widths=[3.2, 7.5, 2, 2, 2])
        T(abl["own_mat"].round(3), "AUC on each design’s own real test set (S1 also scores the embargoed January IPOs)", index=True, size=7.5)
        if len(abl["smote_own"]):
            T(abl["smote_own"], "S5: AUC a SMOTE-before-split paper would report (synthetic rows included in test folds)", size=8)
        H("8.1  Reading the comparison", 2)
        m_ = abl["dmat"]["mean over models"]
        B(f" with identical models and parameters, moving from our chronological design to random 5-fold changes the mean AUC by {m_.iloc[3]:+.3f}. "
          "Random folds let the model learn from IPOs that happened after the ones it is tested on, which is not possible in practice.", "Random CV:")
        B(f" fitting preprocessing on all IPOs changes the mean AUC by {m_.iloc[2]:+.3f} (chronological) and {m_.iloc[4]:+.3f} versus S0 when combined with random CV.", "Global preprocessing:")
        B(f" adding SMOTE before the split moves the mean AUC on real IPOs by {m_.iloc[5]:+.3f}; the figure a paper would report, which scores synthetic rows, is shown in the S5 table above.", "SMOTE before split:")
        B(" the embargo removes test IPOs only, so on the common set S1 equals S0 by construction; its effect is visible only in the ‘own test set’ table.", "Embargo:")
        B(" intervals that include zero mean the data cannot separate that flaw’s effect from noise; the point estimate is then not evidence of inflation. "
          "The magnitudes are specific to this dataset and default-parameter models.", "Caution:")
        P(f"Why our design is preferable: it matches deployment (train on the past, predict the future), isolates preprocessing and tuning from test data, "
          f"avoids synthetic observations, and is validated by negative controls (label permutation ≈ 0.5, time-only control, integrity checks).")
    else:
        P("Ablation was not run (--no-ablation).")
    PB()

    # ------------------------------------------------------------------ 9 limitations
    H("9  Limitations and interpretation", 1)
    for lead, txt in (
            ("Sample size and power.", " Test years contain few IPOs, so fold-level AUCs are noisy and the final test year may be a partial year. Rely on pooled estimates and their intervals."),
            ("Binary target.", " The workbook records whether an IPO was underpriced, not the size of first-day return; magnitude and economic value are not assessed."),
            ("Probabilities are not calibrated frequencies.", " Class weights shift predicted probabilities towards 0.5. Use them for ranking, or recalibrate on a validation block before using absolute probabilities."),
            ("SHAP is not causal.", " It explains the fitted models, shares credit between correlated features, and depends on the background sample. Explanations are only as stable as the models."),
            ("Multiple comparisons.", " Many models, feature sets and tests are reported; individual p-values are not adjusted."),
            ("Regime change.", " IPO markets shift (for example 2020–2021). An expanding window averages across regimes; a rolling window or regime indicators could be examined."),
            ("Ablation scope.", " Inflation estimates apply to this dataset and fixed default parameters; other data may differ in magnitude."),
            ("Reproducibility.", f" All random elements use fixed seeds (seed = {args.seed}); results can differ slightly across library versions.")):
        B(txt, lead)
    H("Appendix A  Tuned hyper-parameters", 1)
    bp = ctx["bp"]
    bp_all = bp[(bp["set"] == ALL_NAME) & (bp["params"] != "{}")][["fold", "model", "params"]]
    T(bp_all, "Selected hyper-parameters per fold (ALL pillars)", widths=[1.2, 3.3, 12.5], size=6.5)
    H("Appendix B  Output files", 1)
    files = [("underpricing_results.xlsx", "All metrics, tests, leakage checks, tuned parameters and out-of-sample predictions"),
             ("preprocessed_data_by_fold.xlsx", "Imputed/winsorized data each fold used, with fill values and bounds"),
             ("shap_values_probability.xlsx / shap_report.html", "SHAP values in probability units (global, pillar, yearly, per IPO)"),
             ("descriptive_statistics_train_foldYYYY.xlsx", "Descriptive tables from the chosen fold’s training rows"),
             ("ablation_results.xlsx", "Validation-flaw ablation tables"),
             ("IPO_underpricing_report.docx", "This report")]
    T(pd.DataFrame(files, columns=["file", "content"]), None, widths=[6, 11], size=8)
    doc.save(path)
    print(f"Saved Word report -> {path}")
    return path


def make_plot(pool, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    pool = pool[pool["model"] != "PRIOR"]
    sets = [s for s in list(FEATURE_SETS) + list(CONTROL_SETS) if s in set(pool["set"])]
    models = [m for m in MODEL_LABEL if m in set(pool["model"])]
    fig, ax = plt.subplots(figsize=(13, 6))
    w = 0.8 / max(len(models), 1)
    for j, m in enumerate(models):
        sub = pool[pool["model"] == m].set_index("set").reindex(sets)
        x = np.arange(len(sets)) + j * w
        err = [sub["auc"] - sub["auc_ci_low"], sub["auc_ci_high"] - sub["auc"]]
        ax.bar(x, sub["auc"], w, yerr=err, capsize=2, label=MODEL_LABEL[m])
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xticks(np.arange(len(sets)) + w * (len(models) - 1) / 2)
    ax.set_xticklabels([s.replace(" ", "\n", 1) for s in sets], fontsize=7)
    ax.set_ylabel("Pooled out-of-sample AUC (95% bootstrap CI)")
    ax.set_title("Underpricing prediction: pooled AUC by pillar and model (expanding window)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="New Dataset Test.xlsx")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--models", default="RF,GB,SVM,NN",
                    help="comma list from LR,LGBM,RF,XGB,GB,SVM,NN (default: Random Forest, Gradient Boosting, SVM, Neural Network)")
    ap.add_argument("--n-iter", type=int, default=20, help="random-search draws for RF/XGB/LGBM")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--leak-reps", type=int, default=20, help="label-permutation refits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--as-given", action="store_true",
                    help="skip in-fold imputation/winsorization (only for an already complete, clean workbook)")
    ap.add_argument("--no-shap", action="store_true", help="skip the SHAP report")
    ap.add_argument("--shap-bg", type=int, default=30, help="background rows per fold for SHAP")
    ap.add_argument("--shap-perms", type=int, default=1,
                    help="built-in estimator: permutation pairs per background row")
    ap.add_argument("--shap-engine", choices=["auto", "builtin"], default="auto",
                    help="auto = use the shap package if installed, else built-in")
    ap.add_argument("--desc-fold", choices=["last", "first"], default="last",
                    help="descriptive statistics use the TRAINING rows of this fold only (default: last fold = largest training set)")
    ap.add_argument("--no-desc", action="store_true", help="skip the descriptive-statistics section")
    ap.add_argument("--no-ablation", action="store_true", help="skip the validation-flaw ablation")
    ap.add_argument("--abl-repeats", type=int, default=5, help="ablation: random-split repeats (S3-S5)")
    ap.add_argument("--abl-boot", type=int, default=1000, help="ablation: bootstrap resamples for the CIs")
    ap.add_argument("--no-report", action="store_true", help="skip the Word report")
    ap.add_argument("--report-name", default="IPO_underpricing_report.docx")
    ap.add_argument("--quick", action="store_true", help="fast smoke test (small search, few resamples)")
    args = ap.parse_args(argv)
    if args.quick:
        args.n_iter, args.n_boot, args.n_perm, args.leak_reps = 3, 200, 300, 3
        args.shap_bg = min(args.shap_bg, 10)
        args.abl_repeats, args.abl_boot = 2, 100
    os.makedirs(args.outdir, exist_ok=True)
    global CLEAN_IN_FOLD
    CLEAN_IN_FOLD = not args.as_given

    df = load_data(args.input)
    X, sector = build_features(df)
    if not CLEAN_IN_FOLD and X.drop(columns=["ipo_year_frac"]).isna().any().any():
        sys.exit("Input has missing/invalid values: drop --as-given so they are imputed inside each fold")
    print("Sector grouping:", sector.value_counts().to_dict())
    const = [c for c in X.columns if X[c].nunique() <= 1 and c != "ipo_year_frac"]
    if const:                      # e.g. an empty sector bucket: a constant column carries no information
        print("Dropping constant (empty) feature columns:", const)
        for cols in list(PILLARS.values()):
            cols[:] = [c for c in cols if c not in const]
        FEATURE_SETS[ALL_NAME][:] = [c for cols in PILLARS.values() for c in cols]
    models = available_models([m.strip() for m in args.models.split(",") if m.strip()])
    folds = make_folds(df)
    ft = fold_table(df, folds)
    print("\nFOLD TABLE\n" + ft.drop(columns=["dropped"]).to_string(index=False))

    print("\nRunning expanding-window experiment ...")
    preds, best_params, caches, fitted = run_experiment(df, X, folds, models, args)
    fold_df, pool = summarize(preds, args)
    d_models, d_pillar, mcn, wil = pairwise_tests(preds, fold_df, models)

    print("\nClassical logit inference (full sample, descriptive) ...")
    coef, set_stats, drop_lr, vif = logit_inference(df, X)

    print("\nLeakage diagnostics ...")
    integrity = leakage_integrity(df, folds, X)
    perm = leakage_label_permutation(df, folds, caches, best_params, models, args.leak_reps, args.seed,
                                     args.jobs, preds)
    uni = leakage_univariate(df, X)
    rvc = leakage_random_vs_chrono(df, X, folds, models, args)

    # ---- tidy tables
    pool["model_name"] = pool["model"].map(lambda m: MODEL_LABEL.get(m, "Baseline: train prior"))
    wide = pool.pivot(index="set", columns="model_name", values="auc").reindex(
        list(FEATURE_SETS) + list(CONTROL_SETS))
    fold_wide = fold_df[fold_df["set"] == ALL_NAME].pivot(index="fold", columns="model", values="auc")
    bp = pd.DataFrame([{"fold": k[0], "set": k[1], "model": MODEL_LABEL[k[2]], "params": str(v)}
                       for k, v in best_params.items()])
    fold_df["model"] = fold_df["model"].map(lambda m: MODEL_LABEL.get(m, "Baseline: train prior"))
    preds_out = preds.assign(model=preds["model"].map(lambda m: MODEL_LABEL.get(m, "Baseline: train prior")))

    xlsx = os.path.join(args.outdir, "underpricing_results.xlsx")
    with pd.ExcelWriter(xlsx) as xw:
        ft.to_excel(xw, sheet_name="fold_table", index=False)
        wide.to_excel(xw, sheet_name="pooled_auc_matrix")
        pool.drop(columns=["model_name"]).assign(model=pool["model_name"]).to_excel(xw, sheet_name="pooled_metrics", index=False)
        fold_df.to_excel(xw, sheet_name="fold_metrics", index=False)
        fold_wide.to_excel(xw, sheet_name="fold_auc_ALL")
        d_models.to_excel(xw, sheet_name="delong_model_vs_model", index=False)
        d_pillar.to_excel(xw, sheet_name="delong_ALL_vs_pillar", index=False)
        mcn.to_excel(xw, sheet_name="mcnemar", index=False)
        wil.to_excel(xw, sheet_name="wilcoxon_fold_auc", index=False)
        coef.to_excel(xw, sheet_name="logit_ALL_odds_ratios", index=False)
        set_stats.to_excel(xw, sheet_name="logit_fit_by_set", index=False)
        drop_lr.to_excel(xw, sheet_name="logit_drop_pillar_LR", index=False)
        vif.to_excel(xw, sheet_name="vif", index=False)
        integrity.to_excel(xw, sheet_name="leak_integrity", index=False)
        perm.to_excel(xw, sheet_name="leak_label_permutation", index=False)
        uni.to_excel(xw, sheet_name="leak_univariate", index=False)
        rvc.to_excel(xw, sheet_name="leak_random_vs_chrono", index=False)
        bp.to_excel(xw, sheet_name="best_params", index=False)
        preds_out.to_excel(xw, sheet_name="oos_predictions", index=False)
    make_plot(pool, os.path.join(args.outdir, "pooled_auc_by_pillar.png"))
    export_processed(df, X, sector, folds, caches, os.path.join(args.outdir, "preprocessed_data_by_fold.xlsx"))
    cand = pool[(pool["set"] == ALL_NAME) & (pool["model"].isin(models))]
    best_model = cand.sort_values("auc", ascending=False)["model"].iloc[0]
    shp = None
    if not args.no_shap:
        print("\nSHAP report (probability units) ...")
        shp = build_shap_report(df, folds, caches, fitted, models, args, best=best_model)
    desc = None
    if not args.no_desc:
        print(f"\nDescriptive statistics (training rows of the {args.desc_fold} fold only) ...")
        desc = descriptive_analysis(df, X, sector, folds, caches, args)
    abl = None
    if not args.no_ablation:
        print("\nValidation-flaw ablation (default hyper-parameters, ALL pillars) ...")
        abl = ablation_study(df, X, models, folds, args)
    ok_int = bool(integrity[["train_before_test", "embargo_respected", "dropped_not_in_test",
                             "test_dates_after_all_train_dates"]].all().all()
                  and (integrity["row_overlap_train_test"] == 0).all())
    if not args.no_report:
        print("\nWord report ...")
        build_word_report({"args": args, "df": df, "folds": folds, "ft": ft, "pool": pool, "models": models,
                           "best_model": best_model, "wide": wide, "fold_wide": fold_wide, "fold_df": fold_df,
                           "d_models": d_models, "d_pillar": d_pillar, "mcn": mcn, "wil": wil, "coef": coef,
                           "set_stats": set_stats, "drop_lr": drop_lr, "vif": vif, "integrity": integrity,
                           "integrity_ok": ok_int, "perm": perm, "uni": uni, "rvc": rvc, "bp": bp,
                           "auc_png": os.path.join(args.outdir, "pooled_auc_by_pillar.png"),
                           "shap": shp, "desc": desc, "abl": abl},
                          os.path.join(args.outdir, args.report_name))

    # ---- console summary
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\nPOOLED OUT-OF-SAMPLE AUC (all test years combined)")
    print(wide.round(3).to_string())
    top = pool[pool["set"] == ALL_NAME][["model_name", "auc", "auc_ci_low", "auc_ci_high", "perm_p_value_auc",
                                         "accuracy", "balanced_accuracy", "mean_fold_auc"]]
    print("\nALL PILLARS COMBINED\n" + top.round(3).to_string(index=False))
    print("\nLEAKAGE CHECKS")
    print(f"  temporal integrity (train < embargo < test, no row overlap): {'PASS' if ok_int else 'FAIL'}")
    print(f"  label-permutation refit (AUC should be ~0.5):\n{perm.round(3).to_string(index=False)}")
    sus = uni[uni.iloc[:, -1]]
    print(f"  suspicious single features (|AUC-0.5|>0.20): {list(sus['feature']) or 'none'}")
    print(f"  random vs chronological gap (diagnostic):\n{rvc.round(3).to_string(index=False)}")
    print(f"\nSaved: {xlsx}\n       {os.path.join(args.outdir, 'pooled_auc_by_pillar.png')}")
    return preds, pool


if __name__ == "__main__":
    main(sys.argv[1:])
