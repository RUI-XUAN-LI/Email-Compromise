#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Email multi-class classification using SVM (TF-IDF + LinearSVC):
"""

import os
import glob
import random
import warnings
import numpy as np
import pandas as pd
from typing import List, Tuple

import pyarrow.parquet as pq
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import dump

warnings.filterwarnings("ignore")

# -------------------- CONFIG --------------------
DATA_DIR = "./data/email_logs/"  # Replace with your dataset directory
TRAIN_LABELS = ["Phish", "Fraud", "Gambling", "Drug", "Adult"]
N_PER_CLASS = 2000       # Number of training samples per class
RANDOM_STATE = 42
MIN_CONTENT_LEN = 10     # Filter out too-short text
MODEL_PATH = "./model/svm_tfidf_phishurl.joblib"
TEST_PRED_CSV = "./output/svm_test_predictions.csv"
OTHER_PRED_CSV = "./output/svm_other_predictions.csv"
SAMPLE_SUMMARY_CSV = "./output/svm_training_sampling_summary.csv"
REQUIRED_COLUMNS = ["mail_type", "content", "phishurl"]
# ------------------------------------------------


def list_parquet_files(root: str) -> List[str]:
    patterns = [
        os.path.join(root, "**", "*.parquet"),
        os.path.join(root, "**", "*.parq"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    return files


def safe_read_one_parquet(path: str, required_cols: List[str]) -> pd.DataFrame:
    """
    Read only existing columns; add missing columns with NaN to avoid KeyError.
    """
    try:
        pf = pq.ParquetFile(path)
        names = pf.schema.names
        read_cols = [c for c in required_cols if c in names]
        if not read_cols:
            # No required columns found, return empty DataFrame
            df = pd.DataFrame(columns=required_cols)
        else:
            table = pf.read(columns=read_cols)
            df = table.to_pandas()
            # Fill missing columns with NaN
            for miss in set(required_cols) - set(df.columns):
                df[miss] = np.nan
            df = df[required_cols]
        return df
    except Exception as e:
        print(f"[WARN] Failed to read: {path} -> {e}")
        return pd.DataFrame(columns=required_cols)


def read_parquet_selected_columns(files: List[str], columns: List[str]) -> pd.DataFrame:
    dfs = []
    for f in files:
        df = safe_read_one_parquet(f, columns)
        if not df.empty:
            dfs.append(df)
    if not dfs:
        raise SystemExit("No valid parquet data found. Please check directory and column names.")
    return pd.concat(dfs, ignore_index=True)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mail_type"] = df["mail_type"].astype(str)
    df["content"] = df["content"].astype(str)
    df["phishurl"] = df["phishurl"].astype(str).fillna("")
    df["has_phish_url"] = df["phishurl"].apply(lambda x: 1 if str(x).strip() else 0)
    df["content"] = df["content"].fillna("").str.strip()
    df = df[df["content"].str.len() >= MIN_CONTENT_LEN].reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    return df


def stratified_sample_train_test(
    df: pd.DataFrame,
    train_labels: List[str],
    n_per_class: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    df_known = df[df["mail_type"].isin(train_labels)].copy()
    df_other = df[~df["mail_type"].isin(train_labels)].copy()
    train_chunks = []
    for label in train_labels:
        sub = df_known[df_known["mail_type"] == label]
        if sub.empty:
            print(f"[WARN] Class {label} not found in data, skipping.")
            continue
        k = min(n_per_class, len(sub))
        idx = rng.choice(sub.index.to_numpy(), size=k, replace=False)
        train_chunks.append(sub.loc[idx])
    if not train_chunks:
        raise SystemExit("No samples available for training classes.")
    train_df = pd.concat(train_chunks, ignore_index=False)
    test_df = df_known.loc[~df_known.index.isin(train_df.index)].copy()
    train_df = shuffle(train_df, random_state=random_state).reset_index(drop=True)
    test_df = shuffle(test_df, random_state=random_state).reset_index(drop=True)
    return train_df, test_df, df_other


class ContentSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X["content"].tolist()


class PhishURLSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.array(X["has_phish_url"]).reshape(-1, 1)


def build_pipeline() -> Pipeline:
    text_features = Pipeline([
        ("selector", ContentSelector()),
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
            lowercase=True
        ))
    ])
    phish_url_feature = Pipeline([
        ("selector", PhishURLSelector())
    ])
    combined_features = FeatureUnion([
        ("text", text_features),
        ("phish_flag", phish_url_feature)
    ])
    pipe = Pipeline([
        ("features", combined_features),
        ("clf", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE))
    ])
    return pipe


def main():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    print(">>> Scanning parquet files...")
    files = list_parquet_files(DATA_DIR)
    if not files:
        raise SystemExit(f"No parquet files found in: {DATA_DIR}")
    print(f"Found {len(files)} parquet files.")

    print(">>> Reading data (mail_type, content, phishurl)...")
    df = read_parquet_selected_columns(files, columns=REQUIRED_COLUMNS)
    print(f"Total records: {len(df):,}")

    print(">>> Cleaning and feature engineering...")
    df = prepare_data(df)
    print(f"After cleaning: {len(df):,}")
    print("Label distribution:")
    print(df["mail_type"].value_counts())

    print(">>> Stratified sampling (train/test split)...")
    train_df, test_df, other_df = stratified_sample_train_test(
        df, TRAIN_LABELS, N_PER_CLASS, RANDOM_STATE
    )

    pd.DataFrame({
        "label": TRAIN_LABELS,
        "available_count": [int((df["mail_type"] == lb).sum()) for lb in TRAIN_LABELS],
        "train_count": [int((train_df["mail_type"] == lb).sum()) for lb in TRAIN_LABELS],
        "test_count": [int((test_df["mail_type"] == lb).sum()) for lb in TRAIN_LABELS],
    }).to_csv(SAMPLE_SUMMARY_CSV, index=False)
    print(f"Sampling summary saved: {SAMPLE_SUMMARY_CSV}")

    print(">>> Training pipeline (TF-IDF + phishing flag + LinearSVC)...")
    pipe = build_pipeline()
    pipe.fit(train_df, train_df["mail_type"])

    print(">>> Evaluating on test set...")
    y_true = test_df["mail_type"].tolist()
    y_pred = pipe.predict(test_df)
    acc = accuracy_score(y_true, y_pred)
    print(f"\n[TEST] Accuracy: {acc:.4f}\n")
    print(classification_report(y_true, y_pred, digits=4, labels=TRAIN_LABELS, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=TRAIN_LABELS)
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in TRAIN_LABELS],
                         columns=[f"pred_{c}" for c in TRAIN_LABELS])
    print("\nConfusion Matrix:")
    print(cm_df)

    test_out = test_df.copy()
    test_out["pred_mail_type"] = y_pred
    test_out.to_csv(TEST_PRED_CSV, index=False)
    print(f"\nTest predictions saved: {TEST_PRED_CSV}")

    if len(other_df) > 0:
        print("\n>>> Predicting on other samples...")
        other_pred = pipe.predict(other_df)
        other_out = other_df.copy()
        other_out["pred_mail_type"] = other_pred
        other_out.to_csv(OTHER_PRED_CSV, index=False)
        print(f"Other predictions saved: {OTHER_PRED_CSV}")
    else:
        print("\nNo samples outside the training classes found.")

    dump(pipe, MODEL_PATH)
    print(f"\nModel saved: {MODEL_PATH}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
