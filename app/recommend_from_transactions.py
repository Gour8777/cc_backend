# app/recommend_from_transactions.py
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import lru_cache

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

FEATURES = [
    "similarity",
    "bert_similarity",
    "user_cat_enc",
    "card_cat_enc",
    "card_apr_scaled_inv",
    "card_annual_fee_scaled_inv",
    "card_joining_fee_scaled_inv",
]


# ---------- Lazy loaders for models + artefacts ----------

@lru_cache(maxsize=1)
def get_ranker_and_artefacts():
    """
    Load XGBRanker, label encoders, scalers and cards_df once.
    """
    ranker = xgb.XGBRanker()
    ranker.load_model(str(MODELS_DIR / "xgbranker_model.json"))

    with open(MODELS_DIR / "le_user.pkl", "rb") as f:
        le_user = pickle.load(f)
    with open(MODELS_DIR / "le_card.pkl", "rb") as f:
        le_card = pickle.load(f)
    with open(MODELS_DIR / "scaler_apr.pkl", "rb") as f:
        scaler_apr = pickle.load(f)
    with open(MODELS_DIR / "scaler_af.pkl", "rb") as f:
        scaler_af = pickle.load(f)
    with open(MODELS_DIR / "scaler_jf.pkl", "rb") as f:
        scaler_jf = pickle.load(f)

    cards_df = pd.read_csv(MODELS_DIR / "cards.csv")

    return ranker, le_user, le_card, scaler_apr, scaler_af, scaler_jf, cards_df


@lru_cache(maxsize=1)
def get_tfidf_for_cards():
    """
    Fit TF-IDF on card texts and cache:
      - vectorizer
      - card-term matrix
    This is done once per process and reused for every user.
    """
    _, _, _, _, _, _, cards_df = get_ranker_and_artefacts()

    card_texts = (
        cards_df["card_description"].fillna("")
        + " "
        + cards_df["card_reward_category"].fillna("")
    ).tolist()

    vectorizer = TfidfVectorizer(
        max_features=5000,      # cap vocab size for memory
        ngram_range=(1, 2),     # unigrams + bigrams
        stop_words="english",
    )
    card_matrix = vectorizer.fit_transform(card_texts)  # (num_cards, vocab_size)

    return vectorizer, card_matrix


# ---------- Feature builder ----------

def build_pairs_for_user(user_id: str, txns: pd.DataFrame) -> pd.DataFrame:
    """
    Build a 'pairs'-like dataframe for a single user from their
    Firestore transactions. Assumes txns already has ONLY non-income rows.
    """
    if txns.empty:
        return pd.DataFrame()

    (
        ranker,
        le_user,
        le_card,
        scaler_apr,
        scaler_af,
        scaler_jf,
        cards_df,
    ) = get_ranker_and_artefacts()

    vectorizer, card_matrix = get_tfidf_for_cards()

    # --------- 1) Build user profile text ---------
    user_text = " ".join(
        (txns["category"].fillna("") + " " + txns["description"].fillna("")).tolist()
    )
    if not user_text.strip():
        # No meaningful text to compare
        return pd.DataFrame()

    # --------- 2) Compute TF-IDF similarity between user and cards ---------
    user_vec = vectorizer.transform([user_text])  # (1, vocab_size)

    sims = cosine_similarity(
        user_vec,
        card_matrix,
    )[0]  # shape: (num_cards,)

    df = cards_df.copy()
    df["similarity"] = sims
    # Keep the feature name 'bert_similarity' so XGB model still works
    df["bert_similarity"] = df["similarity"]

    # --------- 3) Encode user category safely (handle unseen) ---------
    if not txns["category"].mode().empty:
        main_cat = txns["category"].mode().iloc[0]
    else:
        main_cat = "other"

    if main_cat not in le_user.classes_:
        if "other" in le_user.classes_:
            fallback = "other"
        else:
            fallback = le_user.classes_[0]
        print(
            f"⚠️ Unseen user category '{main_cat}', mapping to '{fallback}'",
            flush=True,
        )
        main_cat_enc = le_user.transform([fallback])[0]
    else:
        main_cat_enc = le_user.transform([main_cat])[0]

    df["user_cat_enc"] = main_cat_enc

    # --------- 4) Encode card categories safely (handle unseen) ---------
    card_cats = df["card_reward_category"].fillna("other")
    safe_card_enc = []
    for cat in card_cats:
        if cat not in le_card.classes_:
            if "other" in le_card.classes_:
                fallback = "other"
            else:
                fallback = le_card.classes_[0]
            print(
                f"⚠️ Unseen card category '{cat}', mapping to '{fallback}'",
                flush=True,
            )
            safe_card_enc.append(le_card.transform([fallback])[0])
        else:
            safe_card_enc.append(le_card.transform([cat])[0])

    df["card_cat_enc"] = safe_card_enc

    # --------- 5) Scale APR / fees & invert (same as training) ---------
    df["card_apr_scaled"] = scaler_apr.transform(df[["card_apr"]])
    df["card_apr_scaled_inv"] = 1 - df["card_apr_scaled"]

    df["card_annual_fee_scaled"] = scaler_af.transform(df[["card_annual_fee"]])
    df["card_annual_fee_scaled_inv"] = 1 - df["card_annual_fee_scaled"]

    df["card_joining_fee_scaled"] = scaler_jf.transform(df[["card_joining_fee"]])
    df["card_joining_fee_scaled_inv"] = 1 - df["card_joining_fee_scaled"]

    df["user_id"] = user_id

    return df


# ---------- Main public function ----------

def recommend_for_user(user_id: str, txns: pd.DataFrame, top_k: int = 5):
    """
    Full pipeline for one user:
      - build features from their Firestore transactions
      - run XGBRanker
      - return top_k recommended cards
    """
    pairs_user = build_pairs_for_user(user_id, txns)
    if pairs_user.empty:
        return []

    ranker, *_ = get_ranker_and_artefacts()

    X = pairs_user[FEATURES]
    pairs_user["score"] = ranker.predict(X)
    top = pairs_user.sort_values("score", ascending=False).head(top_k)

    results = []
    for _, row in top.iterrows():
        results.append(
            {
                "card_name": row["card_name"],
                "card_bank": row.get("card_bank", ""),
                "card_reward_category": row.get("card_reward_category", ""),
                "score": float(row["score"]),
                "annual_fee": float(row["card_annual_fee"]),
                "joining_fee": float(row["card_joining_fee"]),
                "apr": float(row["card_apr"]),
            }
        )
    return results
