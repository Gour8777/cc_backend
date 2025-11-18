# app/firestore_client.py
from pathlib import Path
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import os, json
from google.cloud import firestore
from google.oauth2 import service_account

BASE_DIR = Path(__file__).resolve().parent.parent
SERVICE_ACCOUNT_FILE = BASE_DIR / "serviceAccount.json"
load_dotenv()

# Initialize Firebase app once
if not firebase_admin._apps:
    firebase_creds = json.loads(os.environ["FIREBASE_CREDENTIALS"])
    credentials = service_account.Credentials.from_service_account_info(firebase_creds)
    firebase_admin.initialize_app(credentials)


db = firestore.Client(credentials=credentials, project=firebase_creds["project_id"])




def fetch_user_transactions(user_id: str) -> pd.DataFrame:
    """
    Fetch transactions for a given Firebase user ID from:
        users/{user_id}/transactions/{txnId}

    Returns a DataFrame with ONLY non-income transactions
    (i.e., filters out rows where type == 'income').
    """
    docs = (
        db.collection("users")
        .document(user_id)
        .collection("transactions")
        .stream()
    )

    rows = []
    for doc in docs:
        d = doc.to_dict() or {}
        rows.append(
            {
                "amount": float(d.get("amount", 0)),
                "category": d.get("category", "other"),
                "currency": d.get("currency", "INR"),
                "date": d.get("date"),
                "description": d.get("description", ""),
                "type": d.get("type", "expense"),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # 🔥 Filter-out income transactions
    # Normalize 'type' to lowercase, then drop rows where type == 'income'
    df["type"] = df["type"].fillna("").str.lower()
    df = df[df["type"] != "income"]

    return df
