# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from .firestore_client import fetch_user_transactions
from .recommend_from_transactions import recommend_for_user

app = FastAPI(title="CC Recommender with Firestore")

class Recommendation(BaseModel):
    card_name: str
    card_bank: str | None = None
    card_reward_category: str | None = None
    score: float
    annual_fee: float
    joining_fee: float
    apr: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/recommendations/{user_id}", response_model=List[Recommendation])
def get_recommendations(user_id: str, top_k: int = 5):
    print("➡️ /recommendations called for user:", user_id, flush=True)

    txns = fetch_user_transactions(user_id)
    print("📊 fetched raw transactions shape:", txns.shape, flush=True)

    if txns.empty:
        raise HTTPException(status_code=404, detail="No non-income transactions for this user")

    print("🤖 calling recommend_for_user...", flush=True)
    results = recommend_for_user(user_id, txns, top_k=top_k)
    print("✅ got results length:", len(results), flush=True)

    if not results:
        raise HTTPException(status_code=500, detail="No recommendations generated")

    return results
