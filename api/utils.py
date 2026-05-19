import requests
import os

# Google Fact Check Tools API Key (User should set this)
# https://developers.google.com/fact-check/tools/api
FACT_CHECK_API_KEY = os.getenv("FACT_CHECK_API_KEY", "YOUR_API_KEY")

def cross_verify_fact(query: str):
    """
    Highly advanced cross-verification using Google Fact Check Tools API.
    Ensembles model prediction with verified fact databases.
    """
    if not query or FACT_CHECK_API_KEY == "YOUR_API_KEY":
        return None

    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query}&key={FACT_CHECK_API_KEY}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if "claims" in data and len(data["claims"]) > 0:
            # Found matching claims!
            top_claim = data["claims"][0]
            rating = top_claim["claimReview"][0]["textualRating"].lower()
            
            # Simple heuristic mapping
            if any(x in rating for x in ["false", "incorrect", "fake", "misleading"]):
                return {"verdict": "FAKE", "source": top_claim["claimReview"][0]["publisher"]["name"]}
            elif any(x in rating for x in ["true", "correct", "verified"]):
                return {"verdict": "REAL", "source": top_claim["claimReview"][0]["publisher"]["name"]}
        
        return None
    except Exception as e:
        print(f"Fact-Check API Error: {e}")
        return None

def calculate_truth_points(prediction: str, confidence: float, fact_check_match: bool):
    """
    Gamification logic: Rewards users for identifying debunked news.
    """
    base_points = 10 if prediction == "FAKE" else 2
    multiplier = 2 if fact_check_match else 1
    return int(base_points * confidence * multiplier)
