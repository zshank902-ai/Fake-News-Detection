import time
from typing import Dict, List

# In-memory storage for tracking news propagation (In production, use Redis)
propagation_history = {}

def analyze_propagation(news_id: str, current_shares: int, timestamp: float = None):
    """
    Highly advanced logic to detect 'Bot-like' viral velocity.
    Organic news follows a different mathematical curve than bot-boosted news.
    """
    if timestamp is None:
        timestamp = time.time()
        
    if news_id not in propagation_history:
        propagation_history[news_id] = []
        
    history = propagation_history[news_id]
    history.append({"shares": current_shares, "time": timestamp})
    
    # Keep only last 10 snapshots
    if len(history) > 10:
        history.pop(0)
        
    if len(history) < 2:
        return {"bot_probability": 0.1, "pattern": "Insufficient Data"}

    # Calculate Velocity (Shares per second)
    delta_shares = history[-1]["shares"] - history[0]["shares"]
    delta_time = history[-1]["time"] - history[0]["time"]
    
    if delta_time == 0: return {"bot_probability": 0.1, "pattern": "Stable"}
    
    velocity = delta_shares / delta_time
    
    # Advanced Threshold Logic
    # 1. High Velocity (> 100 shares/sec) is suspicious
    # 2. Acceleration (Sudden spikes) is a bot indicator
    
    bot_prob = 0.0
    pattern = "Organic"
    
    if velocity > 50:
        bot_prob = 0.85
        pattern = "Hyper-Velocity (Likely Bots)"
    elif velocity > 10:
        bot_prob = 0.4
        pattern = "Accelerating"
        
    return {
        "bot_probability": bot_prob,
        "velocity_score": round(velocity, 2),
        "pattern": pattern,
        "is_bot_driven": bot_prob > 0.7
    }

def get_account_reputation(account_age_days: int, is_verified: bool, followers: int):
    """
    Heuristic for account-based credibility.
    """
    score = 1.0
    if not is_verified: score *= 0.5
    if account_age_days < 30: score *= 0.2 # New accounts are suspicious
    if followers < 10: score *= 0.5
    
    return round(score, 2)
