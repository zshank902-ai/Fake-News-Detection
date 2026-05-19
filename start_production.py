import uvicorn
import multiprocessing
import os

def start_truth_shield_production():
    """
    Launches the Truth Shield SOTA API in production mode.
    Uses multiple workers to ensure high availability and zero-crash capability.
    """
    print("🛡️ LAUNCHING TRUTH SHIELD SOTA ENGINE: PRODUCTION MODE")
    print("🚀 Workers: ", multiprocessing.cpu_count() if multiprocessing.cpu_count() < 4 else 4)
    print("🌐 Port: 8000")
    
    # Run with 4 workers for high availability
    # In production, we use 4 workers to handle concurrent load and provide failover
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,  # Multiple processes for zero-crash
        log_level="info",
        reload=False  # No reload in production for stability
    )

if __name__ == "__main__":
    # Ensure logs directory exists
    if not os.path.exists('api/logs'):
        os.makedirs('api/logs')
        
    try:
        start_truth_shield_production()
    except Exception as e:
        print(f"🚨 CRITICAL SYSTEM FAILURE: {str(e)}")
