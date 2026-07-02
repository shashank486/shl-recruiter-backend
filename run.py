import uvicorn
import os
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    port = int(os.getenv("PORT", 8000))
    # reload=True only for local dev, never in production
    is_dev = os.getenv("ENV", "production").lower() == "development"
    print(f"Starting SHL Recruiter Backend on port {port} (dev={is_dev})...")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=is_dev,
    )
