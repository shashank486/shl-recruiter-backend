from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from dotenv import load_dotenv

# Load env variables at startup
load_dotenv()

from app.schemas import ChatRequest, ChatResponse, HealthResponse, AnalyticsOverview
from app.rag import process_chat
from app.catalog import catalog_manager
from app.analytics import analytics_tracker

app = FastAPI(
    title="SHL AI Recruiter Assistant Backend",
    description="Conversational AI recommender backend for SHL products",
    version="1.0.0"
)

# Enable CORS for Next.js frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to the frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok")

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")
    try:
        response = process_chat(request.messages)
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# Alias: /chat → same handler (assignment requires POST /chat)
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint_alias(request: ChatRequest):
    return chat_endpoint(request)

@app.get("/api/catalog")
def get_catalog_endpoint(
    query: Optional[str] = None,
    duration: Optional[str] = None,
    language: Optional[str] = None,
    test_type: Optional[str] = None,
    skill: Optional[str] = None
):
    try:
        results = catalog_manager.manual_filter(
            query=query,
            duration=duration,
            language=language,
            test_type=test_type,
            skill=skill
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying catalog: {str(e)}")

@app.get("/api/analytics", response_model=AnalyticsOverview)
def get_analytics_endpoint():
    try:
        summary = analytics_tracker.get_summary()
        return AnalyticsOverview(**summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving analytics: {str(e)}")
