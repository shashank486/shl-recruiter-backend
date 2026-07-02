# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Message(BaseModel):
    role: str = Field(..., description="Role of the message author: 'user', 'assistant', 'system'")
    content: str = Field(..., description="Text content of the message")

class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., description="List of previous messages in the conversation")

class Recommendation(BaseModel):
    name: str = Field(..., description="Name of the assessment")
    url: str = Field(..., description="URL to view assessment details")
    test_type: str = Field(..., description="Single-letter abbreviation of the test type (e.g., K, P, A, B, S, C, D)")

class RichRecommendationDetail(BaseModel):
    name: str = Field(..., description="Name of the assessment")
    description: str = Field(..., description="Detailed description of the assessment")
    duration: str = Field(..., description="Approximate completion time")
    languages: List[str] = Field(default_factory=list, description="Supported languages")
    reason: str = Field(..., description="Justification explaining why this test was recommended")
    match_score: int = Field(..., description="Confidence score from 0 to 100")

class TimelineEvent(BaseModel):
    stage: str = Field(..., description="Conversational stage/milestone")
    completed: bool = Field(..., description="True if this stage has been completed")

class Filters(BaseModel):
    role: Optional[str] = Field(None, description="Extracted target job role")
    experience: Optional[str] = Field(None, description="Target experience level")
    industry: Optional[str] = Field(None, description="Target industry")
    hiring_purpose: Optional[str] = Field(None, description="Hiring purpose (e.g., selection vs development)")

class ChatState(BaseModel):
    summary: str = Field(..., description="Brief conversation context summary")
    skills: List[str] = Field(default_factory=list, description="Extracted skills required for the role")
    experience: str = Field(..., description="Extracted experience level")
    leadership_required: str = Field(..., description="Whether leadership behaviors are required ('Yes', 'No', 'Unsure')")
    communication_required: str = Field(..., description="Whether communication skills are key ('Yes', 'No', 'Unsure')")
    filters: Filters = Field(default_factory=Filters, description="Active catalog filters derived from conversation")
    progress: int = Field(..., description="Intake process progress percentage (0-100)")
    missing_fields: List[str] = Field(default_factory=list, description="Information fields still needed from the user")
    timeline: List[TimelineEvent] = Field(default_factory=list, description="Timeline of intake milestones")
    recommendation_details: List[RichRecommendationDetail] = Field(default_factory=list, description="Rich details for recommended assessments")

class ChatResponse(BaseModel):
    reply: str = Field(..., description="Conversational response from the AI assistant")
    recommendations: List[Recommendation] = Field(default_factory=list, description="List of recommended assessments, if ready")
    end_of_conversation: bool = Field(..., description="True if recommendation has been final and complete")
    state: ChatState = Field(..., description="Current derived conversation state details")

class HealthResponse(BaseModel):
    status: str = "ok"

class AnalyticsOverview(BaseModel):
    conversation_count: int
    average_recommendation_time_sec: float
    popular_assessments: List[Dict[str, Any]]
    average_clarification_questions: float
    hallucination_count: int
