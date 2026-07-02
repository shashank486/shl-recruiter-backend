import os
import json
import time
import urllib.request
from typing import List, Dict, Any, Optional

from app.catalog import catalog_manager
from app.schemas import (
    Message, ChatResponse, ChatState, ChatRequest,
    Recommendation, RichRecommendationDetail, Filters, TimelineEvent
)
from app.analytics import analytics_tracker

# ─── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an elite SHL Conversational Recruiter Assistant. Your goal is to guide
recruiters to the best SHL assessments for their hiring needs.

CONVERSATION FLOW RULES:
1. DO NOT IMMEDIATELY RECOMMEND assessments. First ask 2-3 clarification questions to understand:
   - Target role & key skills.
   - Experience level (e.g., Entry-level, Graduate, Mid-Professional, Director).
   - Hiring purpose or special needs (e.g. client interaction, leadership, personality fit).
2. Maintain full context from previous turns.
3. If the user asks anything unrelated to hiring, recruitment, or assessments (e.g. how to bake a
   cake, write code, etc.), politely refuse and redirect them back to SHL assessments.
4. If the user asks to compare specific assessments (e.g. "OPQ32r vs Verify G+"), explain the
   comparison in the reply and trigger the comparison view.
5. Only set 'is_ready_to_recommend' to true when you have sufficient details (Role AND Experience
   Level at minimum, and ideally communication or personality preferences) OR if the user
   explicitly demands recommendations.

OUTPUT FORMAT:
Respond with a single valid JSON object — no markdown fences, no extra text. Keys required:
{
  "summary": "Brief conversation summary",
  "skills": ["extracted", "skills"],
  "experience": "extracted experience level",
  "leadership_required": "Yes | No | Unsure",
  "communication_required": "Yes | No | Unsure",
  "filters": {
    "role": "extracted role or null",
    "experience": "extracted experience or null",
    "industry": "extracted industry or null",
    "hiring_purpose": "extracted purpose or null"
  },
  "progress": <integer 0-100>,
  "missing_fields": ["field1", "field2"],
  "is_ready_to_recommend": true | false,
  "refusal": "refusal text if out-of-scope, else null",
  "reply": "Your next conversational response",
  "comparison_request": {
    "is_comparison": true | false,
    "assessments": ["name1", "name2"]
  }
}
"""

# ─── LLM call (OpenAI → Gemini → rule-based fallback) ────────────────────────
def call_llm(messages: List[Message]) -> Dict[str, Any]:
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    history = [{"role": m.role, "content": m.content} for m in messages]
    payload_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    # ── OpenAI ────────────────────────────────────────────────────────────────
    if openai_key:
        try:
            url = "https://api.openai.com/v1/chat/completions"
            data = json.dumps({
                "model": "gpt-4o-mini",
                "messages": payload_messages,
                "response_format": {"type": "json_object"},
                "temperature": 0.2,
            }).encode("utf-8")
            req = urllib.request.Request(
                url, data=data,
                headers={"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return json.loads(body["choices"][0]["message"]["content"])
        except Exception as e:
            print(f"RAG: OpenAI failed, falling back: {e}")

    # ── Gemini ────────────────────────────────────────────────────────────────
    if gemini_key:
        try:
            contents = []
            for m in messages:
                role = "user" if m.role == "user" else "model"
                contents.append({"role": role, "parts": [{"text": m.content}]})

            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-2.5-flash:generateContent?key={gemini_key}"
            )
            data = json.dumps({
                "contents": contents,
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "temperature": 0.2,
                },
            }).encode("utf-8")
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                text = body["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(text)
        except Exception as e:
            print(f"RAG: Gemini failed, falling back: {e}")

    # ── Rule-based fallback ───────────────────────────────────────────────────
    return _rule_based_fallback(messages)


# ─── Rule-based fallback ──────────────────────────────────────────────────────
def _default_state(reply: str) -> Dict[str, Any]:
    return {
        "summary": "Started conversation",
        "skills": [],
        "experience": "Unknown",
        "leadership_required": "Unsure",
        "communication_required": "Unsure",
        "filters": {"role": None, "experience": None, "industry": None, "hiring_purpose": None},
        "progress": 10,
        "missing_fields": ["Role", "Experience", "Skills"],
        "is_ready_to_recommend": False,
        "refusal": None,
        "reply": reply,
        "comparison_request": {"is_comparison": False, "assessments": []},
    }


def _rule_based_fallback(messages: List[Message]) -> Dict[str, Any]:
    user_inputs = [m.content.lower() for m in messages if m.role == "user"]
    if not user_inputs:
        return _default_state(
            "Welcome! I'm your SHL Assessment Recommender. What role are you hiring for today?"
        )

    last = user_inputs[-1]

    # Out-of-scope refusal
    oos_keywords = [
        "bake", "cake", "recipe", "weather", "translate",
        "code a", "programming", "quantum", "history", "joke",
    ]
    if any(kw in last for kw in oos_keywords):
        return {
            **_default_state(""),
            "summary": "Out of scope query",
            "progress": 0,
            "missing_fields": ["Role"],
            "refusal": (
                "I can only assist with SHL assessments and hiring. "
                "Let's get back to your hiring needs — what role are you hiring for?"
            ),
            "reply": (
                "I can only assist with SHL assessments and hiring. "
                "Let's get back to your hiring needs — what role are you hiring for?"
            ),
        }

    # Comparison request
    compare_kw = ["compare", " vs ", "versus", "difference between"]
    if any(kw in last for kw in compare_kw):
        assessments: List[str] = []
        if "opq32" in last:
            assessments.append("OPQ32r")
        if "verify g+" in last or "g+" in last:
            assessments.append("Verify G+")
        if "numerical" in last:
            assessments.append("Verify Numerical Reasoning")
        if "inductive" in last:
            assessments.append("Verify Inductive Reasoning")
        if "deductive" in last:
            assessments.append("Verify Deductive Reasoning")
        if len(assessments) < 2:
            assessments = ["OPQ32r", "Verify G+"]
        return {
            **_default_state(""),
            "summary": "Comparison requested",
            "progress": 50,
            "missing_fields": [],
            "reply": (
                f"Sure! Here's a side-by-side comparison for "
                f"**{', '.join(assessments)}**. "
                "You can review purpose, duration, languages, and ideal use cases below."
            ),
            "comparison_request": {"is_comparison": True, "assessments": assessments},
        }

    # Role / skill extraction
    role: Optional[str] = None
    skills: List[str] = []
    experience = "Unknown"
    leadership = "Unsure"
    communication = "Unsure"
    purpose = "Selection"

    for inp in user_inputs:
        if "java" in inp:
            role = "Java Developer"
            skills += ["Java", "Software Engineering", "Spring Boot", "Coding"]
        elif "python" in inp or "data scientist" in inp:
            role = "Data Scientist"
            skills += ["Python", "Statistics", "Machine Learning", "Data Analysis"]
        elif "sales" in inp or "account executive" in inp:
            role = "Sales Representative"
            skills += ["Sales", "Communication", "Negotiation", "Client Relationship"]
        elif "leader" in inp or "manager" in inp or "director" in inp:
            role = "Leadership Role"
            skills += ["Leadership", "Strategy", "Management", "People Skills"]
            leadership = "Yes"
        elif "analyst" in inp:
            role = "Business Analyst"
            skills += ["Analysis", "Problem Solving", "Data Interpretation"]

        if any(w in inp for w in ["senior", "mid", "4 year", "5 year", "6 year"]):
            experience = "Mid-Professional / Senior"
        elif any(w in inp for w in ["junior", "entry", "graduate", "fresher"]):
            experience = "Entry-Level / Graduate"
        elif any(w in inp for w in ["exec", "director", "vp", "c-level"]):
            experience = "Executive"

        if any(w in inp for w in ["client", "customer", "communication", "interpersonal"]):
            communication = "Yes"

        if any(w in inp for w in ["personality", "behaviour", "opq", "fit"]):
            purpose = "Selection & Fit"

    if not role:
        return _default_state(
            "To find the right assessments, could you tell me what role you're hiring for? "
            "(e.g. Java Developer, Sales Executive, General Manager)"
        )

    skills = list(set(skills))
    progress = 30
    missing: List[str] = []
    reply = ""
    is_ready = False

    if experience == "Unknown":
        missing.append("Experience Level")
        reply = "What's the experience level for this role? (e.g. Graduate, Entry-Level, Mid-Professional, Director)"
        progress = 50
    elif communication == "Unsure":
        missing.append("Client Interaction")
        reply = "Will the candidate interact with clients or need strong communication / interpersonal skills?"
        progress = 75
    elif len(user_inputs) < 4:
        missing.append("Assessment Focus")
        reply = (
            "Should we include personality/behavioural assessments (like OPQ32r) "
            "to measure work-style fit, or focus on technical skills tests?"
        )
        progress = 90
    else:
        progress = 100
        is_ready = True
        reply = (
            "Great — I have all the details I need. "
            "Here are the top SHL assessments I'd recommend for your role:"
        )

    return {
        "summary": f"Hiring for {role} ({experience})",
        "skills": skills,
        "experience": experience,
        "leadership_required": leadership,
        "communication_required": communication,
        "filters": {
            "role": role,
            "experience": experience if experience != "Unknown" else None,
            "industry": None,
            "hiring_purpose": purpose,
        },
        "progress": progress,
        "missing_fields": missing,
        "is_ready_to_recommend": is_ready,
        "refusal": None,
        "reply": reply,
        "comparison_request": {"is_comparison": False, "assessments": []},
    }


# ─── Main entry point ─────────────────────────────────────────────────────────
def process_chat(messages: List[Message]) -> ChatResponse:
    start = time.time()

    analysis = call_llm(messages)

    # Build timeline
    is_ready = bool(analysis.get("is_ready_to_recommend", False))
    progress = int(analysis.get("progress", 10))
    timeline = [
        TimelineEvent(stage="Intake Started",        completed=True),
        TimelineEvent(stage="Role Clarification",    completed=progress > 30),
        TimelineEvent(stage="Experience Level",      completed=progress > 60),
        TimelineEvent(stage="Requirements Defined",  completed=progress > 85),
        TimelineEvent(stage="Recommendations Ready", completed=is_ready),
    ]

    # ── Refusal path ──────────────────────────────────────────────────────────
    if analysis.get("refusal"):
        analytics_tracker.log_conversation(
            clarification_questions=len(messages) // 2,
            recommendation_time=time.time() - start,
            hallucination=True,
        )
        state = ChatState(
            summary=analysis.get("summary", "Out of scope"),
            skills=analysis.get("skills", []),
            experience=analysis.get("experience", "Unknown"),
            leadership_required=analysis.get("leadership_required", "Unsure"),
            communication_required=analysis.get("communication_required", "Unsure"),
            filters=Filters(**analysis.get("filters", {})),
            progress=analysis.get("progress", 0),
            missing_fields=analysis.get("missing_fields", []),
            timeline=timeline,
            recommendation_details=[],
        )
        return ChatResponse(
            reply=analysis["refusal"],
            recommendations=[],
            end_of_conversation=False,
            state=state,
        )

    # ── Build state ───────────────────────────────────────────────────────────
    state = ChatState(
        summary=analysis.get("summary", ""),
        skills=analysis.get("skills", []),
        experience=analysis.get("experience", "Unknown"),
        leadership_required=analysis.get("leadership_required", "Unsure"),
        communication_required=analysis.get("communication_required", "Unsure"),
        filters=Filters(**analysis.get("filters", {})),
        progress=progress,
        missing_fields=analysis.get("missing_fields", []),
        timeline=timeline,
        recommendation_details=[],
    )

    # ── Recommendations ───────────────────────────────────────────────────────
    recommendations: List[Recommendation] = []
    rich_details: List[RichRecommendationDetail] = []

    if is_ready:
        role = state.filters.role or "general"
        skills_str = " ".join(state.skills)
        query = f"{role} {skills_str}"

        job_levels: List[str] = []
        exp = state.experience.lower()
        if any(w in exp for w in ["entry", "graduate"]):
            job_levels = ["Entry-Level", "Graduate"]
        elif any(w in exp for w in ["mid", "professional", "senior"]):
            job_levels = ["Mid-Professional", "Professional Individual Contributor"]
        elif any(w in exp for w in ["exec", "director", "manager"]):
            job_levels = ["Executive", "Director", "Manager", "Front Line Manager"]

        test_types: List[str] = []
        if state.leadership_required == "Yes":
            test_types.append("Competencies")
            query += " Leadership Management"
        if state.communication_required == "Yes":
            query += " Communication Interpersonal"

        results = catalog_manager.search(
            query=query,
            top_k=6,
            job_levels=job_levels or None,
            test_types=test_types or None,
        )

        for prod, score in results:
            name = prod.get("name", "")
            desc = prod.get("description", "")
            keys_list = prod.get("keys") or []
            # Derive single-letter type code for frontend badge
            test_type_code = keys_list[0][:1].upper() if keys_list else "K"

            # Build reason string
            if state.skills:
                matching = [s for s in state.skills if s.lower() in (desc + name).lower()]
                reason = (
                    f"Measures key skills for this role: {', '.join(matching)}. "
                    f"Aligned with {state.experience} requirements."
                    if matching else
                    f"Industry-standard assessment for {role} benchmarking and behavioural fit."
                )
            else:
                reason = f"Recommended as a top SHL assessment for the {role} role."

            languages = prod.get("languages") or []

            recommendations.append(Recommendation(
                name=name,
                url=prod.get("link", "https://www.shl.com"),
                test_type=test_type_code,
            ))
            rich_details.append(RichRecommendationDetail(
                name=name,
                description=desc,
                duration=prod.get("duration", "Varies"),
                languages=languages,
                reason=reason,
                match_score=min(score, 100),
            ))

        # Embed rich details into state
        state.recommendation_details = rich_details

        analytics_tracker.log_conversation(
            clarification_questions=len(messages) // 2,
            recommendation_time=time.time() - start,
            recommended_assessments=[r.name for r in recommendations],
        )

    return ChatResponse(
        reply=analysis.get("reply", "Here are your recommended assessments:"),
        recommendations=recommendations,
        end_of_conversation=is_ready,
        state=state,
    )
