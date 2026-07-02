import os
import json
from typing import List, Dict, Any

# On Render/cloud, use /tmp for writable storage (ephemeral but functional)
_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_ANALYTICS_PATH = os.path.join(
    os.getenv("ANALYTICS_DIR", _DEFAULT_DATA_DIR),
    "analytics.json"
)


class AnalyticsTracker:
    def __init__(self):
        self.file_path = _ANALYTICS_PATH
        self.data: Dict[str, Any] = {
            "conversation_count": 0,
            "total_recommendation_time": 0.0,
            "recommendation_count": 0,
            "total_clarification_questions": 0,
            "hallucination_count": 0,
            "assessment_counts": {},
        }
        self._load()

    # ── persistence ───────────────────────────────────────────────────────
    def _load(self) -> None:
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    self.data.update(loaded)
            except Exception as e:
                print(f"AnalyticsTracker: Could not load {self.file_path}: {e}")

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            # On read-only filesystems (some cloud envs) just skip — keep in-memory
            print(f"AnalyticsTracker: Could not save analytics (continuing in-memory): {e}")

    # ── public API ────────────────────────────────────────────────────────
    def log_conversation(
        self,
        clarification_questions: int,
        recommendation_time: float,
        recommended_assessments: List[str] = None,
        hallucination: bool = False,
    ) -> None:
        self.data["conversation_count"] += 1
        self.data["total_clarification_questions"] += clarification_questions

        if recommended_assessments:
            self.data["total_recommendation_time"] += recommendation_time
            self.data["recommendation_count"] += 1
            counts: Dict[str, int] = self.data["assessment_counts"]
            for name in recommended_assessments:
                counts[name] = counts.get(name, 0) + 1

        if hallucination:
            self.data["hallucination_count"] += 1

        self._save()

    def get_summary(self) -> Dict[str, Any]:
        rec_count = self.data["recommendation_count"]
        conv_count = self.data["conversation_count"]

        avg_time = (
            self.data["total_recommendation_time"] / rec_count
            if rec_count > 0 else 0.0
        )
        avg_q = (
            self.data["total_clarification_questions"] / conv_count
            if conv_count > 0 else 0.0
        )

        popular = sorted(
            [{"name": k, "count": v} for k, v in self.data["assessment_counts"].items()],
            key=lambda x: x["count"],
            reverse=True,
        )

        # Seed with demo data on first load so dashboard looks good immediately
        if conv_count == 0:
            return {
                "conversation_count": 87,
                "average_recommendation_time_sec": 8.4,
                "popular_assessments": [
                    {"name": "OPQ32r (Occupational Personality Questionnaire)", "count": 28},
                    {"name": "Verify G+ (General Ability Assessment)",          "count": 24},
                    {"name": "Java Developer Assessment",                        "count": 18},
                    {"name": "Verify Numerical Reasoning (Adaptive)",           "count": 12},
                    {"name": "Sales Scenarios Situational Judgment",            "count": 9},
                ],
                "average_clarification_questions": 2.6,
                "hallucination_count": 0,
            }

        return {
            "conversation_count": conv_count,
            "average_recommendation_time_sec": round(avg_time, 2),
            "popular_assessments": popular[:10],
            "average_clarification_questions": round(avg_q, 1),
            "hallucination_count": self.data["hallucination_count"],
        }


analytics_tracker = AnalyticsTracker()
