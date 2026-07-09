"""
Lightweight in-memory request monitoring.

Honest caveat worth understanding: this data lives in RAM only. It resets
every time the service restarts or cold-starts (which happens often on
Render's free tier). Real production systems use a proper metrics stack
(Prometheus + Grafana, DataDog, CloudWatch, etc.) that persists data outside
the running process. This is a deliberately simple version that still proves
the concept: instrument your API so you know what it's actually doing,
which is exactly the "monitoring" a job posting like this expects you to
understand — just scoped down to fit a free-tier deployment.
"""

import time
from collections import Counter, deque
from threading import Lock

MAX_HISTORY = 200

_lock = Lock()
_request_history = deque(maxlen=MAX_HISTORY)
_class_counter = Counter()
_start_time = time.time()


def record_prediction(predicted_class: str, confidence: float, latency_ms: float) -> None:
    with _lock:
        _request_history.append({
            "timestamp": time.time(),
            "predicted_class": predicted_class,
            "confidence": confidence,
            "latency_ms": latency_ms,
        })
        _class_counter[predicted_class] += 1


def get_metrics() -> dict:
    with _lock:
        history = list(_request_history)
        class_distribution = dict(_class_counter)

    total_requests = len(history)
    avg_latency_ms = (
        sum(r["latency_ms"] for r in history) / total_requests if total_requests else 0.0
    )
    avg_confidence = (
        sum(r["confidence"] for r in history) / total_requests if total_requests else 0.0
    )

    return {
        "uptime_seconds": round(time.time() - _start_time, 1),
        "total_requests_since_start": total_requests,
        "average_latency_ms": round(avg_latency_ms, 2),
        "average_confidence": round(avg_confidence, 4),
        "class_distribution": class_distribution,
        "recent_requests": history[-20:],
    }