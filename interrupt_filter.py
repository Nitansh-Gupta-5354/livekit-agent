# interrupt_filter.py
from typing import List
import os
import json

class InterruptFilter:
    def __init__(self, ignored_words: List[str] = None, confidence_threshold: float = 0.6):
        # 1) try env var, else use default list
        if ignored_words is None:
            env_val = os.getenv("IGNORED_WORDS")
            if env_val:
                ignored_words = json.loads(env_val)
            else:
                ignored_words = ["uh", "umm", "hmm", "haan"]

        self.ignored_words = set(w.lower() for w in ignored_words)
        self.confidence_threshold = confidence_threshold
        self.priority_keywords = {"stop", "wait", "no", "cancel"}

    async def is_meaningful(self, transcript: str, confidence: float, agent_speaking: bool) -> bool:
        if not transcript:
            return False

        text = transcript.lower().strip()

        # if agent is silent → accept everything
        if not agent_speaking:
            return True

        # agent is speaking → be stricter
        if confidence is not None and confidence < self.confidence_threshold:
            return False

        tokens = text.split()
        if tokens and all(tok in self.ignored_words for tok in tokens):
            return False

        if any(k in text for k in self.priority_keywords):
            return True

        return True

    def update_ignored_words(self, new_list: List[str]):
        self.ignored_words = set(w.lower() for w in new_list)
