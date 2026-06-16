from __future__ import annotations

import json
from typing import Optional

from .types import Candidate, PairwiseOutcome


class GitHubModelsJudge:
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key

    def compare(
        self,
        left: Candidate,
        right: Candidate,
        artifact: object | None,
    ) -> PairwiseOutcome:
        import os
        import urllib.request
        import json as json_module

        api_key = self.api_key or os.environ.get("GITHUB_TOKEN", "")
        if not api_key:
            raise ValueError("GITHUB_TOKEN not set and api_key not provided")

        prompt = f"""You are an impartial judge. Compare these two model outputs.

Left Output ({left.name}):
{artifact if artifact else "(No artifact provided)"}

Right Output ({right.name}):
{artifact if artifact else "(No artifact provided)"}

Which is better? Respond with ONLY: "LEFT" or "RIGHT" or "TIE". No explanation."""

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        body = json_module.dumps(
            {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 1,
                "top_p": 1,
                "max_tokens": 10,
            }
        ).encode("utf-8")

        try:
            request = urllib.request.Request(
                "https://api.github.com/models/chat/completions",
                data=body,
                headers=headers,
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json_module.loads(response.read().decode("utf-8"))

            choice = result.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "TIE").strip().upper()

            if content == "LEFT":
                winner_id = left.id
            elif content == "RIGHT":
                winner_id = right.id
            else:
                winner_id = None

            return PairwiseOutcome(
                left_id=left.id,
                right_id=right.id,
                winner_id=winner_id,
                judge_id=self.model_name,
                rationale=content,
            )
        except Exception as e:
            return PairwiseOutcome(
                left_id=left.id,
                right_id=right.id,
                winner_id=None,
                judge_id=self.model_name,
                rationale=f"Error: {str(e)}",
            )
