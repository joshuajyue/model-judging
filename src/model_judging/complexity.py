from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ComplexityTier(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    REASONING = "reasoning"


@dataclass(slots=True)
class ComplexityFeatures:
    token_count: int
    code_keywords: float
    reasoning_markers: float
    technical_terms: float
    simple_indicators: float
    multi_step_patterns: float
    question_count: float


# Pre-compiled regex patterns
_MULTI_STEP_PATTERNS = [
    re.compile(r"first\s+.*?\s+then", re.IGNORECASE),
    re.compile(r"step\s*\d", re.IGNORECASE),
    re.compile(r"^\d+\.", re.MULTILINE),
    re.compile(r"^[a-z]\)", re.MULTILINE),
]

_CODE_KEYWORDS = {
    "function",
    "class",
    "def",
    "import",
    "async",
    "await",
    "sql",
    "api",
    "endpoint",
    "docker",
    "git",
    "refactor",
    "optimize",
    "bug",
    "error",
    "debug",
    "compile",
    "execute",
}

_REASONING_KEYWORDS = {
    "step by step",
    "think through",
    "let's think",
    "chain of thought",
    "analyze",
    "pros and cons",
    "explain your reasoning",
    "show your work",
    "deduce",
    "reason about",
}

_TECHNICAL_KEYWORDS = {
    "architecture",
    "distributed",
    "scalable",
    "microservice",
    "machine learning",
    "neural network",
    "encryption",
    "authentication",
    "performance",
    "concurrency",
    "database",
    "cache",
    "algorithm",
}

_SIMPLE_KEYWORDS = {
    "what is",
    "what's",
    "define",
    "who is",
    "when did",
    "how many",
    "hello",
    "hi",
    "thanks",
    "thank you",
}


def extract_features(text: str) -> ComplexityFeatures:
    """Extract complexity features from text."""
    if not text:
        return ComplexityFeatures(
            token_count=0,
            code_keywords=0.0,
            reasoning_markers=0.0,
            technical_terms=0.0,
            simple_indicators=0.0,
            multi_step_patterns=0.0,
            question_count=0.0,
        )

    text_lower = text.lower()
    token_count = len(text) // 4

    code_score = _score_keyword_set(text_lower, _CODE_KEYWORDS)
    reasoning_score = _score_keyword_set(text_lower, _REASONING_KEYWORDS)
    technical_score = _score_keyword_set(text_lower, _TECHNICAL_KEYWORDS)
    simple_score = _score_keyword_set(text_lower, _SIMPLE_KEYWORDS)
    multi_step_score = _score_multi_step(text)
    question_count = text.count("?")

    return ComplexityFeatures(
        token_count=token_count,
        code_keywords=code_score,
        reasoning_markers=reasoning_score,
        technical_terms=technical_score,
        simple_indicators=simple_score,
        multi_step_patterns=multi_step_score,
        question_count=min(1.0, question_count / 3.0),
    )


def score_complexity(features: ComplexityFeatures) -> float:
    """Compute weighted complexity score from features."""
    score = 0.0
    score += features.code_keywords * 0.30
    score += features.reasoning_markers * 0.25
    score += features.technical_terms * 0.25
    score += features.multi_step_patterns * 0.10
    score += features.question_count * 0.05
    score -= features.simple_indicators * 0.15
    return max(0.0, min(1.0, score))


def classify_tier(complexity_score: float) -> ComplexityTier:
    """Map complexity score to tier."""
    if complexity_score < 0.15:
        return ComplexityTier.SIMPLE
    if complexity_score < 0.35:
        return ComplexityTier.MEDIUM
    if complexity_score < 0.60:
        return ComplexityTier.COMPLEX
    return ComplexityTier.REASONING


def _score_keyword_set(text: str, keywords: set[str]) -> float:
    """Score presence of keyword set in text (0.0-1.0)."""
    if not keywords:
        return 0.0
    matches = sum(1 for kw in keywords if kw in text)
    if matches == 0:
        return 0.0
    # Any 3+ matches maxes out the score
    return min(1.0, matches / 3.0)


def _score_multi_step(text: str) -> float:
    """Score multi-step patterns in text."""
    matches = sum(1 for pattern in _MULTI_STEP_PATTERNS if pattern.search(text))
    return min(1.0, matches / len(_MULTI_STEP_PATTERNS))
