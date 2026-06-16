from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_judging.complexity import (
    ComplexityTier,
    classify_tier,
    extract_features,
    score_complexity,
)


class ComplexityFeatureExtractionTests(unittest.TestCase):
    def test_simple_query(self) -> None:
        text = "What is Python?"
        features = extract_features(text)
        score = score_complexity(features)
        tier = classify_tier(score)

        self.assertEqual(ComplexityTier.SIMPLE, tier)

    def test_code_heavy_query(self) -> None:
        text = "Write a function that implements a class hierarchy with async/await for database calls"
        features = extract_features(text)
        score = score_complexity(features)
        tier = classify_tier(score)

        self.assertIn(tier, [ComplexityTier.MEDIUM, ComplexityTier.COMPLEX, ComplexityTier.REASONING])

    def test_reasoning_query(self) -> None:
        text = "Step by step, let's think through the pros and cons of this architecture"
        features = extract_features(text)
        score = score_complexity(features)
        tier = classify_tier(score)

        self.assertIn(tier, [ComplexityTier.MEDIUM, ComplexityTier.COMPLEX, ComplexityTier.REASONING])

    def test_empty_text(self) -> None:
        features = extract_features("")
        score = score_complexity(features)

        self.assertEqual(0.0, score)

    def test_multiline_numbered_list(self) -> None:
        text = "1. First step\n2. Second step\n3. Third step"
        features = extract_features(text)
        score = score_complexity(features)
        tier = classify_tier(score)

        # Simple numbered list with generic words stays simple
        self.assertEqual(ComplexityTier.SIMPLE, tier)


if __name__ == "__main__":
    unittest.main()
