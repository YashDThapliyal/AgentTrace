"""Tests for retrieval.rank()."""
import math

import pytest

from agenttrace.retrieval import rank


class TestEmptyAndEdgeCases:
    def test_empty_stored_embeddings(self):
        assert rank([1.0, 0.0], [], top_k=3, threshold=0.75) == []

    def test_top_k_zero(self):
        stored = [("id1", [1.0, 0.0])]
        assert rank([1.0, 0.0], stored, top_k=0, threshold=0.0) == []

    def test_all_below_threshold(self):
        stored = [("id1", [0.0, 1.0])]  # orthogonal to query → similarity 0
        assert rank([1.0, 0.0], stored, top_k=3, threshold=0.75) == []

    def test_zero_vector_handled(self):
        stored = [("id1", [0.0, 0.0])]
        result = rank([1.0, 0.0], stored, top_k=3, threshold=0.0)
        assert result[0][1] == pytest.approx(0.0)

    def test_zero_query_vector_handled(self):
        stored = [("id1", [1.0, 0.0])]
        result = rank([0.0, 0.0], stored, top_k=3, threshold=0.0)
        assert result[0][1] == pytest.approx(0.0)


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        v = [1.0, 2.0, 3.0]
        result = rank(v, [("id1", v)], top_k=1, threshold=0.0)
        assert result[0][1] == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        result = rank([1.0, 0.0], [("id1", [0.0, 1.0])], top_k=1, threshold=0.0)
        assert result[0][1] == pytest.approx(0.0)

    def test_opposite_vectors_score_negative_one(self):
        result = rank([1.0, 0.0], [("id1", [-1.0, 0.0])], top_k=1, threshold=-1.0)
        assert result[0][1] == pytest.approx(-1.0)

    def test_known_similarity(self):
        # cos([1,1], [1,0]) = 1/sqrt(2) ≈ 0.7071
        result = rank([1.0, 1.0], [("id1", [1.0, 0.0])], top_k=1, threshold=0.0)
        assert result[0][1] == pytest.approx(1.0 / math.sqrt(2))


class TestTopK:
    def test_limits_results_to_top_k(self):
        stored = [
            ("id1", [1.0, 0.0]),
            ("id2", [0.9, 0.1]),
            ("id3", [0.8, 0.2]),
            ("id4", [0.1, 0.9]),
        ]
        result = rank([1.0, 0.0], stored, top_k=2, threshold=0.0)
        assert len(result) == 2

    def test_results_sorted_descending(self):
        stored = [
            ("low", [0.0, 1.0]),   # low similarity
            ("high", [1.0, 0.0]),  # high similarity
            ("mid", [0.7, 0.7]),   # mid similarity
        ]
        result = rank([1.0, 0.0], stored, top_k=3, threshold=0.0)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_larger_than_results(self):
        stored = [("id1", [1.0, 0.0]), ("id2", [0.9, 0.1])]
        result = rank([1.0, 0.0], stored, top_k=10, threshold=0.0)
        assert len(result) == 2


class TestThreshold:
    def test_threshold_filters_correctly(self):
        stored = [
            ("above", [1.0, 0.0]),   # similarity 1.0 → passes 0.75
            ("below", [0.0, 1.0]),   # similarity 0.0 → fails 0.75
        ]
        result = rank([1.0, 0.0], stored, top_k=5, threshold=0.75)
        ids = [id_ for id_, _ in result]
        assert "above" in ids
        assert "below" not in ids

    def test_threshold_zero_passes_all(self):
        stored = [("id1", [1.0, 0.0]), ("id2", [0.0, 1.0])]
        result = rank([1.0, 0.0], stored, top_k=5, threshold=0.0)
        assert len(result) == 2

    def test_threshold_one_passes_only_identical(self):
        v = [1.0, 0.0]
        stored = [
            ("exact", [1.0, 0.0]),
            ("close", [0.99, 0.01]),
        ]
        result = rank(v, stored, top_k=5, threshold=1.0)
        ids = [id_ for id_, _ in result]
        assert "exact" in ids
        # "close" has similarity < 1.0 so should be excluded
        assert "close" not in ids

    def test_result_ids_match_correct_traces(self):
        stored = [
            ("match", [1.0, 0.0]),
            ("no_match", [0.0, 1.0]),
        ]
        result = rank([1.0, 0.0], stored, top_k=3, threshold=0.5)
        assert result[0][0] == "match"
