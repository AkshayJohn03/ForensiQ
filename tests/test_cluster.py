"""Pattern mining: hand-rolled TF-IDF, agglomerative clustering, failure
family recovery on the planted corpus, k-means fallback."""

from __future__ import annotations

import numpy as np
import pytest

from forensiq.patterns.cluster import (
    FailureClusterer,
    TfidfVectorizer,
    agglomerative,
    cluster_distances,
    cosine_distance_matrix,
    evidence_text,
    kmeans,
)


class TestTfidf:
    def test_fit_transform_shapes_and_normalization(self):
        texts = ["retrieve hit_count=0 empty", "generate json parse failed", "retrieve hit_count=0 empty"]
        vec = TfidfVectorizer().fit(texts)
        X = vec.transform(texts)
        assert X.shape == (3, len(vec.vocab_))
        norms = np.linalg.norm(X, axis=1)
        assert np.allclose(norms, 1.0)
        # identical rows -> identical vectors -> zero distance
        assert cosine_distance_matrix(X)[0, 2] == pytest.approx(0.0, abs=1e-9)

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            TfidfVectorizer().transform(["x"])

    def test_min_df_filters(self):
        X = TfidfVectorizer(min_df=2).fit_transform(["alpha beta", "alpha gamma"])
        assert X.shape[1] == 1  # only 'alpha' survives min_df=2


class TestAgglomerative:
    def test_two_clear_groups(self):
        # 4 points in two tight pairs, far apart (L2-normalized 2D-ish rows)
        X = np.array(
            [
                [1.0, 0.0],
                [0.99, 0.1],
                [0.0, 1.0],
                [0.1, 0.99],
            ]
        )
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        labels = agglomerative(cosine_distance_matrix(X), threshold=0.2)
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

    def test_high_threshold_yields_single_cluster(self):
        X = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]])
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        labels = agglomerative(cosine_distance_matrix(X), threshold=0.9)
        assert len(set(labels.tolist())) == 1


class TestKmeans:
    def test_seeded_and_separates(self):
        rng = np.random.default_rng(3)
        a = rng.normal(0, 0.05, size=(20, 4))
        b = rng.normal(5, 0.05, size=(20, 4))
        X = np.vstack([a, b])
        labels = kmeans(X, k=2, seed=7)
        assert len(set(labels[:20].tolist())) == 1
        assert len(set(labels[20:].tolist())) == 1
        assert labels[0] != labels[20]
        labels2 = kmeans(X, k=2, seed=7)
        assert (labels == labels2).all()


class TestFailureFamilyRecovery:
    def test_evidence_text_masks_topic_and_numbers(self, corpus, failures):
        failure = failures[0]
        text = evidence_text(failure)
        assert "stage=" in text
        assert failure.taxonomy_id not in text  # clustering must not peek
        for f in failures[:30]:
            assert "supplier" not in evidence_text(f).lower() or "Nordwerk" not in evidence_text(f)

    def test_planted_families_recovered(self, corpus, failures):
        """Clustering must rediscover the failure families from evidence shape
        alone: >= 7 clusters, each planted family's dominant cluster purity
        >= 0.9, and intra-cluster distance well below inter-cluster."""
        traces, planted = corpus
        cards = FailureClusterer().fit(traces, failures)
        assert len(cards) >= 7

        # rebuild the matrix for the distance sanity check
        texts = [evidence_text(f) for f in failures]
        X = TfidfVectorizer().fit_transform(texts)
        family_labels = np.zeros(len(failures), dtype=int)
        fam_index = {}
        for i, f in enumerate(failures):
            fam_index.setdefault(f.taxonomy_id, len(fam_index))
            family_labels[i] = fam_index[f.taxonomy_id]
        intra, inter = cluster_distances(X, family_labels)
        assert intra < inter
        assert inter - intra > 0.3

        # purity: every planted family maps to a >= 0.9-pure cluster
        cluster_by_card = {c.dominant_taxonomy: c.taxonomy_purity for c in cards}
        planted_families = {p.taxonomy_id for p in planted}
        for tax_id in planted_families:
            assert cluster_by_card.get(tax_id, 0) >= 0.9, tax_id

    def test_cluster_cards_have_terms_and_representative(self, corpus, failures):
        traces, _ = corpus
        cards = FailureClusterer().fit(traces, failures)
        for card in cards:
            assert card.size >= 2
            assert 0.0 < card.share <= 1.0
            assert card.top_terms
            assert card.representative_trace_id in {t.trace_id for t in traces}

    def test_empty_input(self):
        assert FailureClusterer().fit([], []) == []
