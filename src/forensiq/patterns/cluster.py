"""Failure family mining: hand-rolled TF-IDF + agglomerative clustering.

No sklearn, on purpose:
  * TfidfVectorizer — vocabulary, smoothed idf, L2-normalized rows.
  * Agglomerative average-linkage clustering with a cosine-distance threshold
    (numpy), with a seeded k-means fallback when the dendrogram over-splits.
Evidence text is built from stage + attr KEYS + categorical values; numeric
values are quantized to a placeholder so clustering groups failure MECHANISMS
rather than individual magnitudes.
"""

from __future__ import annotations

import re

import numpy as np
from pydantic import BaseModel, Field

from forensiq.ingest.schema import Trace
from forensiq.taxonomy.classifier import FailureRecord

_TOKEN_RE = re.compile(r"[a-z_][a-z0-9_]{2,}")
_NUM_RE = re.compile(r"-?\d+\.\d+|-?\d+")


def evidence_text(failure: FailureRecord, trace: Trace | None = None) -> str:
    """Build the clustering text for one failure.

    Stage + evidence attr keys + categorical values; numeric values collapse
    to '<n>' so scores don't fragment families. The taxonomy id itself is
    deliberately EXCLUDED — clustering must rediscover the families from
    evidence shape alone (that's what makes the purity test meaningful).
    ``trace`` is accepted for interface stability; only rule evidence is used
    because span-attr key dumps are shared across families and dilute the
    signal.
    """
    parts = [f"stage={failure.stage}"]
    seen_keys: set[str] = set()
    for ev in failure.evidence:
        if ev.key in seen_keys:
            continue
        seen_keys.add(ev.key)
        # Free-text values (query/output/context) leak TOPIC vocabulary that
        # splits one failure family per query topic; mask them to constants.
        if ev.key in ("query", "output", "context"):
            value = f"<{ev.key}_text>"
        else:
            value = _NUM_RE.sub("<n>", str(ev.value))[:60]
        note = _NUM_RE.sub("<n>", ev.note)[:60] if ev.note else ""
        parts.append(f"{ev.key}={value}")
        if note:
            parts.append(note)
    return " ".join(parts)


class TfidfVectorizer:
    """Minimal TF-IDF with smoothed idf and L2 row normalization."""

    def __init__(self, min_df: int = 1, lowercase: bool = True) -> None:
        self.min_df = min_df
        self.lowercase = lowercase
        self.vocab_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None

    def _tokenize(self, text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower() if self.lowercase else text)

    def fit(self, texts: list[str]) -> TfidfVectorizer:
        df: dict[str, int] = {}
        for text in texts:
            for tok in set(self._tokenize(text)):
                df[tok] = df.get(tok, 0) + 1
        self.vocab_ = {
            tok: i
            for i, tok in enumerate(sorted(t for t, c in df.items() if c >= self.min_df))
        }
        n = len(texts)
        self.idf_ = np.array(
            [np.log((1.0 + n) / (1.0 + df[tok])) + 1.0 for tok in self.vocab_],
            dtype=np.float64,
        )
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.idf_ is None:
            raise RuntimeError("fit before transform")
        X = np.zeros((len(texts), len(self.vocab_)), dtype=np.float64)
        for row, text in enumerate(texts):
            for tok in self._tokenize(text):
                col = self.vocab_.get(tok)
                if col is not None:
                    X[row, col] += 1.0
        X *= self.idf_  # tf * idf (tf = raw count)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return X / norms

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)


def cosine_distance_matrix(X: np.ndarray) -> np.ndarray:
    sim = np.clip(X @ X.T, 0.0, 1.0)
    return 1.0 - sim


def agglomerative(dist: np.ndarray, threshold: float) -> np.ndarray:
    """Average-linkage agglomerative clustering with a distance cutoff.

    Maintains cluster-level cross-sums S(i, c) = sum of pairwise distances
    between members of i and c; the average linkage is S(i, c) / (|i|*|c|).
    Merging j into i updates one row/column in O(n), so a full run is ~O(n^2)
    numpy work — fast enough to cluster a whole day of failures in the CLI.
    """
    n = dist.shape[0]
    if n == 0:
        return np.empty(0, dtype=int)
    S = dist.astype(np.float64).copy()
    np.fill_diagonal(S, 0.0)
    sizes = np.ones(n, dtype=np.float64)
    alive = np.ones(n, dtype=bool)
    labels = np.arange(n)

    while alive.sum() > 1:
        idx = np.where(alive)[0]
        sub = S[np.ix_(idx, idx)] / np.outer(sizes[idx], sizes[idx])
        np.fill_diagonal(sub, np.inf)
        p = np.unravel_index(np.argmin(sub), sub.shape)
        i, j = int(idx[p[0]]), int(idx[p[1]])
        if sub[p] > threshold:
            break
        # merge j into i: cross-sums add, sizes add, labels re-point
        S[i, :] += S[j, :]
        S[:, i] += S[:, j]
        S[i, i] = 0.0
        sizes[i] += sizes[j]
        labels[labels == j] = i
        alive[j] = False

    # compact labels
    uniq = sorted(set(labels.tolist()))
    remap = {old: new for new, old in enumerate(uniq)}
    return np.array([remap[lab] for lab in labels], dtype=int)


def kmeans(X: np.ndarray, k: int, seed: int = 7, iters: int = 50) -> np.ndarray:
    """Seeded k-means fallback (cosine-ish: Euclidean on L2-normalized rows)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = min(k, n)
    centroids = X[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for c in range(k):
            members = X[labels == c]
            if len(members):
                centroids[c] = members.mean(axis=0)
            else:
                centroids[c] = X[rng.integers(n)]
    # compact
    uniq = sorted(set(labels.tolist()))
    remap = {old: new for new, old in enumerate(uniq)}
    return np.array([remap[lab] for lab in labels], dtype=int)


class ClusterCard(BaseModel):
    cluster_id: int
    size: int
    share: float
    top_terms: list[str]
    dominant_taxonomy: str | None
    taxonomy_purity: float
    representative_trace_id: str
    member_trace_ids: list[str] = Field(default_factory=list)


class FailureClusterer:
    def __init__(
        self,
        threshold: float = 0.62,
        max_clusters: int = 10,
        min_cluster_size: int = 2,
        seed: int = 7,
    ) -> None:
        self.threshold = threshold
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size
        self.seed = seed

    def fit(self, traces: list[Trace], failures: list[FailureRecord]) -> list[ClusterCard]:
        if not failures:
            return []
        trace_by_id = {t.trace_id: t for t in traces}
        texts = [evidence_text(f, trace_by_id.get(f.trace_id)) for f in failures]
        vec = TfidfVectorizer(min_df=1)
        X = vec.fit_transform(texts)
        dist = cosine_distance_matrix(X)
        labels = agglomerative(dist, self.threshold)
        if len(set(labels.tolist())) > self.max_clusters:
            labels = kmeans(X, self.max_clusters, seed=self.seed)

        cards: list[ClusterCard] = []
        n = len(failures)
        vocab = list(vec.vocab_.keys())  # insertion order == column order (sorted at fit)
        for cid in sorted(set(labels.tolist())):
            members = [i for i, lab in enumerate(labels) if lab == cid]
            if len(members) < self.min_cluster_size:
                continue
            tax_counts: dict[str, int] = {}
            for i in members:
                tax_counts[failures[i].taxonomy_id] = tax_counts.get(failures[i].taxonomy_id, 0) + 1
            dominant, dominant_n = sorted(tax_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            # top terms: highest mean tf-idf across members
            mean_vec = X[members].mean(axis=0)
            top_idx = np.argsort(-mean_vec)[:6]
            top_terms = [vocab[i] for i in top_idx if mean_vec[i] > 0][:5]
            member_ids = [failures[i].trace_id for i in members]
            cards.append(
                ClusterCard(
                    cluster_id=cid,
                    size=len(members),
                    share=round(len(members) / n, 4),
                    top_terms=top_terms,
                    dominant_taxonomy=dominant,
                    taxonomy_purity=round(dominant_n / len(members), 4),
                    representative_trace_id=member_ids[0],
                    member_trace_ids=member_ids[:12],
                )
            )
        cards.sort(key=lambda c: (-c.size, c.cluster_id))
        for new_id, card in enumerate(cards):
            card.cluster_id = new_id
        return cards


def cluster_distances(X: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Mean intra-cluster vs inter-cluster cosine distance (sanity metric)."""
    n = X.shape[0]
    dist = cosine_distance_matrix(X)
    intra: list[float] = []
    inter: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                intra.append(dist[i, j])
            else:
                inter.append(dist[i, j])
    return (
        float(np.mean(intra)) if intra else 0.0,
        float(np.mean(inter)) if inter else 0.0,
    )
