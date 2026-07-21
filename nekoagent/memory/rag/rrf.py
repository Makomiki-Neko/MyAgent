"""Reciprocal Rank Fusion (RRF) 凸显融合。

参数 k（默认 60）用于平滑各源 rank 影响：score(d) = sum(1 / (k + rank_i(d)))。
"""

from __future__ import annotations

from typing import Iterable


def reciprocal_rank_fusion(
    *ranked_lists: Iterable[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """输入任意条 ranked list，每条 [(doc_id, score)]。RRF 不关注分数，仅看 rank。"""

    buckets: dict[str, float] = {}
    for lst in ranked_lists:
        ranked = list(lst)
        for rank_idx, (doc_id, _) in enumerate(ranked):
            buckets[doc_id] = buckets.get(doc_id, 0.0) + 1.0 / (k + rank_idx + 1)
    fused = sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)
    return fused