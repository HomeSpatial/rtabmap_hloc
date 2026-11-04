"""Audit loop rotations against an image-only, short-range spanning forest.

The forest is a consistency reference, not an absolute pose estimate. Its
connected components and unverifiable loops must accompany any filter result.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class RotationEdge:
    source: int
    target: int
    rotation: np.ndarray  # R_target_from_source
    inliers: int


def audit_rotation_cycles(
    ordered_ids: list[int],
    edges: list[RotationEdge],
    *,
    temporal_window: int = 3,
    minimum_inliers: int = 30,
    base_tolerance_deg: float = 8,
    per_hop_tolerance_deg: float = 0.25,
    maximum_tolerance_deg: float = 25,
) -> dict:
    if len(set(ordered_ids)) != len(ordered_ids) or not ordered_ids:
        raise ValueError("Ordered unique image IDs required")
    rank = {fid: i for i, fid in enumerate(ordered_ids)}
    parent = {fid: fid for fid in ordered_ids}
    adjacency = {fid: [] for fid in ordered_ids}

    def find(fid):
        while parent[fid] != fid:
            parent[fid] = parent[parent[fid]]
            fid = parent[fid]
        return fid

    for edge in edges:
        r = np.asarray(edge.rotation)
        if edge.source not in rank or edge.target not in rank:
            raise ValueError("Edge endpoint absent from the ordered images")
        if r.shape != (3, 3) or not np.isfinite(r).all():
            raise ValueError("Finite 3x3 relative rotation required")
        if not np.allclose(r @ r.T, np.eye(3), atol=1e-5) or not np.isclose(
            np.linalg.det(r), 1, atol=1e-5
        ):
            raise ValueError("Relative rotation must be in SO(3)")
    local = sorted(
        (e for e in edges if e.inliers >= minimum_inliers
         and abs(rank[e.source] - rank[e.target]) <= temporal_window),
        key=lambda e: (
            abs(rank[e.source] - rank[e.target]), -e.inliers,
            rank[e.source], rank[e.target],
        ),
    )
    forest_edges = set()
    for edge in local:
        a, b = find(edge.source), find(edge.target)
        if a == b:
            continue
        parent[b] = a
        adjacency[edge.source].append((edge.target, edge.rotation))
        adjacency[edge.target].append((edge.source, edge.rotation.T))
        forest_edges.add(tuple(sorted((edge.source, edge.target))))
    rotations, components = {}, []
    for seed in ordered_ids:
        if seed in rotations:
            continue
        rotations[seed] = np.eye(3)
        component = []
        queue = deque([seed])
        while queue:
            source = queue.popleft()
            component.append(source)
            for target, rotation in adjacency[source]:
                if target in rotations:
                    continue
                rotations[target] = rotation @ rotations[source]
                queue.append(target)
        components.append(component)
    labels = {fid: i for i, c in enumerate(components) for fid in c}

    def hops(source, target):
        queue = deque([(source, 0)])
        seen = {source}
        while queue:
            current, count = queue.popleft()
            if current == target:
                return count
            for neighbor, _ in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, count + 1))
        raise RuntimeError("Forest component has no path")

    rows = []
    for edge in edges:
        key = tuple(sorted((edge.source, edge.target)))
        row = {"source": edge.source, "target": edge.target,
               "inliers": edge.inliers, "backbone": key in forest_edges}
        if labels[edge.source] != labels[edge.target]:
            row.update(verifiable=False, rejected=False,
                       reason="no_independent_temporal_path")
        else:
            count = hops(edge.source, edge.target)
            expected = rotations[edge.target] @ rotations[edge.source].T
            error = float(np.rad2deg(Rotation.from_matrix(
                edge.rotation @ expected.T
            ).magnitude()))
            threshold = min(maximum_tolerance_deg,
                            base_tolerance_deg + count * per_hop_tolerance_deg)
            rejected = key not in forest_edges and error > threshold
            row.update(verifiable=True, path_hops=count,
                       rotation_error_deg=error, tolerance_deg=threshold,
                       rejected=rejected,
                       reason="rotation_cycle_inconsistent" if rejected else "consistent")
        rows.append(row)
    return {"components": components, "backbone_pairs": sorted(forest_edges),
            "edges": rows, "rejected_count": sum(r["rejected"] for r in rows),
            "unverifiable_count": sum(not r["verifiable"] for r in rows)}
