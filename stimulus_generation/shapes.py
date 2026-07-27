"""
The 11 reference quadrilaterals from Sablé-Meyer, Fagot, Caparos, van
Kerkoerle, Amalric & Dehaene (2021), "Sensitivity to geometric shape
regularity in humans and baboons: A putative signature of human
singularity," PNAS 118(16) -- SI Appendix, Table S1 (exact coordinates).

Bottom-left vertex is (0,0) for every shape; Table S1 gives topLeft,
topRight, bottomRight (bottomRight y is always 0). Vertex order used
throughout this file is [BL, BR, TR, TL] (counter-clockwise), so the
"bottom edge" is BL->BR -- matching the paper's deviant-generation rule,
which displaces the bottom-right vertex.

regularity_score for each shape is taken directly from Table S1's own
published "Number of properties" column (15, 19, 5, 7, 9, 5, 7, 1, 2, 1, 0)
-- i.e., the paper's own ground truth, rather than a re-derivation. An
attempted from-scratch reimplementation of their 22-bit symbolic vector
(6 equal-length side pairs, 6 parallel side pairs, 6 equal-angle pairs, 4
right-angle bits, tolerance theta=12.5%) is kept below as
`symbolic_property_vector` for reference, but it does NOT exactly
reproduce their published counts -- it disagrees not just by an additive
offset but in relative ranking for some shapes (e.g. it ranks
"right_hinge" as more regular than "iso_trapezoid", the opposite of the
published ordering), meaning some tolerance/edge-case detail of their
actual implementation isn't captured by the SI's prose description alone.
Using their published numbers directly avoids that guesswork.
"""
import numpy as np
from itertools import combinations

THETA = 0.125  # paper's tolerance parameter, as used for its figures/analyses
DEVIANT_FRACTION = 0.30  # displacement as a fraction of avg pairwise vertex distance

# Table S1: shape -> (topLeft, topRight, bottomRight); bottomLeft is always (0,0)
_TABLE_S1 = {
    "rectangle":      ((0, 1), (1.5, 1), (1.5, 0)),
    "square":         ((0, 1.26), (1.26, 1.26), (1.26, 0)),
    "iso_trapezoid":  ((0.365, 1.362), (1.109, 1.362), (1.5, 0)),
    "parallelogram":  ((-0.517, 0.896), (0.983, 0.896), (1.5, 0)),
    "rhombus":        ((-0.908, 0.931), (0.392, 0.931), (1.3, 0)),
    "kite":           ((0.766, 1.29), (1.77, 1.007), (1.5, 0)),
    "right_kite":     ((0.529, 1.404), (1.5, 1.038), (1.5, 0)),
    "hinge":          ((-0.248, 0.533), (0.98, 1.393), (1.5, 0)),
    "right_hinge":    ((-0.296, 0.634), (1.064, 1.268), (1.5, 0)),
    "trapezoid":      ((-0.227, 1.2), (0.724, 1.2), (1.5, 0)),
    "irregular":      ((-0.45, 1.058), (0.227, 1.24), (1.5, 0)),
}

# Published "number of properties" per shape (Table S1), used only to
# self-check our reimplementation of the symbolic model below.
_PUBLISHED_PROPERTY_COUNT = {
    "rectangle": 15, "square": 19, "iso_trapezoid": 5, "parallelogram": 7,
    "rhombus": 9, "kite": 5, "right_kite": 7, "hinge": 1, "right_hinge": 2,
    "trapezoid": 1, "irregular": 0,
}


def _vertices(shape_name):
    (tlx, tly), (trx, try_), (brx, bry) = _TABLE_S1[shape_name]
    bl = np.array([0.0, 0.0])
    br = np.array([brx, bry])
    tr = np.array([trx, try_])
    tl = np.array([tlx, tly])
    return np.array([bl, br, tr, tl])  # order: BL, BR, TR, TL


def _avg_pairwise_dist(pts):
    return np.mean([np.linalg.norm(pts[i] - pts[j]) for i, j in combinations(range(4), 2)])


def _angle_at(pts, i):
    prev, cur, nxt = pts[(i - 1) % 4], pts[i], pts[(i + 1) % 4]
    v1, v2 = prev - cur, nxt - cur
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))


def _side_vec(pts, i):
    return pts[(i + 1) % 4] - pts[i]


def symbolic_property_vector(pts):
    """22-bit vector: 6 equal-length, 6 parallel, 6 equal-angle, 4 right-angle."""
    angles = [_angle_at(pts, i) for i in range(4)]
    sides = [np.linalg.norm(_side_vec(pts, i)) for i in range(4)]
    dirs = [_side_vec(pts, i) for i in range(4)]

    bits = []
    for i, j in combinations(range(4), 2):  # 6 equal-length bits
        l1, l2 = max(sides[i], sides[j]), min(sides[i], sides[j])
        bits.append(1 if (l1 / l2 - 1) < THETA else 0)

    for i, j in combinations(range(4), 2):  # 6 parallel bits
        u, v = dirs[i], dirs[j]
        cos_a = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        ang = np.degrees(np.arccos(np.clip(abs(cos_a), -1, 1)))  # undirected, 0-90
        bits.append(1 if ang < THETA * 90 else 0)

    for i, j in combinations(range(4), 2):  # 6 equal-angle bits
        bits.append(1 if abs(angles[i] - angles[j]) < THETA * 90 else 0)

    for a in angles:  # 4 right-angle bits
        bits.append(1 if abs(a - 90) < THETA * 90 else 0)

    return np.array(bits, dtype=int)


def build_shapes():
    shapes = {}
    for name in _TABLE_S1:
        pts = _vertices(name)
        shapes[name] = {
            "vertices": pts,
            "property_vector_reimplementation": symbolic_property_vector(pts),  # reference only, see module docstring
            "regularity_score": _PUBLISHED_PROPERTY_COUNT[name],  # paper's own Table S1 value, ground truth
        }
    return dict(sorted(shapes.items(), key=lambda kv: -kv[1]["regularity_score"]))


def make_deviants(pts):
    """4 deviants of vertex 1 ('bottom-right'): 2 slide along the adjacent
    edge (lengthen/shorten), 2 rotate about the neighboring vertex (index 0,
    'bottom-left'), all at DEVIANT_FRACTION of the average pairwise vertex
    distance -- matching the paper's rule exactly."""
    v0, v1 = pts[0], pts[1]
    edge = v1 - v0
    r = np.linalg.norm(edge)
    unit = edge / r
    d = DEVIANT_FRACTION * _avg_pairwise_dist(pts)

    lengthen = v1 + unit * d
    shorten = v1 - unit * d

    delta = 2 * np.arcsin(min(d / (2 * r), 1.0))

    def rotate_about(origin, point, theta):
        c, s = np.cos(theta), np.sin(theta)
        rel = point - origin
        rot = np.array([[c, -s], [s, c]])
        return origin + rot @ rel

    rotate_pos = rotate_about(v0, v1, delta)
    rotate_neg = rotate_about(v0, v1, -delta)

    deviants = {}
    for label, newv1 in [
        ("lengthen", lengthen),
        ("shorten", shorten),
        ("rotate_pos", rotate_pos),
        ("rotate_neg", rotate_neg),
    ]:
        newpts = pts.copy()
        newpts[1] = newv1
        deviants[label] = newpts
    return deviants


if __name__ == "__main__":
    shapes = build_shapes()
    print(f"{'shape':16s} {'regularity_score (published)':>28s}")
    for name, s in shapes.items():
        print(f"{name:16s} {s['regularity_score']:28d}")
