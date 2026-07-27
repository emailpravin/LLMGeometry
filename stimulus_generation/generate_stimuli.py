"""
Generates the geometric intruder task using the paper's exact 11 shapes
(Table S1): N_PER_SHAPE trials per shape, deviant type drawn at random each
time, canonical presentation (5 reference instances + 1 deviant, position
randomized, each instance independently rotated/scaled), saved as PNG
displays + a manifest.json with ground truth.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shapes import build_shapes, make_deviants

OUT_DIR = "trials"
N_PER_SHAPE = 20
ROTATIONS_DEG = [-25, -15, -5, 5, 15, 25]
SCALES = [0.875, 0.925, 0.975, 1.025, 1.075, 1.125]
N_POSITIONS = 6
CIRCLE_RADIUS = 3.2

RNG = np.random.default_rng(1907260544)  # fresh, unseen seed for this exact-replication run


def transform(pts, rotation_deg, scale, center):
    theta = np.radians(rotation_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    return (pts @ rot.T) * scale + center


def position_centers():
    centers = []
    for k in range(N_POSITIONS):
        angle = np.radians(90 + k * 360 / N_POSITIONS)
        centers.append(CIRCLE_RADIUS * np.array([np.cos(angle), np.sin(angle)]))
    return centers


def render_trial(instances, path):
    fig, ax = plt.subplots(figsize=(7, 7))
    centers = position_centers()
    for pos_idx, pts in enumerate(instances):
        cx, cy = centers[pos_idx]
        poly = np.vstack([pts, pts[0]])
        ax.plot(poly[:, 0], poly[:, 1], color="black", linewidth=2)
        ax.fill(poly[:, 0], poly[:, 1], color="0.85")
        ax.text(
            cx, cy - 1.35, str(pos_idx + 1),
            ha="center", va="center", fontsize=13, color="gray",
        )
    ax.set_xlim(-CIRCLE_RADIUS - 1.6, CIRCLE_RADIUS + 1.6)
    ax.set_ylim(-CIRCLE_RADIUS - 1.6, CIRCLE_RADIUS + 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    shapes = build_shapes()
    deviant_labels = ["lengthen", "shorten", "rotate_pos", "rotate_neg"]
    manifest = []
    trial_id = 0

    for shape_name, shape in shapes.items():
        ref_pts = shape["vertices"]
        deviants = make_deviants(ref_pts)

        for _ in range(N_PER_SHAPE):
            trial_id += 1
            deviant_type = RNG.choice(deviant_labels)
            dev_pts = deviants[deviant_type]

            intruder_position = int(RNG.integers(1, N_POSITIONS + 1))  # 1..6
            rotations = RNG.choice(ROTATIONS_DEG, size=N_POSITIONS, replace=False)
            scales = RNG.choice(SCALES, size=N_POSITIONS, replace=False)

            instances = []
            centers = position_centers()
            for pos in range(1, N_POSITIONS + 1):
                base = dev_pts if pos == intruder_position else ref_pts
                placed = transform(
                    base, rotations[pos - 1], scales[pos - 1], centers[pos - 1]
                )
                instances.append(placed)

            fname = f"trial_{trial_id:03d}_{shape_name}_{deviant_type}.png"
            render_trial(instances, os.path.join(OUT_DIR, fname))

            manifest.append({
                "trial_id": trial_id,
                "file": fname,
                "shape": shape_name,
                "deviant_type": deviant_type,
                "regularity_score": shape["regularity_score"],
                "correct_position": intruder_position,
            })

    with open("manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated {trial_id} trials in {OUT_DIR}/, manifest.json written.")


if __name__ == "__main__":
    main()
