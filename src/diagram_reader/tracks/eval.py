"""Évaluation de l'extraction de voies contre un masque de vérité terrain.

    python -m diagram_reader.tracks.eval data/truth --tol 6 [--only Bonneville] [--overlay out/]

Pour chaque <nom>.png ayant un <nom>.tracks.png :

- précision / rappel du **squelette** : fraction des pixels du squelette prédit à
  moins de `tol` px du squelette vrai, et réciproquement ;
- rappel / précision des **jonctions** (aiguilles) : nœuds de degré ≥ 3, appariés
  à `tol_junction` px ;
- longueur totale prédite vs vraie.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

from diagram_reader.tracks.extract import ExtractConfig, extract_tracks
from diagram_reader.tracks.graph import prune, skeleton_to_graph


def _skeleton_pr(pred: np.ndarray, truth: np.ndarray, tol: float) -> tuple[float, float]:
    """(précision, rappel) entre deux squelettes binaires avec tolérance en px."""
    if not pred.any() or not truth.any():
        return 0.0, 0.0
    d_truth = cv2.distanceTransform((~truth).astype(np.uint8), cv2.DIST_L2, 5)
    d_pred = cv2.distanceTransform((~pred).astype(np.uint8), cv2.DIST_L2, 5)
    precision = float((d_truth[pred] <= tol).mean())
    recall = float((d_pred[truth] <= tol).mean())
    return precision, recall


def _drop_small(mask: np.ndarray, min_extent: float) -> np.ndarray:
    """Retire les composantes dont la diagonale de boîte < min_extent (points, lettres…)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        keep[i] = np.hypot(w, h) >= min_extent
    return keep[lab]


def _match_points(a: list[tuple[float, float]], b: list[tuple[float, float]], tol: float) -> int:
    """Nombre de points de `a` ayant un point de `b` à moins de tol (appariement glouton)."""
    if not a or not b:
        return 0
    b_arr = np.array(b)
    used = np.zeros(len(b), bool)
    n = 0
    for x, y in a:
        d = np.hypot(b_arr[:, 0] - x, b_arr[:, 1] - y)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True
            n += 1
    return n


def evaluate_one(image: np.ndarray, truth_mask: np.ndarray, cfg: ExtractConfig,
                 tol: float, tol_junction: float, overlay: Path | None = None) -> dict:
    t0 = time.time()
    res = extract_tracks(image, cfg)
    dt = time.time() - t0
    g, half_width = res.graph, res.half_width
    del res
    if overlay is not None:
        ov = g.draw(image)
        ov[truth_mask > 0] = (0.5 * ov[truth_mask > 0] + 0.5 * np.array([255, 200, 120])).astype(np.uint8)
        cv2.imwrite(str(overlay), ov)
        del ov
    # squelette prédit : redessiné depuis le graphe élagué (ce qui sera utilisé en aval)
    pred = np.zeros(truth_mask.shape, np.uint8)
    for e in g.edges:
        cv2.polylines(pred, [np.array(e.points, np.int32).reshape(-1, 1, 2)], False, 1, 1)
    truth_skel = skeletonize(_drop_small(truth_mask > 0, min_extent=0.012 * truth_mask.shape[1]))
    p, r = _skeleton_pr(pred.astype(bool), truth_skel, tol)
    # jonctions vraies : même graphe, construit sur le masque vrai
    tg = prune(skeleton_to_graph(truth_skel), min_spur=tol_junction, merge_dist=tol_junction,
               min_extent=tol_junction * 2)
    tj = [(n.x, n.y) for n in tg.junctions()]
    pj = [(n.x, n.y) for n in g.junctions()]
    hit = _match_points(tj, pj, tol_junction)
    return {
        "half_width": round(half_width, 2),
        "skel_precision": round(p, 4),
        "skel_recall": round(r, 4),
        "junctions_true": len(tj),
        "junctions_pred": len(pj),
        "junction_recall": round(hit / len(tj), 3) if tj else None,
        "junction_precision": round(hit / len(pj), 3) if pj else None,
        "length_pred": round(g.total_length()),
        "length_true": round(tg.total_length()),
        "seconds": round(dt, 1),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Évalue l'extraction de voies.")
    ap.add_argument("truth_dir", type=Path)
    ap.add_argument("--tol", type=float, default=6.0)
    ap.add_argument("--tol-junction", type=float, default=40.0)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--overlay", type=Path, help="dossier où écrire les superpositions (réduites)")
    ap.add_argument("--json", type=Path, help="fichier de résultats")
    args = ap.parse_args(argv)

    results = {}
    for png in sorted(args.truth_dir.glob("*.png")):
        if png.name.endswith(".tracks.png"):
            continue
        if args.only and png.stem not in args.only:
            continue
        tpath = png.with_suffix(".tracks.png")
        if not tpath.exists():
            continue
        img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        truth = cv2.imread(str(tpath), cv2.IMREAD_GRAYSCALE)
        cfg = ExtractConfig()
        ov_path = None
        if args.overlay:
            args.overlay.mkdir(parents=True, exist_ok=True)
            ov_path = args.overlay / f"{png.stem}.overlay.png"
        r = evaluate_one(img, truth, cfg, args.tol, args.tol_junction, overlay=ov_path)
        results[png.stem] = r
        print(f"{png.stem:14s} P={r['skel_precision']:.3f} R={r['skel_recall']:.3f} "
              f"jonctions {r['junctions_pred']}/{r['junctions_true']} "
              f"(rappel {r['junction_recall']}, précision {r['junction_precision']}) "
              f"hw={r['half_width']} {r['seconds']}s")
    if results:
        ps = [r["skel_precision"] for r in results.values()]
        rs = [r["skel_recall"] for r in results.values()]
        print(f"\nmoyenne : P={np.mean(ps):.3f} R={np.mean(rs):.3f} sur {len(results)} plans")
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
