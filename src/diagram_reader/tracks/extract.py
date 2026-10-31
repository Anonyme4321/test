"""Image de plan → masque des voies → squelette → graphe."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from diagram_reader.tracks.graph import TrackGraph, bridge_gaps, prune, skeleton_to_graph


@dataclass
class ExtractConfig:
    # binarisation
    adaptive: bool = True  # seuil adaptatif (scans inégaux) plutôt qu'Otsu global
    block_size: int = 51
    gap_close_px: int = 1  # fermeture des interstices ≤ 2·n px (voies en double trait)
    # épaisseur : seuil sur la demi-largeur (px). None = estimation automatique
    half_width_px: float | None = None
    thick_ratio: float = 1.8  # repli auto : voie = thick_ratio × demi-largeur médiane des traits
    auto_min_extent_factor: float = 40.0  # composante « étendue » : diagonale ≥ facteur × demi-largeur
    auto_cover_frac: float = 0.4  # … et leur étendue cumulée ≥ fraction de la largeur d'image
    auto_family_jump: float = 1.5  # une famille plus fine est retenue si elle accroît la couverture d'autant
    auto_margin: float = 0.85  # seuil = 85 % de la demi-largeur de la famille la plus fine retenue
    # nettoyage du masque
    min_component_len: float = 40.0  # composantes dont le squelette est plus court (px) : ignorées
    max_width_ratio: float = 6.0  # aplats : demi-largeur > ratio × celle des voies, retirés (une convergence de voies reste bien en dessous)
    close_px: int = 3  # fermeture pour recoller les petites coupures
    # graphe : seuils exprimés en mm « papier », convertis en px via l'épaisseur
    # mesurée des voies (track_width_mm sur le plan d'origine). Indépendant de la
    # résolution du scan et du format (rouleau ou feuille).
    track_width_mm: float = 0.7
    spur_mm: float = 10.0  # barbules plus courtes : élaguées
    merge_mm: float = 3.0  # nœuds plus proches : fusionnés
    extent_mm: float = 30.0  # composantes moins étendues : supprimées
    gap_mm: float = 25.0  # coupures plus courtes entre deux bouts alignés : pontées
    gap_angle_deg: float = 20.0
    epsilon: float = 1.5


@dataclass
class ExtractResult:
    graph: TrackGraph
    ink: np.ndarray  # masque encre (uint8 0/1)
    track_mask: np.ndarray  # masque voies (uint8 0/1)
    skeleton: np.ndarray  # squelette (bool)
    half_width: float  # demi-largeur retenue pour les voies (px)


# ------------------------------------------------------------------- binarisation
def binarize(gray: np.ndarray, cfg: ExtractConfig) -> np.ndarray:
    """Renvoie un masque uint8 où 1 = encre."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if cfg.adaptive:
        bs = cfg.block_size | 1
        ink = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, bs, 15)
        # le seuil adaptatif creuse l'intérieur des gros aplats : on le complète
        # par Otsu, qui lui les garde pleins
        _, otsu = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if otsu.mean() < 0.3:  # sinon le fond est sombre/sale : Otsu n'est pas fiable
            ink = ink | otsu
    else:
        _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = ink.astype(np.uint8)
    if cfg.gap_close_px > 0:
        # certaines voies sont dessinées en double trait séparé d'un filet blanc :
        # on fond les interstices d'un ou deux pixels pour les mesurer comme un seul trait
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * cfg.gap_close_px + 1,) * 2)
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k)
    return ink


# ---------------------------------------------------------------- traits épais
def estimate_half_widths(ink: np.ndarray, max_pixels: int = 30_000_000, seed: int = 0) -> np.ndarray:
    """Demi-largeur locale (distance au fond) mesurée sur le squelette de l'encre.

    Sur les très grandes images, mesurée sur des dalles tirées au hasard (assez
    pour être représentatives, sans squelettiser 200 Mpx).
    """
    h, w = ink.shape
    if h * w <= max_pixels:
        dist = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
        return dist[skeletonize(ink.astype(bool))]
    rng = np.random.default_rng(seed)
    tile = 2048
    n_tiles = max(4, int(max_pixels / (tile * tile)))
    out = []
    for _ in range(n_tiles):
        y = int(rng.integers(0, max(1, h - tile)))
        x = int(rng.integers(0, max(1, w - tile)))
        sub = np.ascontiguousarray(ink[y:y + tile, x:x + tile])
        if sub.sum() < 100:
            continue
        dist = cv2.distanceTransform(sub, cv2.DIST_L2, 5)
        out.append(dist[skeletonize(sub.astype(bool))])
    return np.concatenate(out) if out else np.array([], np.float32)


def auto_half_width(ink: np.ndarray, dist: np.ndarray, cfg: ExtractConfig) -> float:
    """Seuil de demi-largeur séparant les voies des autres traits.

    Les voies sont les seuls traits épais qui *traversent* le plan. On descend
    le seuil depuis les plus grandes épaisseurs et on retient le plus haut pour
    lequel les composantes épaisses et étendues couvrent une bonne part de la
    largeur de l'image. Les textes et boîtes de symboles, même épais, sont
    courts : ils ne passent pas ce critère.
    """
    hws = estimate_half_widths(ink)
    if hws.size == 0:
        return 1.0
    base = float(np.median(hws))  # demi-largeur des traits fins
    lo = base * 1.3
    hi = float(np.percentile(hws, 99.5))
    w = ink.shape[1]
    best = None
    best_cov = 0.0
    t = hi
    while t >= lo:
        core = (dist >= t).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
        del labels, core
        cov = 0.0
        if n > 1:
            ext = np.hypot(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT])
            cov = float(ext[ext >= cfg.auto_min_extent_factor * t].sum() / w)
        if best is None:
            if cov >= cfg.auto_cover_frac:
                best, best_cov = t, cov  # famille la plus épaisse qui traverse le plan
        elif cov >= cfg.auto_family_jump * best_cov:
            best, best_cov = t, cov  # une famille de voies plus fines apparaît : on l'inclut
        t -= 0.25
    if best is None:
        return max(base * cfg.thick_ratio, base + 1.0)
    return max(lo, best * cfg.auto_margin)


def thick_mask(ink: np.ndarray, cfg: ExtractConfig) -> tuple[np.ndarray, float]:
    """Isole les traits dont la demi-largeur dépasse le seuil : les voies."""
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    if cfg.half_width_px is not None:
        hw = float(cfg.half_width_px)
    else:
        hw = auto_half_width(ink, dist, cfg)
    core = (dist >= hw).astype(np.uint8)
    # reconstruction géodésique : on regagne toute l'épaisseur du trait à partir
    # de son cœur, sans déborder dans les traits fins qui le touchent
    allowed = (dist >= hw * 0.55).astype(np.uint8) & ink
    mask = core.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # quelques itérations suffisent pour regagner les bords du trait ; on ne
    # veut surtout pas « voyager » le long d'un trait moyen qui touche la voie
    for _ in range(int(np.ceil(hw)) + 2):
        grown = cv2.dilate(mask, kernel) & allowed
        if np.array_equal(grown, mask):
            break
        mask = grown
    del core, allowed
    # aplats (points pleins, carrés noirs, hachures denses) : zones bien plus
    # larges qu'une voie, retirées localement — jamais par composante entière,
    # car une voie est presque toujours connectée à quelque chose de plein
    blob_hw = cfg.max_width_ratio * hw
    blob = (dist >= blob_hw).astype(np.uint8)
    if blob.any():
        for _ in range(int(np.ceil(blob_hw)) + 2):
            grown = cv2.dilate(blob, kernel) & mask
            if np.array_equal(grown, blob):
                break
            blob = grown
        mask &= 1 - blob
    del dist, blob
    if cfg.close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_px * 2 + 1,) * 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    # composantes trop courtes (restes de symboles, lettres) : ignorées
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n:
        skel = skeletonize(mask.astype(bool))
        lengths = ndi.sum(skel, lab, index=range(1, n + 1))
        keep = np.concatenate([[False], lengths >= cfg.min_component_len])
        mask = keep[lab].astype(np.uint8)
        del lab, skel
    return mask, hw


# ------------------------------------------------------------------- pipeline
def extract_tracks(image: np.ndarray, cfg: ExtractConfig | None = None) -> ExtractResult:
    """Extrait le graphe de voies d'une image (niveaux de gris ou BGR)."""
    cfg = cfg or ExtractConfig()
    ink = binarize(image, cfg)
    mask, hw = thick_mask(ink, cfg)
    skel = skeletonize(mask.astype(bool))
    g = skeleton_to_graph(skel)
    px_per_mm = (2 * hw) / cfg.track_width_mm
    g = prune(
        g,
        min_spur=cfg.spur_mm * px_per_mm,
        merge_dist=cfg.merge_mm * px_per_mm,
        epsilon=cfg.epsilon,
        min_extent=cfg.extent_mm * px_per_mm,
    )
    if cfg.gap_mm > 0:
        bridge_gaps(g, max_gap=cfg.gap_mm * px_per_mm, max_angle_deg=cfg.gap_angle_deg)
    return ExtractResult(g, ink, mask, skel, hw)
