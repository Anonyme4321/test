"""Génération d'un plan synthétique annoté.

Un plan = un cadre, 2 à 4 voies quasi parallèles légèrement inclinées, des
zones de détection délimitées par des joints, des signaux avec leur nom et
leurs indications, des repères kilométriques numérotés, et quelques
équipements ponctuels (téléphones, pédales, compteurs d'essieux, pétards,
aiguilles, flèches de sens).

La vérité terrain produite va au-delà des boîtes : pour chaque signal on
conserve son nom, sa voie et son PK, ce qui servira à évaluer l'étage
d'association en aval.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image, ImageFilter

from diagram_reader.synth.classes import CLASS_IDS
from diagram_reader.synth.fonts import load_font
from diagram_reader.synth.library import LIBRARY
from diagram_reader.synth.symbols import BBox, Style, SymbolDrawer

INDICATIONS = ["C", "S", "RR", "A", "VL", "M", "R", "D", "G"]


# --------------------------------------------------------------------------- données
@dataclass
class Annotation:
    cls: str
    bbox: BBox  # pixels absolus (x1, y1, x2, y2)


@dataclass
class TextAnnotation:
    text: str
    bbox: BBox
    role: str  # signal_name | indication | pk | zone | track_name | title | cm_name | fixed


@dataclass
class SignalTruth:
    name: str
    track: str
    pk_m: int  # point kilométrique en mètres
    indications: list[str]
    direction: int
    bbox: BBox
    name_bbox: BBox


@dataclass
class TrackTruth:
    name: str
    y_left: float
    y_right: float
    direction: int


@dataclass
class PlanTruth:
    width: int
    height: int
    tracks: list[TrackTruth]
    objects: list[Annotation]
    texts: list[TextAnnotation]
    signals: list[SignalTruth]
    pk_origin_m: int

    def to_dict(self) -> dict:
        return asdict(self)

    def yolo_lines(self) -> list[str]:
        out = []
        for a in self.objects:
            x1, y1, x2, y2 = a.bbox
            cx = (x1 + x2) / 2 / self.width
            cy = (y1 + y2) / 2 / self.height
            w = (x2 - x1) / self.width
            h = (y2 - y1) / self.height
            out.append(f"{CLASS_IDS[a.cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        return out


# ---------------------------------------------------------------------- config
@dataclass
class GenerationConfig:
    width: int = 2400
    height: int = 1600
    n_tracks: tuple[int, int] = (2, 4)
    track_spacing: float = 190.0
    max_slope_deg: float = 2.5
    scale: tuple[float, float] = (0.9, 1.4)
    signals_per_track: tuple[int, int] = (2, 5)
    zones_per_track: tuple[int, int] = (3, 6)
    pk_dots: tuple[int, int] = (4, 9)
    extra_equipment: tuple[int, int] = (4, 12)
    vertical_indications_p: float = 0.7
    noise: bool = True
    seed: int | None = None
    exclude: tuple[str, ...] = field(default_factory=tuple)  # classes à ne pas dessiner


# --------------------------------------------------------------------- helpers
class _Track:
    def __init__(self, name: str, y0: float, slope: float, x0: float, direction: int):
        self.name, self.y0, self.slope, self.x0, self.direction = name, y0, slope, x0, direction

    def y_at(self, x: float) -> float:
        return self.y0 + (x - self.x0) * self.slope


def _overlaps(box: BBox, others: list[BBox], margin: float = 4) -> bool:
    x1, y1, x2, y2 = box
    for ox1, oy1, ox2, oy2 in others:
        if x1 < ox2 + margin and x2 > ox1 - margin and y1 < oy2 + margin and y2 > oy1 - margin:
            return True
    return False


def _free_x(rng: random.Random, x_min: float, x_max: float, used: list[tuple[float, float]],
            width: float, tries: int = 40) -> float | None:
    """Tire une abscisse dont l'intervalle [x - w/2, x + w/2] ne chevauche pas `used`."""
    for _ in range(tries):
        x = rng.uniform(x_min + width / 2, x_max - width / 2)
        if all(x + width / 2 < a or x - width / 2 > b for a, b in used):
            used.append((x - width / 2, x + width / 2))
            return x
    return None


# ------------------------------------------------------------------- generator
def generate_plan(cfg: GenerationConfig, rng: random.Random | None = None) -> tuple[Image.Image, PlanTruth]:
    rng = rng or random.Random(cfg.seed)
    W, H = cfg.width, cfg.height
    style = Style(
        scale=rng.uniform(*cfg.scale),
        track_width=rng.choice([3, 4, 5]),
        line_width=rng.choice([2, 2, 3]),
    )
    bg = rng.choice([255, 255, 252, 248])
    img = Image.new("L", (W, H), bg)
    d = SymbolDrawer(img, style)
    objects: list[Annotation] = []
    texts: list[TextAnnotation] = []
    signals: list[SignalTruth] = []
    occupied: list[BBox] = []

    def add(cls: str, box: BBox) -> None:
        if cls in cfg.exclude:
            return
        objects.append(Annotation(cls, box))
        occupied.append(box)

    def add_text(role: str, content: str, box: BBox) -> None:
        texts.append(TextAnnotation(content, box, role))
        occupied.append(box)

    # Cadre et cartouche de titre
    margin = int(style.s(60))
    d.draw.rectangle([margin, margin, W - margin, H - margin], outline=0, width=2)
    title = f"PLAN DE SIGNALISATION - LIGNE {rng.randint(100, 999)} - POSTE {rng.randint(1, 60)}"
    tb = d.text(W - margin - style.s(20), H - margin - style.s(18), title, style.s(16), anchor="rm")
    add_text("title", title, tb)

    # Voies
    n_tracks = rng.randint(*cfg.n_tracks)
    slope = np.tan(np.radians(rng.uniform(-cfg.max_slope_deg, cfg.max_slope_deg)))
    x_left, x_right = margin + style.s(80), W - margin - style.s(80)
    block_h = (n_tracks - 1) * cfg.track_spacing
    y_top = rng.uniform(margin + style.s(220), H - margin - block_h - style.s(160))
    tracks: list[_Track] = []
    pool = [f"V{k}" for k in range(1, 7)] + ["V1bis", "V2bis", "VU", "VE"]
    track_names = rng.sample(pool, n_tracks)
    for i in range(n_tracks):
        name = track_names[i]
        tr = _Track(name, y_top + i * cfg.track_spacing, slope, x_left, 1 if i % 2 == 0 else -1)
        tracks.append(tr)
        d.draw.line(
            [(x_left, tr.y_at(x_left)), (x_right, tr.y_at(x_right))], fill=0, width=style.track_width
        )
        # nom de voie à gauche
        tb = d.text(x_left - style.s(14), tr.y_at(x_left), name, style.s(18), bold=True, anchor="rm")
        add_text("track_name", name, tb)
        # flèche de sens
        ax = rng.uniform(x_left + style.s(40), x_left + style.s(160))
        add("arrow", d.arrow(ax, tr.y_at(ax), tr.direction))

    pk_origin = rng.randrange(10_000, 400_000, 100)
    m_per_px = rng.uniform(0.8, 1.6)  # échelle fictive du plan

    def pk_at(x: float) -> int:
        return int(pk_origin + (x - x_left) * m_per_px)

    # Repères kilométriques : ligne de points numérotés entre deux voies (ou au-dessus)
    n_dots = rng.randint(*cfg.pk_dots)
    if n_tracks > 1:
        yd = (tracks[0].y0 + tracks[1].y0) / 2
    else:
        yd = tracks[0].y0 - cfg.track_spacing / 2
    used_dots: list[tuple[float, float]] = []
    for _ in range(n_dots):
        x = _free_x(rng, x_left, x_right, used_dots, style.s(70))
        if x is None:
            break
        y = yd + (x - x_left) * slope
        add("dot", d.dot(x, y))
        label = str(pk_at(x) // 100) if rng.random() < 0.5 else str(rng.randint(1, 40))
        tb = d.text(x + style.s(9), y, label, style.s(15), anchor="lm")
        add_text("pk", label, tb)

    # Par voie : zones (joints + nom de zone), signaux, équipements
    used_signal_names: set[str] = set()
    # intervalles d'abscisses déjà pris, par voie (réserve initiale : flèche + nom de voie)
    used_by_track: dict[int, list[tuple[float, float]]] = {
        i: [(x_left, x_left + style.s(180))] for i in range(n_tracks)
    }
    for ti, tr in enumerate(tracks):
        used_x = used_by_track[ti]

        # aiguille vers une voie voisine (placée en premier : c'est le symbole le plus encombrant)
        if n_tracks >= 2 and "point" not in cfg.exclude and rng.random() < 0.65:
            tj = ti + 1 if ti + 1 < n_tracks else ti - 1
            target = tracks[tj]
            for _ in range(12):
                direction = rng.choice([-1, 1])
                x = rng.uniform(x_left + style.s(220), x_right - style.s(120))
                x_end = x + direction * cfg.track_spacing * 1.6
                if not (x_left + style.s(200) < x_end < x_right - style.s(40)):
                    continue
                y, y_end = tr.y_at(x), target.y_at(x_end)
                xa, xb = sorted((x, x_end))
                est = (xa - 6, min(y, y_end) - 6, xb + 6, max(y, y_end) + 6)
                busy = any(xb > a and xa < b for a, b in used_x + used_by_track[tj])
                if busy or _overlaps(est, occupied):
                    continue
                add("point", d.point(x, y, y_end, direction))
                used_x.append((xa - style.s(10), xb + style.s(10)))
                used_by_track[tj].append((xa - style.s(10), xb + style.s(10)))
                break

        # joints délimitant les zones (espacés d'au moins ~120 px)
        n_zones = rng.randint(*cfg.zones_per_track)
        cuts: list[float] = []
        for _ in range(n_zones - 1):
            cx = _free_x(rng, x_left + style.s(200), x_right - style.s(60), used_x, style.s(60))
            if cx is not None:
                cuts.append(cx)
        cuts.sort()
        for cx in cuts:
            add("joint", d.joint(cx, tr.y_at(cx)))

        # signaux
        n_sig = rng.randint(*cfg.signals_per_track)
        for _ in range(n_sig):
            n_ind = rng.randint(1, 5)
            indications = rng.sample(INDICATIONS, n_ind)
            w_est = style.s(60) + n_ind * style.s(16) + style.s(90)
            x = _free_x(rng, x_left, x_right, used_x, w_est)
            if x is None:
                break
            y = tr.y_at(x)
            above = (tr.direction == 1) if rng.random() < 0.85 else rng.random() < 0.5
            direction = tr.direction if rng.random() < 0.9 else -tr.direction
            vertical = rng.random() < cfg.vertical_indications_p
            box, ind_texts = d.signal(x, y, indications, above=above, direction=direction,
                                      vertical_text=vertical)
            add("signal", box)
            for content, tbx in ind_texts:
                texts.append(TextAnnotation(content, tbx, "indication"))
            # nom du signal : nombre en gras, du côté du cartouche
            while True:
                name = str(rng.randint(100, 999)) + (rng.choice(["", "", "", "A", "B"]))
                if name not in used_signal_names:
                    used_signal_names.add(name)
                    break
            gap = style.s(10)
            ny = (box[1] + box[3]) / 2
            if direction == 1:
                nb = d.text(box[2] + gap, ny, name, style.s(28), bold=True, anchor="lm")
            else:
                nb = d.text(box[0] - gap, ny, name, style.s(28), bold=True, anchor="rm")
            add_text("signal_name", name, nb)
            signals.append(SignalTruth(name, tr.name, pk_at(x), indications, direction, box, nb))

        # équipements ponctuels
        n_eq = rng.randint(*cfg.extra_equipment) // n_tracks + 1
        for _ in range(n_eq):
            kind = rng.choice(["tel", "pd", "ce", "deto", "deto"])
            x = _free_x(rng, x_left, x_right, used_x, style.s(70))
            if x is None:
                continue
            y = tr.y_at(x)
            # Estimation de l'emprise AVANT de dessiner : on ne veut jamais de
            # symbole dessiné mais non annoté.
            est = (x - style.s(22), y - style.s(48), x + style.s(22), y + style.s(48))
            if _overlaps(est, occupied):
                continue
            if kind == "tel":
                off = style.s(32) * (1 if rng.random() < 0.5 else -1)
                box = d.tel(x, y + off)
            elif kind == "pd":
                box = d.pd(x, y)
            elif kind == "ce":
                box = d.ce(x, y)
            else:
                box = d.deto(x, y, above=rng.random() < 0.5)
            add(kind, box)

        # commutateurs de limitation de vitesse (bibliothèque vectorielle)
        for _ in range(rng.choice([0, 0, 1, 1, 2])):
            if "cm" in cfg.exclude:
                break
            x = _free_x(rng, x_left, x_right, used_x, style.s(70))
            if x is None:
                continue
            y = tr.y_at(x)
            above = (tr.direction == 1) if rng.random() < 0.8 else rng.random() < 0.5
            sy = -1 if above else 1
            est = (x - style.s(26), y + sy * style.s(80), x + style.s(26), y)
            est = (est[0], min(est[1], est[3]), est[2], max(est[1], est[3]))
            if _overlaps(est, occupied):
                continue
            cm_name = f"{rng.randint(1, 19)},{rng.randint(1, 9)}"
            box, cm_texts = d.render(LIBRARY["cm_t"], x, y, {"name": cm_name}, above=above)
            add("cm", box)
            for content, tb, role in cm_texts:
                add_text(role, content, tb)

        # noms de zone : du côté opposé aux signaux, uniquement si la place est libre
        bounds = [x_left + style.s(200), *cuts, x_right - style.s(40)]
        side = 1 if tr.direction == 1 else -1  # signaux au-dessus si direction +1 → zones dessous
        for a, b in itertools.pairwise(bounds):
            zname = f"z {rng.randint(100, 999)}{rng.choice(['', '', 'A', 'B'])}"
            font = load_font(int(style.s(15)))
            for frac in (0.5, 0.3, 0.7):
                zx = a + (b - a) * frac
                zy = tr.y_at(zx) + side * style.s(26)
                est = d.draw.textbbox((zx, zy), zname, font=font, anchor="mm")
                if not _overlaps(est, occupied, margin=6):
                    tb = d.text(zx, zy, zname, style.s(15), anchor="mm")
                    add_text("zone", zname, tb)
                    break

    # Bruit "scan" léger
    if cfg.noise and rng.random() < 0.8:
        arr = np.asarray(img).astype(np.float32)
        arr += rng.gauss(0, 1) * 2 + np.random.default_rng(rng.randrange(1 << 30)).normal(
            0, rng.uniform(2, 7), arr.shape
        )
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        if rng.random() < 0.4:
            img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))

    truth = PlanTruth(
        width=W,
        height=H,
        tracks=[TrackTruth(t.name, t.y_at(x_left), t.y_at(x_right), t.direction) for t in tracks],
        objects=objects,
        texts=texts,
        signals=signals,
        pk_origin_m=pk_origin,
    )
    return img, truth
