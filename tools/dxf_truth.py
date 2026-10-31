"""Vérité terrain « voies » à partir d'un DXF.

    python tools/dxf_truth.py data/dxf data/truth --dpi 200

Pour chaque DXF, écrit dans le dossier de sortie :

    <nom>.png        rendu complet du plan (ce que verra le système)
    <nom>.tracks.png masque des voies (polylignes à largeur constante ≥ --min-width)
    <nom>.json       métadonnées (transformation unités DXF → pixels, largeurs)

Le masque sert à évaluer l'extraction raster (`python -m diagram_reader.tracks.eval`).
Les voies sont reconnues par leur épaisseur, pas par leur calque (les calques ne
sont pas cohérents d'un plan à l'autre).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ezdxf
import matplotlib
import numpy as np
from ezdxf import bbox
from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing.config import ColorPolicy, Configuration, LineweightPolicy
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # rouleaux de plusieurs mètres à 200 dpi

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_plan(doc, ext, width_px: int, out: Path) -> None:
    msp = doc.modelspace()
    for name in {e.dxf.layer for e in msp}:  # calques absents de la table → invisibles sinon
        if name not in doc.layers:
            doc.layers.add(name)
    w, h = ext.size.x, ext.size.y
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    cfg = Configuration(
        color_policy=ColorPolicy.BLACK,
        lineweight_policy=LineweightPolicy.RELATIVE,
        lineweight_scaling=1.0,
        min_lineweight=0.2,
    )
    Frontend(RenderContext(doc), MatplotlibBackend(ax), config=cfg).draw_layout(msp, finalize=True)
    ax.set_xlim(ext.extmin.x, ext.extmax.x)
    ax.set_ylim(ext.extmin.y, ext.extmax.y)
    dpi = 100
    fig.set_size_inches(width_px / dpi, width_px / dpi * h / w)
    fig.savefig(out, dpi=dpi, facecolor="white")
    plt.close(fig)


class TrackFrontend(Frontend):
    """Frontend qui ne dessine que les traits épais : polylignes à largeur
    constante ≥ min_width, ou entités dont l'épaisseur résolue ≥ min_lw (mm).
    Les INSERT passent pour que leurs entités internes soient filtrées à leur tour."""

    def __init__(self, ctx, backend, min_width: float, min_lw: float, stats: dict):
        super().__init__(ctx, backend, config=Configuration(color_policy=ColorPolicy.BLACK,
                                                            lineweight_policy=LineweightPolicy.ABSOLUTE,
                                                            min_lineweight=0.05))
        self.min_width, self.min_lw, self.stats = min_width, min_lw, stats

    def draw_entity(self, entity, properties):
        t = entity.dxftype()
        if t == "INSERT":
            return super().draw_entity(entity, properties)
        thick = False
        if t == "LWPOLYLINE":
            cw = entity.dxf.const_width if entity.dxf.hasattr("const_width") else 0.0
            ws = [cw] + [max(p[2], p[3]) for p in entity.get_points() if len(p) >= 4]
            thick = max(ws) >= self.min_width
            if thick:
                self.stats.setdefault("widths", set()).add(round(max(ws), 3))
        elif t == "POLYLINE":
            ws = [max(getattr(v.dxf, "start_width", 0.0), getattr(v.dxf, "end_width", 0.0))
                  for v in entity.vertices]
            thick = bool(ws) and max(ws) >= self.min_width
        elif t in ("LINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE"):
            thick = (properties.lineweight or 0.0) >= self.min_lw
            if thick:
                self.stats.setdefault("lineweights", set()).add(round(properties.lineweight, 2))
        if thick:
            self.stats["n"] = self.stats.get("n", 0) + 1
            super().draw_entity(entity, properties)


def render_track_mask(doc, ext, width_px: int, out: Path, min_width: float, min_lw: float) -> dict:
    msp = doc.modelspace()
    w, h = ext.size.x, ext.size.y
    stats: dict = {}
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    TrackFrontend(RenderContext(doc), MatplotlibBackend(ax), min_width, min_lw, stats).draw_layout(
        msp, finalize=True)
    ax.set_xlim(ext.extmin.x, ext.extmax.x)
    ax.set_ylim(ext.extmin.y, ext.extmax.y)
    dpi = 100
    fig.set_size_inches(width_px / dpi, width_px / dpi * h / w)
    fig.savefig(out, dpi=dpi, facecolor="white")
    plt.close(fig)
    # binarisation : encre → 255
    img = np.array(Image.open(out).convert("L"))
    Image.fromarray(((img < 128) * 255).astype(np.uint8)).save(out)
    return {"n_thick_entities": stats.get("n", 0),
            "widths": sorted(stats.get("widths", set())),
            "lineweights": sorted(stats.get("lineweights", set()))}


def process(dxf: Path, out_dir: Path, width_px: int, min_width: float, min_lw: float) -> dict:
    doc = ezdxf.readfile(dxf)
    msp = doc.modelspace()
    ext = bbox.extents(msp, fast=True)
    height_px = round(width_px * ext.size.y / ext.size.x)
    stem = dxf.stem
    render_plan(doc, ext, width_px, out_dir / f"{stem}.png")
    st = render_track_mask(doc, ext, width_px, out_dir / f"{stem}.tracks.png", min_width, min_lw)
    meta = {
        "source": dxf.name,
        "width_px": width_px,
        "height_px": height_px,
        "extmin": [ext.extmin.x, ext.extmin.y],
        "extmax": [ext.extmax.x, ext.extmax.y],
        "px_per_unit": width_px / ext.size.x,
        **st,
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=1))
    return meta


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Rendu + masque des voies depuis des DXF.")
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument("--dpi", type=float, default=200.0,
                   help="résolution du rendu (unités DXF supposées en mm) ; 0 = utiliser --width")
    p.add_argument("--width", type=int, default=10000, help="largeur du rendu en px si --dpi 0")
    p.add_argument("--max-pixels", type=float, default=250e6, help="plafond de pixels par rendu")
    p.add_argument("--min-width", type=float, default=0.45,
                   help="largeur de polyligne (unités DXF) à partir de laquelle c'est une voie")
    p.add_argument("--min-lw", type=float, default=0.5,
                   help="épaisseur de trait résolue (mm) à partir de laquelle une ligne est une voie")
    p.add_argument("--only", nargs="*", help="ne traiter que ces fichiers (noms sans extension)")
    args = p.parse_args(argv)
    args.dst.mkdir(parents=True, exist_ok=True)
    files = sorted(args.src.glob("*.dxf"))
    if args.only:
        files = [f for f in files if f.stem in args.only]
    for f in files:
        if args.dpi > 0:
            ext = bbox.extents(ezdxf.readfile(f).modelspace(), fast=True)
            width_px = round(ext.size.x / 25.4 * args.dpi)
            # garde-fou mémoire : les espaces objet qui empilent plusieurs planches
            # deviennent gigantesques ; on plafonne le nombre de pixels
            if width_px * width_px * ext.size.y / ext.size.x > args.max_pixels:
                width_px = int((args.max_pixels * ext.size.x / ext.size.y) ** 0.5)
                print(f"{f.name}: rendu plafonné à {width_px} px de large")
        else:
            width_px = args.width
        meta = process(f, args.dst, width_px, args.min_width, args.min_lw)
        print(f"{f.name}: {meta['n_thick_entities']} entités épaisses, largeurs {meta['widths']}, "
              f"lineweights {meta['lineweights']}")


if __name__ == "__main__":
    main()
