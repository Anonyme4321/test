"""Planche de contrôle : dessine chaque symbole isolé, en grand, avec sa boîte.

    python -m diagram_reader.synth.sheet --out docs/symbols.png

Sert à valider visuellement un symbole fraîchement ajouté à la bibliothèque
(ou aux primitives codées) avant de le lâcher dans les plans.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from diagram_reader.synth.library import LIBRARY
from diagram_reader.synth.symbols import Style, SymbolDrawer

CELL = 220


def _draw_all(scale: float = 2.0) -> Image.Image:
    style = Style(scale=scale, track_width=4, line_width=2)
    # symboles codés
    coded = [
        ("signal", lambda d, x, y: d.signal(x, y, ["C", "S", "RR"], above=True)[0]),
        ("joint", lambda d, x, y: d.joint(x, y)),
        ("dot", lambda d, x, y: d.dot(x, y)),
        ("arrow", lambda d, x, y: d.arrow(x, y, 1)),
        ("point", lambda d, x, y: d.point(x, y, y + 60, 1)),
        ("deto", lambda d, x, y: d.deto(x, y)),
        ("tel", lambda d, x, y: d.tel(x, y - 40)),
        ("pd", lambda d, x, y: d.pd(x, y)),
        ("ce", lambda d, x, y: d.ce(x, y)),
    ]
    def _lib(sym):
        return lambda d, x, y: d.render(sym, x, y, dict.fromkeys(sym.slots, "9,9"))[0]

    library = [(sym.name, _lib(sym)) for sym in LIBRARY.values()]
    items = coded + library
    cols = 5
    rows = (len(items) + cols - 1) // cols
    img = Image.new("L", (cols * CELL, rows * CELL), 255)
    d = SymbolDrawer(img, style)
    overlay = ImageDraw.Draw(img)
    for i, (name, fn) in enumerate(items):
        cx = (i % cols) * CELL + CELL // 2
        cy = (i // cols) * CELL + CELL * 0.62
        # bout de voie
        d.draw.line([(cx - 80, cy), (cx + 80, cy)], fill=0, width=style.track_width)
        box = fn(d, cx, cy)
        overlay.rectangle(box, outline=120, width=1)
        d.text((i % cols) * CELL + 8, (i // cols) * CELL + 14, name, 16, anchor="lm")
    return img


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Planche de contrôle des symboles.")
    p.add_argument("--out", type=Path, default=Path("docs/symbols.png"))
    args = p.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _draw_all().save(args.out)
    print(f"OK -> {args.out}")


if __name__ == "__main__":
    main()
