"""Export des symboles de la bibliothèque en DXF (un bloc par symbole).

    python -m diagram_reader.synth.dxf_export --out docs/symbols.dxf

Nécessite `ezdxf` (`pip install ezdxf`). Le repère DXF a l'axe y vers le haut :
on inverse y par rapport au repère image de la bibliothèque. Les slots de texte
variables sont exportés comme ATTDEF (attributs de bloc).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from diagram_reader.synth.library import LIBRARY, VectorSymbol


def symbol_to_block(doc, sym: VectorSymbol) -> None:
    blk = doc.blocks.new(name=sym.name.upper())
    for pr in sym.prims:
        pts = [(px, -py) for px, py in pr.pts]
        if pr.kind == "line":
            blk.add_lwpolyline(pts)
        elif pr.kind == "rect":
            (x1, y1), (x2, y2) = pts
            blk.add_lwpolyline([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], close=True)
        elif pr.kind == "circle":
            blk.add_circle(pts[0], radius=pr.r)
        elif pr.kind == "polygon":
            blk.add_lwpolyline(pts, close=True)
        elif pr.kind == "text":
            cx, cy = pts[0]
            if pr.slot is None:
                t = blk.add_text(pr.text or "", height=pr.size * 0.72)
                t.set_placement((cx, cy), align=_align())
            else:
                blk.add_attdef(
                    tag=pr.slot.upper(),
                    insert=(cx, cy),
                    text=pr.slot,
                    height=pr.size * 0.72,
                    dxfattribs={"prompt": sym.slots.get(pr.slot, pr.slot)},
                )


def _align():
    from ezdxf.enums import TextEntityAlignment

    return TextEntityAlignment.MIDDLE_CENTER


def export(path: Path, spacing: float = 120.0) -> None:
    import ezdxf

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    for i, sym in enumerate(LIBRARY.values()):
        symbol_to_block(doc, sym)
        msp.add_blockref(sym.name.upper(), (i * spacing, 0))
        msp.add_text(sym.name, height=6).set_placement((i * spacing, -20), align=_align())
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(path)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Exporte la bibliothèque de symboles en DXF.")
    p.add_argument("--out", type=Path, default=Path("docs/symbols.dxf"))
    args = p.parse_args(argv)
    export(args.out)
    print(f"OK -> {args.out} ({len(LIBRARY)} symboles)")


if __name__ == "__main__":
    main()
