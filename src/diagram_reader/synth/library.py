"""Bibliothèque vectorielle de symboles.

Chaque symbole est décrit par des primitives dans un repère local en pixels
(échelle 1) dont l'origine (0, 0) est le **point d'attache sur la voie**, axe y
vers le bas (convention image). Le symbole est décrit pour une pose « au-dessus
de la voie » ; le rendu peut le retourner verticalement.

Les textes peuvent être fixes (`text`) ou variables (`slot`) : la valeur d'un
slot est fournie au rendu et annotée avec le `role` indiqué.

Cette description est volontairement proche d'un DXF : elle s'exporte en DXF
(voir `dxf_export.py`) pour être ouverte ou retouchée dans un outil de CAO.
"""

from __future__ import annotations

from dataclasses import dataclass, field

Point = tuple[float, float]


@dataclass(frozen=True)
class Prim:
    kind: str  # "rect" | "line" | "circle" | "polygon" | "text"
    pts: tuple[Point, ...] = ()  # rect : 2 coins ; line/polygon : sommets ; circle/text : centre
    r: float = 0.0  # rayon (circle)
    fill: bool = False
    width: float | None = None  # épaisseur de trait ; None = épaisseur par défaut du style
    text: str | None = None  # texte fixe
    slot: str | None = None  # nom du slot de texte variable
    role: str = "fixed"  # rôle d'annotation du texte
    size: float = 12.0  # hauteur de police (px, échelle 1)
    bold: bool = False
    in_bbox: bool = True  # participe à la boîte englobante du symbole
    group: str = "default"  # textes d'un même groupe : ordre de lecture conservé au retournement


@dataclass(frozen=True)
class VectorSymbol:
    name: str
    cls: str  # classe de détection
    description: str
    prims: tuple[Prim, ...]
    slots: dict[str, str] = field(default_factory=dict)  # slot -> description du format


# --------------------------------------------------------------------------- symboles

CM_T = VectorSymbol(
    name="cm_t",
    cls="cm",
    description=(
        "Commutateur de limitation de vitesse (Cm T 160). Carré avec « Cm T » et « 160 », "
        "petit mât vers la voie, nom du commutateur (ex. 9,9 ou 9,9bis) du côté opposé à la voie."
    ),
    prims=(
        # mât : de la voie (0,0) jusqu'au bas du carré
        Prim("line", ((0, 0), (0, -12))),
        # carré 44 × 44 au-dessus du mât
        Prim("rect", ((-22, -56), (22, -12)), width=2.5),
        # textes fixes, deux lignes
        Prim("text", ((0, -44),), text="Cm T", size=13, group="box"),
        Prim("text", ((0, -25),), text="160", size=13, group="box"),
        # nom variable du côté opposé à la voie ; hors boîte du symbole (texte à associer)
        Prim("text", ((0, -66),), slot="name", role="cm_name", size=16, in_bbox=False, group="name"),
    ),
    slots={"name": "numéro du commutateur, « c,c » ou « c,c^bis » (bis en exposant)"},
)

LIBRARY: dict[str, VectorSymbol] = {s.name: s for s in (CM_T,)}
