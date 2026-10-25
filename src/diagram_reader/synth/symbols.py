"""Primitives de dessin des symboles de signalisation.

Chaque méthode dessine un symbole sur l'image et renvoie sa boîte englobante
en pixels (x1, y1, x2, y2). Les textes sont rendus séparément et renvoient
aussi leur boîte, pour constituer la vérité terrain OCR.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image, ImageDraw

from diagram_reader.synth.fonts import load_font

BBox = tuple[float, float, float, float]


@dataclass
class Style:
    """Paramètres graphiques partagés par un plan."""

    scale: float = 1.0  # multiplicateur global de taille des symboles
    track_width: int = 4  # épaisseur des voies
    line_width: int = 2  # épaisseur des traits de symboles
    ink: int = 0  # niveau de gris de l'encre (0 = noir)

    def s(self, v: float) -> float:
        return v * self.scale


def _union(*boxes: BBox) -> BBox:
    xs1, ys1, xs2, ys2 = zip(*boxes)
    return (min(xs1), min(ys1), max(xs2), max(ys2))


def _pad(box: BBox, p: float) -> BBox:
    return (box[0] - p, box[1] - p, box[2] + p, box[3] + p)


class SymbolDrawer:
    def __init__(self, image: Image.Image, style: Style):
        self.image = image
        self.draw = ImageDraw.Draw(image)
        self.style = style

    # ------------------------------------------------------------------ texte
    def text(
        self,
        x: float,
        y: float,
        content: str,
        size: float,
        bold: bool = False,
        anchor: str = "lm",
        angle: int = 0,
    ) -> BBox:
        """Écrit `content` ; `anchor` suit la convention Pillow (ex. "lm" = gauche, milieu).

        `angle` ∈ {0, 90, 270} : texte horizontal ou vertical (lecture de bas en
        haut pour 90). Renvoie la boîte englobante réelle du texte.
        """
        font = load_font(int(size), bold=bold)
        if angle == 0:
            box = self.draw.textbbox((x, y), content, font=font, anchor=anchor)
            self.draw.text((x, y), content, font=font, fill=self.style.ink, anchor=anchor)
            return tuple(float(v) for v in box)  # type: ignore[return-value]

        # Texte tourné : rendu sur un calque, rotation, collage.
        l, t, r, b = self.draw.textbbox((0, 0), content, font=font, anchor="la")
        w, h = math.ceil(r - l) + 2, math.ceil(b - t) + 2
        layer = Image.new("L", (w, h), 255)
        ImageDraw.Draw(layer).text((-l + 1, -t + 1), content, font=font, fill=self.style.ink)
        layer = layer.rotate(angle, expand=True, fillcolor=255)
        lw, lh = layer.size
        # Positionnement : on interprète (x, y) comme le centre du bloc tourné.
        px, py = round(x - lw / 2), round(y - lh / 2)
        self.image.paste(layer, (px, py), mask=Image.eval(layer, lambda v: 255 - v))
        return (float(px), float(py), float(px + lw), float(py + lh))

    # --------------------------------------------------------------- symboles
    def joint(self, x: float, y: float) -> BBox:
        """Joint isolant : deux petits traits perpendiculaires à la voie."""
        st = self.style
        h = st.s(9)
        gap = st.s(3)
        for dx in (-gap, gap):
            self.draw.line([(x + dx, y - h), (x + dx, y + h)], fill=st.ink, width=st.line_width)
        return _pad((x - gap, y - h, x + gap, y + h), st.s(2))

    def dot(self, x: float, y: float) -> BBox:
        """Point noir plein (repère kilométrique, hectométrique…)."""
        r = self.style.s(5)
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=self.style.ink)
        return _pad((x - r, y - r, x + r, y + r), self.style.s(1.5))

    def deto(self, x: float, y: float, above: bool = True) -> BBox:
        """Pétard : demi-disque plein posé sur la voie."""
        r = self.style.s(7)
        if above:
            self.draw.pieslice([x - r, y - r, x + r, y + r], 180, 360, fill=self.style.ink)
            return _pad((x - r, y - r, x + r, y), self.style.s(2))
        self.draw.pieslice([x - r, y - r, x + r, y + r], 0, 180, fill=self.style.ink)
        return _pad((x - r, y, x + r, y + r), self.style.s(2))

    def arrow(self, x: float, y: float, direction: int = 1) -> BBox:
        """Flèche pleine sur la voie ; direction +1 = vers la droite."""
        L, H = self.style.s(18), self.style.s(7)
        tip = (x + direction * L / 2, y)
        base = x - direction * L / 2
        self.draw.polygon([tip, (base, y - H), (base, y + H)], fill=self.style.ink)
        return _pad((x - L / 2, y - H, x + L / 2, y + H), self.style.s(2))

    def tel(self, x: float, y: float) -> BBox:
        """Téléphone : boîtier rectangulaire surmonté d'un combiné (arc)."""
        st = self.style
        w, h = st.s(16), st.s(11)
        box = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
        self.draw.rectangle(box, outline=st.ink, width=st.line_width)
        arc = (x - w * 0.45, y - h / 2 - st.s(7), x + w * 0.45, y - h / 2 + st.s(5))
        self.draw.arc(arc, 200, 340, fill=st.ink, width=st.line_width)
        return _pad((box[0], arc[1], box[2], box[3]), st.s(2))

    def pd(self, x: float, y: float) -> BBox:
        """Pédale : petit rectangle barré d'une diagonale, à cheval sur la voie."""
        st = self.style
        w, h = st.s(18), st.s(9)
        box = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
        self.draw.rectangle(box, fill=255, outline=st.ink, width=st.line_width)
        self.draw.line([(box[0], box[3]), (box[2], box[1])], fill=st.ink, width=st.line_width)
        return _pad(box, st.s(2))

    def ce(self, x: float, y: float) -> BBox:
        """Compteur d'essieux : cercle avec une croix, à cheval sur la voie."""
        st = self.style
        r = st.s(8)
        box = (x - r, y - r, x + r, y + r)
        self.draw.ellipse(box, fill=255, outline=st.ink, width=st.line_width)
        self.draw.line([(x - r, y), (x + r, y)], fill=st.ink, width=st.line_width)
        self.draw.line([(x, y - r), (x, y + r)], fill=st.ink, width=st.line_width)
        return _pad(box, st.s(2))

    def point(self, x: float, y_from: float, y_to: float, direction: int = 1) -> BBox:
        """Aiguille : branche oblique reliant deux voies, avec un talon plein à l'origine."""
        st = self.style
        dx = direction * abs(y_to - y_from) * 1.6
        self.draw.line([(x, y_from), (x + dx, y_to)], fill=st.ink, width=st.track_width)
        # talon : petit triangle plein à la racine
        L, H = st.s(14), st.s(5)
        self.draw.polygon(
            [(x, y_from), (x + direction * L, y_from - H), (x + direction * L, y_from + H)],
            fill=st.ink,
        )
        # La boîte annotée est celle de la pointe d'aiguille (talon + départ de la
        # branche), pas de toute la branche oblique : c'est ce qu'un détecteur
        # peut apprendre de façon stable.
        reach = L * 2.2
        x1, x2 = sorted((x - st.s(4), x + direction * reach))
        y_br = y_from + (y_to - y_from) * (reach / abs(dx))
        y1, y2 = min(y_from - H, y_br), max(y_from + H, y_br)
        return _pad((x1, y1, x2, y2), st.s(3))

    def signal(
        self,
        x: float,
        y: float,
        indications: list[str],
        above: bool = True,
        direction: int = 1,
        vertical_text: bool = True,
    ) -> tuple[BBox, list[tuple[str, BBox]]]:
        """Panneau de signalisation.

        Mât perpendiculaire à la voie, pied carré, puis cartouche à cases contenant
        les lettres d'indication. Renvoie la boîte du symbole et la liste des
        textes (lettres) avec leur boîte.
        """
        st = self.style
        mast = st.s(34)
        foot = st.s(16)
        cell_w = st.s(14) if vertical_text else st.s(19)
        cell_h = st.s(30)
        font_size = st.s(13)
        sign = -1 if above else 1

        # mât
        y_mast_end = y + sign * mast
        self.draw.line([(x, y), (x, y_mast_end)], fill=st.ink, width=st.line_width)
        # pied carré, centré sur le mât
        fy1, fy2 = sorted((y_mast_end, y_mast_end + sign * foot))
        foot_box = (x - foot / 2, fy1, x + foot / 2, fy2)
        self.draw.rectangle(foot_box, fill=255, outline=st.ink, width=st.line_width + 1)
        # cartouche à côté du pied, dans le sens de circulation
        cy = (fy1 + fy2) / 2
        cx1 = x + direction * (foot / 2 + st.s(3))
        cx2 = cx1 + direction * cell_w * max(1, len(indications))
        bx1, bx2 = sorted((cx1, cx2))
        box = (bx1, cy - cell_h / 2, bx2, cy + cell_h / 2)
        self.draw.rectangle(box, fill=255, outline=st.ink, width=st.line_width)
        texts: list[tuple[str, BBox]] = []
        for i, ind in enumerate(indications):
            cell_x1 = bx1 + i * cell_w
            if i > 0:
                self.draw.line(
                    [(cell_x1, box[1]), (cell_x1, box[3])], fill=st.ink, width=st.line_width
                )
            tb = self.text(
                cell_x1 + cell_w / 2,
                cy,
                ind,
                font_size,
                anchor="mm",
                angle=90 if vertical_text else 0,
            )
            texts.append((ind, tb))
        full = _union((x - st.s(2), min(y, y_mast_end), x + st.s(2), max(y, y_mast_end)),
                      foot_box, box)
        return _pad(full, st.s(2)), texts
