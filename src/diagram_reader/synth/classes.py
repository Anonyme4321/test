"""Catégories de symboles détectées.

Les classes correspondent aux éléments usuels d'un schéma de signalisation
(terminologie ferroviaire française). L'ordre définit l'identifiant YOLO.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolClass:
    name: str
    description: str


CLASSES: tuple[SymbolClass, ...] = (
    SymbolClass("arrow", "Flèche indiquant le sens de circulation de la voie"),
    SymbolClass("dot", "Point noir : repère kilométrique / hectométrique numéroté"),
    SymbolClass("point", "Aiguille : branchement entre deux voies"),
    SymbolClass("signal", "Panneau de signalisation (cartouche avec indications)"),
    SymbolClass("deto", "Pétard / détonateur de voie"),
    SymbolClass("tel", "Téléphone de voie"),
    SymbolClass("pd", "Pédale : détecteur de passage"),
    SymbolClass("ce", "Compteur d'essieux"),
    SymbolClass("joint", "Joint isolant délimitant une zone de détection"),
)

CLASS_NAMES: tuple[str, ...] = tuple(c.name for c in CLASSES)
CLASS_IDS: dict[str, int] = {c.name: i for i, c in enumerate(CLASSES)}
