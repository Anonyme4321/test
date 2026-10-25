"""Synthétiseur de plans de signalisation annotés.

Génère des images de schémas de voies (style plan technique noir sur blanc)
avec leurs annotations : boîtes des symboles (format YOLO), textes (pour
l'évaluation OCR) et vérité terrain d'association signal ↔ nom ↔ voie ↔ PK.
"""

from diagram_reader.synth.classes import CLASS_IDS, CLASSES
from diagram_reader.synth.generator import GenerationConfig, generate_plan

__all__ = ["CLASSES", "CLASS_IDS", "GenerationConfig", "generate_plan"]
