"""Extraction des voies d'un plan raster sous forme de graphe.

Une voie n'est pas un objet à boîte englobante : c'est un trait épais qui se
divise et fusionne. On l'extrait donc en trois temps :

1. masque des traits épais (les voies sont les seuls traits larges du plan) ;
2. squelette du masque ;
3. graphe : nœuds = extrémités et jonctions (aiguilles), arêtes = segments.
"""

from diagram_reader.tracks.extract import ExtractConfig, TrackGraph, extract_tracks
from diagram_reader.tracks.graph import Edge, Node

__all__ = ["Edge", "ExtractConfig", "Node", "TrackGraph", "extract_tracks"]
