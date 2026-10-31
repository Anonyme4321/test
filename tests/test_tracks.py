import random

import cv2
import numpy as np

from diagram_reader.synth import GenerationConfig, generate_plan
from diagram_reader.tracks import ExtractConfig, extract_tracks
from diagram_reader.tracks.graph import Edge, Node, TrackGraph, bridge_gaps, prune, skeleton_to_graph


def _line_image(w=1200, h=400):
    """Deux voies épaisses, une bretelle oblique, des traits fins parasites."""
    img = np.full((h, w), 255, np.uint8)
    cv2.line(img, (50, 120), (1150, 120), 0, 6)
    cv2.line(img, (50, 280), (1150, 280), 0, 6)
    cv2.line(img, (400, 120), (560, 280), 0, 6)  # bretelle : 2 jonctions
    for x in range(100, 1100, 90):  # mâts fins + petits carrés fins
        cv2.line(img, (x, 120), (x, 60), 0, 2)
        cv2.rectangle(img, (x - 8, 40), (x + 8, 60), 0, 2)
    cv2.rectangle(img, (20, 20), (1180, 380), 0, 2)  # cadre fin
    return img


def test_extract_simple_scene():
    r = extract_tracks(_line_image(), ExtractConfig(adaptive=False))
    g = r.graph
    assert 2.0 <= r.half_width <= 3.5
    assert len(g.junctions()) == 2, [(n.x, n.y, n.kind) for n in g.nodes]
    # ~1100 + 1100 + 226 px de voie
    assert 2200 <= g.total_length() <= 2600
    # le cadre fin et les mâts n'apparaissent pas
    for e in g.edges:
        ys = [p[1] for p in e.points]
        assert min(ys) > 30


def test_bridge_gaps_reconnects_aligned_ends():
    g = TrackGraph(1000, 100)
    g.nodes = [Node(0, 0, 50, "end"), Node(1, 400, 50, "end"), Node(2, 440, 50, "end"),
               Node(3, 900, 50, "end")]
    g.edges = [Edge(0, 0, 1, [(0, 50), (200, 50), (400, 50)], 400.0),
               Edge(1, 2, 3, [(440, 50), (700, 50), (900, 50)], 460.0)]
    assert bridge_gaps(g, max_gap=60) == 1
    assert len(g.edges) == 1 and g.edges[0].length > 890


def test_bridge_gaps_ignores_misaligned_ends():
    g = TrackGraph(1000, 400)
    g.nodes = [Node(0, 0, 50, "end"), Node(1, 400, 50, "end"), Node(2, 420, 90, "end"),
               Node(3, 420, 390, "end")]
    g.edges = [Edge(0, 0, 1, [(0, 50), (400, 50)], 400.0),
               Edge(1, 2, 3, [(420, 90), (420, 390)], 300.0)]
    assert bridge_gaps(g, max_gap=60) == 0


def test_skeleton_graph_prune_roundtrip():
    skel = np.zeros((100, 300), bool)
    skel[50, 10:290] = True
    skel[20:50, 150] = True  # barbule verticale de 30 px
    g = prune(skeleton_to_graph(skel), min_spur=40, merge_dist=5)
    assert len(g.edges) == 1 and len(g.junctions()) == 0


def test_extract_on_synthetic_plan():
    cfg = GenerationConfig(width=2400, height=1600, noise=False, scale=(1.3, 1.3))
    img, truth = generate_plan(cfg, random.Random(3))
    r = extract_tracks(np.asarray(img), ExtractConfig(adaptive=False))
    expected = sum(np.hypot(cfg.width, t.y_right - t.y_left) for t in truth.tracks) * 0.9
    # on retrouve au moins la longueur des voies principales (hors bretelles)
    assert r.graph.total_length() >= expected * 0.85
