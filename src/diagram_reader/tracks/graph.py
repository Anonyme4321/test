"""Graphe de voies : construction depuis un squelette, élagage, simplification."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

Point = tuple[float, float]


@dataclass
class Node:
    id: int
    x: float
    y: float
    kind: str  # "end" | "junction"


@dataclass
class Edge:
    id: int
    a: int  # id du nœud de départ
    b: int  # id du nœud d'arrivée
    points: list[Point]  # polyligne (pixels), de a vers b
    length: float = 0.0
    name: str | None = None  # nom de voie (rempli plus tard par l'association)


@dataclass
class TrackGraph:
    width: int
    height: int
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    # ----------------------------------------------------------------- utilitaires
    def node(self, nid: int) -> Node:
        return next(n for n in self.nodes if n.id == nid)

    def degree(self, nid: int) -> int:
        return sum((e.a == nid) + (e.b == nid) for e in self.edges)

    def junctions(self) -> list[Node]:
        return [n for n in self.nodes if n.kind == "junction"]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> TrackGraph:
        return cls(
            d["width"],
            d["height"],
            [Node(**n) for n in d["nodes"]],
            [Edge(**{**e, "points": [tuple(p) for p in e["points"]]}) for e in d["edges"]],
        )

    def total_length(self) -> float:
        return sum(e.length for e in self.edges)

    def draw(self, image: np.ndarray | None = None) -> np.ndarray:
        """Superpose le graphe (arêtes colorées, nœuds cerclés) sur une image BGR."""
        if image is None:
            canvas = np.full((self.height, self.width, 3), 255, np.uint8)
        else:
            canvas = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            canvas = canvas.copy()
        rng = np.random.default_rng(0)
        for e in self.edges:
            color = tuple(int(c) for c in rng.integers(40, 220, 3))
            pts = np.array(e.points, np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, color, 3)
        for n in self.nodes:
            c = (0, 0, 230) if n.kind == "junction" else (0, 160, 0)
            cv2.circle(canvas, (int(n.x), int(n.y)), 9, c, 2)
        return canvas


# ---------------------------------------------------------------- squelette → graphe
def _polyline_length(pts: list[Point]) -> float:
    if len(pts) < 2:
        return 0.0
    a = np.asarray(pts, float)
    return float(np.sum(np.hypot(*(a[1:] - a[:-1]).T)))


def skeleton_to_graph(skel: np.ndarray) -> TrackGraph:
    """Construit un graphe brut (non élagué) à partir d'un squelette binaire (via skan)."""
    from skan import Skeleton

    skel = skel.astype(bool)
    h, w = skel.shape
    g = TrackGraph(w, h)
    if skel.sum() < 2:
        return g
    sk = Skeleton(skel)
    degrees = sk.degrees  # degré de chaque pixel du squelette
    coords = sk.coordinates  # (n, 2) en (ligne, colonne)
    node_ids: dict[int, int] = {}

    def node_for(px: int) -> int:
        if px not in node_ids:
            y, x = coords[px]
            kind = "junction" if degrees[px] >= 3 else "end"
            node_ids[px] = len(g.nodes)
            g.nodes.append(Node(node_ids[px], float(x), float(y), kind))
        return node_ids[px]

    for i in range(sk.n_paths):
        path = sk.path(i)  # indices de pixels, de la source à la destination
        if len(path) < 2:
            continue
        a, b = node_for(int(path[0])), node_for(int(path[-1]))
        pts = [(float(coords[p][1]), float(coords[p][0])) for p in path]
        if a == b and len(pts) < 4:
            continue
        g.edges.append(Edge(len(g.edges), a, b, pts, _polyline_length(pts)))
    return g


# ----------------------------------------------------------------------- nettoyage
def prune(
    g: TrackGraph,
    min_spur: float,
    merge_dist: float,
    epsilon: float = 1.5,
    min_loop: float | None = None,
    min_extent: float | None = None,
) -> TrackGraph:
    """Élague les barbules, fusionne les nœuds proches, simplifie les polylignes.

    - un *spur* est une arête dont une extrémité est un nœud libre (degré 1) et
      dont la longueur est < min_spur : ce sont les traces des symboles collés
      à la voie, pas des voies ;
    - une petite *boucle* (arête courte dont les deux extrémités sont proches)
      est la trace d'un cartouche ou d'une boîte collée à la voie ;
    - les nœuds à moins de merge_dist px sont fusionnés (une aiguille produit
      souvent deux jonctions très proches dans le squelette) ;
    - les nœuds de degré 2 restants sont absorbés ;
    - les composantes connexes dont l'étendue est < min_extent sont supprimées
      (bâtiments, titres en gras, hachures denses) ;
    - chaque polyligne est simplifiée (Douglas-Peucker, epsilon px).
    """
    min_loop = min_spur if min_loop is None else min_loop
    for _ in range(6):
        before = len(g.edges)
        nodes = {n.id: n for n in g.nodes}
        degree = dict.fromkeys(nodes, 0)
        pair_count: dict[frozenset, int] = {}
        for e in g.edges:
            degree[e.a] += 1
            degree[e.b] += 1
            key = frozenset((e.a, e.b))
            pair_count[key] = pair_count.get(key, 0) + 1
        keep = []
        for e in g.edges:
            da, db = degree[e.a], degree[e.b]
            if e.length < min_spur and (da == 1) != (db == 1):
                continue  # barbule
            na, nb = nodes[e.a], nodes[e.b]
            small = e.length < min_loop * 2 and np.hypot(na.x - nb.x, na.y - nb.y) < min_loop
            if small and (e.a == e.b or pair_count[frozenset((e.a, e.b))] > 1):
                pair_count[frozenset((e.a, e.b))] -= 1
                continue  # petite boucle / doublon entre deux nœuds voisins
            keep.append(e)
        g.edges = keep
        _drop_orphan_nodes(g)
        _absorb_degree2(g)
        if len(g.edges) == before:
            break
    _merge_close_nodes(g, merge_dist)
    _absorb_degree2(g)
    if min_extent:
        _drop_small_components(g, min_extent)
    for e in g.edges:
        pts = np.array(e.points, np.float32).reshape(-1, 1, 2)
        approx = cv2.approxPolyDP(pts, epsilon, False).reshape(-1, 2)
        e.points = [(float(x), float(y)) for x, y in approx]
        e.length = _polyline_length(e.points)
    degree = {n.id: 0 for n in g.nodes}
    for e in g.edges:
        degree[e.a] += 1
        degree[e.b] += 1
    for n in g.nodes:
        n.kind = "junction" if degree[n.id] >= 3 else "end"
    for i, e in enumerate(g.edges):
        e.id = i
    return g


def _end_direction(e: Edge, nid: int) -> tuple[float, float] | None:
    """Direction unitaire « sortante » au nœud `nid` de l'arête `e` (vers l'extérieur)."""
    if len(e.points) < 2:
        return None
    pts = e.points if e.b == nid else e.points[::-1]
    back = max(0, len(pts) - 6)  # un point un peu en arrière pour une direction stable
    dx, dy = pts[-1][0] - pts[back][0], pts[-1][1] - pts[back][1]
    n = np.hypot(dx, dy)
    return (dx / n, dy / n) if n > 0 else None


def bridge_gaps(g: TrackGraph, max_gap: float, max_angle_deg: float = 20.0) -> int:
    """Relie deux nœuds libres qui se font face à travers une courte coupure.

    C'est le cas typique d'un cartouche de nom de signal posé sur la voie : la
    voie s'interrompt, puis repart dans le même alignement. Renvoie le nombre
    de ponts ajoutés.
    """
    from scipy.spatial import cKDTree

    cos_max = np.cos(np.radians(max_angle_deg))
    adj: dict[int, list[Edge]] = {n.id: [] for n in g.nodes}
    for e in g.edges:
        adj[e.a].append(e)
        adj[e.b].append(e)
    ends = [n for n in g.nodes if len(adj[n.id]) == 1 and adj[n.id][0].a != adj[n.id][0].b]
    if len(ends) < 2:
        return 0
    dirs = {n.id: _end_direction(adj[n.id][0], n.id) for n in ends}
    xy = np.array([[n.x, n.y] for n in ends])
    pairs = []
    for i, j in cKDTree(xy).query_pairs(max_gap):
        a, b = ends[i], ends[j]
        pairs.append((float(np.hypot(b.x - a.x, b.y - a.y)), a, b))
    used: set[int] = set()
    added = 0
    for d, a, b in sorted(pairs, key=lambda t: t[0]):
        if d == 0 or a.id in used or b.id in used:
            continue
        da, db = dirs.get(a.id), dirs.get(b.id)
        if da is None or db is None:
            continue
        ux, uy = (b.x - a.x) / d, (b.y - a.y) / d
        # a doit « pointer » vers b, et b vers a
        if da[0] * ux + da[1] * uy < cos_max or db[0] * (-ux) + db[1] * (-uy) < cos_max:
            continue
        g.edges.append(Edge(len(g.edges), a.id, b.id, [(a.x, a.y), (b.x, b.y)], d))
        used.update((a.id, b.id))
        added += 1
    if added:
        _absorb_degree2(g)
        for i, e in enumerate(g.edges):
            e.id = i
    return added


def _drop_small_components(g: TrackGraph, min_extent: float) -> None:
    """Supprime les composantes connexes dont la diagonale de la boîte < min_extent."""
    parent = {n.id: n.id for n in g.nodes}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for e in g.edges:
        parent[find(e.a)] = find(e.b)
    comps: dict[int, list[Edge]] = {}
    for e in g.edges:
        comps.setdefault(find(e.a), []).append(e)
    keep: list[Edge] = []
    for edges in comps.values():
        xs = [p[0] for e in edges for p in e.points]
        ys = [p[1] for e in edges for p in e.points]
        if np.hypot(max(xs) - min(xs), max(ys) - min(ys)) >= min_extent:
            keep.extend(edges)
    g.edges = keep
    _drop_orphan_nodes(g)


def _drop_orphan_nodes(g: TrackGraph) -> None:
    used = {e.a for e in g.edges} | {e.b for e in g.edges}
    g.nodes = [n for n in g.nodes if n.id in used]


def _merge_close_nodes(g: TrackGraph, dist: float) -> None:
    if dist <= 0 or len(g.nodes) < 2:
        return
    ids = [n.id for n in g.nodes]
    xy = np.array([[n.x, n.y] for n in g.nodes])
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    from scipy.spatial import cKDTree

    for i, j in cKDTree(xy).query_pairs(dist):
        parent[find(ids[j])] = find(ids[i])
    groups: dict[int, list[Node]] = {}
    for n in g.nodes:
        groups.setdefault(find(n.id), []).append(n)
    remap = {}
    new_nodes = []
    for root, members in groups.items():
        nid = root
        x = float(np.mean([m.x for m in members]))
        y = float(np.mean([m.y for m in members]))
        new_nodes.append(Node(nid, x, y, "junction"))
        for m in members:
            remap[m.id] = nid
    g.nodes = new_nodes
    edges = []
    for e in g.edges:
        e.a, e.b = remap[e.a], remap[e.b]
        if e.a == e.b and e.length < dist * 3:
            continue  # arête interne au groupe fusionné
        na, nb = g.node(e.a), g.node(e.b)
        e.points = [(na.x, na.y), *e.points[1:-1], (nb.x, nb.y)]
        e.length = _polyline_length(e.points)
        edges.append(e)
    g.edges = edges
    _drop_orphan_nodes(g)


def _absorb_degree2(g: TrackGraph) -> None:
    """Fusionne les paires d'arêtes qui se rejoignent en un nœud de degré 2."""
    adj: dict[int, list[Edge]] = {n.id: [] for n in g.nodes}
    for e in g.edges:
        adj[e.a].append(e)
        if e.b != e.a:
            adj[e.b].append(e)
    removed: set[int] = set()
    alive = {id(e): e for e in g.edges}
    for nid, inc in adj.items():
        if len(inc) != 2:
            continue
        e1, e2 = inc
        if e1.a == e1.b or e2.a == e2.b or id(e1) not in alive or id(e2) not in alive:
            continue
        p1 = e1.points if e1.b == nid else e1.points[::-1]  # … → n
        p2 = e2.points if e2.a == nid else e2.points[::-1]  # n → …
        a = e1.a if e1.b == nid else e1.b
        b = e2.b if e2.a == nid else e2.a
        if a == b == nid:
            continue
        pts = p1 + p2[1:]
        merged = Edge(e1.id, a, b, pts, _polyline_length(pts))
        # mise à jour de l'adjacence pour continuer la chaîne dans la même passe
        for old_e, end in ((e1, a), (e2, b)):
            lst = adj[end]
            lst[:] = [merged if x is old_e else x for x in lst]
            del alive[id(old_e)]
        alive[id(merged)] = merged
        removed.add(nid)
    g.edges = list(alive.values())
    g.nodes = [n for n in g.nodes if n.id not in removed]
