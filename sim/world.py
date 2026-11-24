"""Digital Twin Environment Graph (DTEG) of the hospital floor.

The floor (80 m x 50 m, 0.5 m cells) is described by
  * spatial nodes : semantic zones (Table 2 of the manuscript)
  * object nodes  : walls, doors, exits
  * context nodes : per-zone occupancy, alarm state (updated online)
Geodesic distance fields to every zone and exit are pre-computed with Dijkstra
on the 8-connected free-cell graph; following their descent direction is the
optimal grid path (identical to an A* search with an admissible heuristic).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import distance_transform_edt
from scipy.sparse.csgraph import dijkstra

RES = 0.5
W_M, H_M = 80.0, 50.0

ZONES = ["corridor", "reception", "waiting_A", "waiting_B", "consult_1", "consult_2", "consult_3",
         "consult_4", "emergency", "icu", "admin", "lounge", "exit_main", "exit_east"]
Z = {n: i for i, n in enumerate(ZONES)}
ROOM_ZONES = list(range(1, 12))
EXITS = [Z["exit_main"], Z["exit_east"]]
# rectangles (x0, x1, y0, y1) in metres, interior of each zone
RECTS = {
    "reception": (0.5, 12, 15.5, 34.5),
    "waiting_A": (12.5, 30, 30.5, 49.5), "waiting_B": (12.5, 30, 0.5, 19.5),
    "consult_1": (30.5, 40, 30.5, 49.5), "consult_2": (40.5, 50, 30.5, 49.5),
    "consult_3": (30.5, 40, 0.5, 19.5), "consult_4": (40.5, 50, 0.5, 19.5),
    "emergency": (50.5, 65, 30.5, 49.5), "icu": (50.5, 65, 0.5, 19.5),
    "admin": (65.5, 79.5, 30.5, 49.5), "lounge": (65.5, 79.5, 0.5, 19.5),
}


@dataclass
class World:
    walls: np.ndarray                    # [H, W] bool
    zone: np.ndarray                     # [H, W] int
    dist: np.ndarray                     # [Zn, H, W] geodesic distance (m), inf in walls
    dirs: np.ndarray                     # [Zn, H, W, 2] unit descent direction
    wall_dist: np.ndarray                # [H, W] distance to nearest wall (m)
    wall_grad: np.ndarray                # [H, W, 2] unit vector pointing away from walls
    rects: dict = field(default_factory=dict)
    hazard_center: tuple = (57.5, 40.0)

    @property
    def shape(self):
        return self.walls.shape


def _cells(x0, x1, y0, y1):
    return slice(int(round(y0 / RES)), int(round(y1 / RES))), slice(int(round(x0 / RES)), int(round(x1 / RES)))


def build_world() -> World:
    H, Wc = int(H_M / RES), int(W_M / RES)
    walls = np.zeros((H, Wc), bool)
    zone = np.zeros((H, Wc), np.int64)
    walls[0, :] = walls[-1, :] = True
    walls[:, 0] = walls[:, -1] = True
    # solid blocks outside reception on the west side
    walls[_cells(0, 12, 0, 15)] = True
    walls[_cells(0, 12, 35, 50)] = True
    # horizontal room walls at y = 20 and y = 30 (x >= 12) and reception walls y = 15 / 35
    walls[_cells(12, 80, 19.5, 20.0)] = True
    walls[_cells(12, 80, 30.0, 30.5)] = True
    walls[_cells(0, 12.5, 15.0, 15.5)] = True
    walls[_cells(0, 12.5, 34.5, 35.0)] = True
    # vertical separators
    for x in (12.0, 30.0, 40.0, 50.0, 65.0):
        walls[_cells(x, x + 0.5, 0, 20)] = True
        walls[_cells(x, x + 0.5, 30, 50)] = True
    # doors (2 m) from each room to the corridor
    for name, (x0, x1, y0, y1) in RECTS.items():
        if name == "reception":
            continue
        cx = 0.5 * (x0 + x1)
        if y0 > 25:
            walls[_cells(cx - 1.0, cx + 1.0, 29.5, 31.0)] = False
        else:
            walls[_cells(cx - 1.0, cx + 1.0, 19.0, 20.5)] = False
    # reception opens fully onto the corridor (x = 12 boundary between y 20..30)
    walls[_cells(12.0, 12.5, 20, 30)] = False
    # exits: west main entrance and east emergency exit (y 23..27)
    walls[_cells(0, 0.5, 23, 27)] = False
    walls[_cells(79.5, 80, 23, 27)] = False
    # zones
    zone[_cells(12, 80, 20, 30)] = Z["corridor"]
    for name, r in RECTS.items():
        zone[_cells(*r)] = Z[name]
    zone[_cells(0, 1.5, 23, 27)] = Z["exit_main"]
    zone[_cells(78.5, 80, 23, 27)] = Z["exit_east"]
    zone[walls] = -1

    free = ~walls
    idx = -np.ones((H, Wc), np.int64)
    idx[free] = np.arange(free.sum())
    rows, cols, w = [], [], []
    for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        ys, xs = np.nonzero(free)
        ny, nx = ys + dy, xs + dx
        ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < Wc)
        ys, xs, ny, nx = ys[ok], xs[ok], ny[ok], nx[ok]
        ok = free[ny, nx]
        if dy and dx:   # no corner cutting
            ok &= free[ys, nx] & free[ny, xs]
        rows.append(idx[ys[ok], xs[ok]]); cols.append(idx[ny[ok], nx[ok]])
        w.append(np.full(ok.sum(), RES * (np.sqrt(2) if dy and dx else 1.0)))
    n = free.sum()
    G = sp.csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(n + 1, n + 1))
    wd = distance_transform_edt(free) * RES
    dist = np.full((len(ZONES), H, Wc), np.inf, np.float32)
    for zi in range(len(ZONES)):
        tgt = (zone == zi) & (wd > 0.75 if zi not in EXITS and zi != 0 else zone == zi)
        if zi == 0:
            tgt = zone == 0
        src = idx[tgt]
        G2 = G.tolil(copy=True)
        G2[n, src] = 1e-6
        d = dijkstra(G2.tocsr(), indices=n)[:n]
        dist[zi][free] = d
    dirs = np.zeros((len(ZONES), H, Wc, 2), np.float32)
    pad = np.pad(dist, ((0, 0), (1, 1), (1, 1)), constant_values=np.inf)
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    best = np.full(dist.shape, np.inf, np.float32)
    for dy, dx in nbrs:
        nb = pad[:, 1 + dy:1 + dy + H, 1 + dx:1 + dx + Wc]
        better = nb < best
        best = np.where(better, nb, best)
        v = np.array([dx, dy], np.float32) / np.hypot(dx, dy)
        dirs = np.where(better[..., None], v, dirs)
    # smooth the 8-direction field with a 3x3 average for less zig-zagging
    sm = np.zeros_like(dirs)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            sm += np.roll(np.roll(dirs, dy, 1), dx, 2)
    nrm = np.linalg.norm(sm, axis=-1, keepdims=True)
    dirs = np.where(nrm > 1e-6, sm / np.maximum(nrm, 1e-6), dirs)
    dirs[:, walls] = 0
    gy, gx = np.gradient(wd)
    g = np.stack([gx, gy], -1)
    g /= np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-6)
    return World(walls, zone, dist, dirs.astype(np.float32), wd.astype(np.float32), g.astype(np.float32), RECTS)


def ascii_map(world: World) -> str:
    ch = {-1: "#", 0: ".", 12: "E", 13: "E"}
    rows = []
    for y in range(world.shape[0] - 1, -1, -2):
        rows.append("".join(ch.get(int(world.zone[y, x]), chr(ord("a") + int(world.zone[y, x]) - 1))
                            for x in range(0, world.shape[1], 2)))
    return "\n".join(rows)


if __name__ == "__main__":
    w = build_world()
    print(ascii_map(w))
    for zi, nme in enumerate(ZONES):
        reach = np.isfinite(w.dist[zi][~w.walls]).mean()
        print(nme, f"reachable={reach:.3f}")
