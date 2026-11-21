"""
heuristics.py
Student heuristic starter file for DeepCore:
  - h_naive_manhattan : provided baseline
  - h1_basic          : required student implementation
  - h2_advanced       : required student implementation
"""

from typing import Dict, List, Tuple
from collections import deque
from .environment import DeepCoreEnv, DIRECTION_DELTAS
from .state import DeepCoreState, Pos
import heapq

# Cache wall-aware shortest path distances per environment instance
_WALL_DIST_CACHE: Dict[int, Dict[Tuple[Pos, Pos], int]] = {}


def _get_wall_aware_distances(env: DeepCoreEnv) -> Dict[Tuple[Pos, Pos], int]:
    """Precomputes all-pairs shortest path steps ignoring dynamic obstacles (only walls # block)."""
    env_id = id(env)
    if env_id in _WALL_DIST_CACHE:
        return _WALL_DIST_CACHE[env_id]

    walkable = [
        (r, c)
        for r in range(env.rows)
        for c in range(env.cols)
        if (r, c) not in env.walls
    ]
    dist_map: Dict[Tuple[Pos, Pos], int] = {}

    for src in walkable:
        q = deque([(src, 0)])
        visited = {src}
        while q:
            curr, d = q.popleft()
            dist_map[(src, curr)] = d
            cr, cc = curr
            for dr, dc in DIRECTION_DELTAS.values():
                nxt = (cr + dr, cc + dc)
                if env.in_bounds(nxt) and nxt not in env.walls and nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, d + 1))

    _WALL_DIST_CACHE[env_id] = dist_map
    return dist_map


def _prim_mst_cost(nodes: List[Pos], dist_fn) -> float:
    """Computes Minimum Spanning Tree weight over `nodes` using `dist_fn(u, v)`."""
    if len(nodes) <= 1:
        return 0.0
    in_mst = {nodes[0]}
    total_weight = 0.0

    while len(in_mst) < len(nodes):
        best_edge = float("inf")
        best_node = None
        for u in in_mst:
            for v in nodes:
                if v not in in_mst:
                    w = dist_fn(u, v)
                    if w < best_edge:
                        best_edge = w
                        best_node = v
        if best_node is None:
            break
        in_mst.add(best_node)
        total_weight += best_edge

    return total_weight


def h_naive_manhattan(state: DeepCoreState, env: DeepCoreEnv) -> float:
    """
    Baseline Naive Heuristic: Manhattan distance to nearest waiting miner (or Elevator if none).
    """
    if env.is_goal(state):
        return 0.0
    r, c = state.bot_pos
    if state.miners:
        return float(min(abs(r - mr) + abs(c - mc) for mr, mc in state.miners))
    er, ec = env.elevator_pos
    return float(abs(r - er) + abs(c - ec))


def h1_basic(state: DeepCoreState, env: DeepCoreEnv) -> float:

    if env.is_goal(state):
        return 0.0

    distances = _get_wall_aware_distances(env)

    terminals = list(dict.fromkeys(
        [state.bot_pos, *state.miners, env.elevator_pos]
    ))

    mst_cost = _prim_mst_cost(
        terminals,
        lambda u, v: distances.get((u, v), float("inf"))
    )

    surcharge_distances = {}

    for source in terminals:

        dist = {source: 0}
        queue = deque([source])

        while queue:

            current = queue.popleft()

            cr, cc = current

            for dr, dc in DIRECTION_DELTAS.values():

                nxt = (cr + dr, cc + dc)

                if not env.in_bounds(nxt) or nxt in env.walls:
                    continue


                step_cost = 0 if nxt in state.rubble else 1

                new_dist = dist[current] + step_cost

                if new_dist < dist.get(nxt, float("inf")):

                    dist[nxt] = new_dist

                    if step_cost == 0:
                        queue.appendleft(nxt)
                    else:
                        queue.append(nxt)

        surcharge_distances[source] = dist

    def surcharge_distance(u, v):

        forward = surcharge_distances[u].get(
            v, float("inf")
        )

        backward = surcharge_distances[v].get(
            u, float("inf")
        )

        return min(forward, backward)

    current_payload_steps = _prim_mst_cost(
        terminals,
        surcharge_distance
    )


    future_payload_steps = sum(
        surcharge_distances[miner].get(
            env.elevator_pos, float("inf")
        )
        for miner in state.miners
    )

    payload_cost = env.PAYLOAD_SURCHARGE * (
        state.carried * current_payload_steps
        + future_payload_steps
    )

    return mst_cost + payload_cost


def _is_corner_deadlock(pos: Pos, env: DeepCoreEnv) -> bool:
    """Returns True if a boulder at `pos` is wedged against perpendicular rock walls (#)."""
    r, c = pos
    n_wall = (r - 1, c) in env.walls
    s_wall = (r + 1, c) in env.walls
    e_wall = (r, c + 1) in env.walls
    w_wall = (r, c - 1) in env.walls
    return (n_wall or s_wall) and (e_wall or w_wall)


def h2_advanced(state: DeepCoreState, env: DeepCoreEnv) -> float:
    
    if env.is_goal(state):
        return 0.0

    terminals = list(dict.fromkeys(
        [state.bot_pos, *state.miners, env.elevator_pos]
    ))

    h1_value = h1_basic(state, env)

    wall_distances = _get_wall_aware_distances(env)

    wall_mst = _prim_mst_cost(
        terminals,
        lambda u, v: wall_distances.get(
            (u, v), float("inf")
        )
    )

    payload_cost = h1_value - wall_mst

    walkable = [
        (r, c)
        for r in range(env.rows)
        for c in range(env.cols)
        if (r, c) not in env.walls
    ]

    weighted_distances = {}

    for source in terminals:

        dist = {source: 0}

        unvisited = set(walkable)

        while unvisited:
            current = min(
                unvisited,
                key=lambda pos: dist.get(pos, float("inf"))
            )

            current_cost = dist.get(current, float("inf"))

            if current_cost == float("inf"):
                break

            unvisited.remove(current)

            cr, cc = current

            for dr, dc in DIRECTION_DELTAS.values():

                nxt = (cr + dr, cc + dc)

                if not env.in_bounds(nxt):
                    continue

                if nxt in env.walls:
                    continue

                if nxt not in unvisited:
                    continue

                if nxt in env.sludge:
                    step_cost = env.COST_SLUDGE
                else:
                    step_cost = env.COST_CLEAR_FLOOR

                new_cost = current_cost + step_cost

                if new_cost < dist.get(nxt, float("inf")):
                    dist[nxt] = new_cost

        weighted_distances[source] = dist

    def terrain_distance(u, v):

        forward = weighted_distances[u].get(
            v, float("inf")
        )

        backward = weighted_distances[v].get(
            u, float("inf")
        )

        return min(forward, backward)

    weighted_mst = _prim_mst_cost(
        terminals,
        terrain_distance
    )

    h2_value = weighted_mst + payload_cost
    return max(h1_value, h2_value)
