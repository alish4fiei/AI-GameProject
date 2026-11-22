"""
solver.py
Student implementation file for DeepCore search algorithms.

- Breadth-First Search (BFS) is provided below as a complete reference example
  showing how to interact with `env.get_successors(state)` and `SearchMetrics`.
- Implement the remaining required algorithms. Do not change the public
  function signatures or the environment/state modules.
"""

from collections import deque
import heapq
import math
from typing import Callable, Dict, List, Optional, Set, Tuple

from .environment import DeepCoreEnv
from .state import DeepCoreState, SearchMetrics


def _reconstruct_path(
    parent_map: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]],
    goal_state: DeepCoreState,
) -> List[str]:
    """Helper function to trace actions from the initial state to `goal_state`."""
    actions: List[str] = []
    curr: Optional[DeepCoreState] = goal_state
    while curr is not None:
        prev, act = parent_map.get(curr, (None, None))
        if act is not None:
            actions.append(act)
        curr = prev
    actions.reverse()
    return actions


# =============================================================================
# 1. Breadth-First Search (BFS) — [PROVIDED AS REFERENCE EXAMPLE]
# =============================================================================
def breadth_first_search(env: DeepCoreEnv) -> Optional[SearchMetrics]:
    """
    Reference Graph-Search BFS implementation (FIFO queue).
    Minimizes the number of steps, but ignores terrain/payload energy costs!
    """
    metrics = SearchMetrics(algorithm_name="BFS")
    start = env.get_initial_state()
    if env.is_goal(start):
        metrics.solved = True
        return metrics

    frontier = deque([start])
    frontier_set = {start}
    explored: Set[DeepCoreState] = set()
    parent: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]] = {start: (None, None)}
    cost_so_far: Dict[DeepCoreState, int] = {start: 0}

    while frontier:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
        metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))
        state = frontier.popleft()
        frontier_set.discard(state)

        explored.add(state)
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        for action, next_state, step_cost in env.get_successors(state):
            metrics.nodes_generated += 1
            if next_state not in explored and next_state not in frontier_set:
                parent[next_state] = (state, action.value)
                cost_so_far[next_state] = cost_so_far[state] + step_cost
                if env.is_goal(next_state):
                    metrics.solved = True
                    metrics.solution_actions = _reconstruct_path(parent, next_state)
                    metrics.total_cost = cost_so_far[next_state]
                    return metrics
                frontier.append(next_state)
                frontier_set.add(next_state)
                metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))

    metrics.status_message = "No solution found."
    return metrics


# =============================================================================
# 2. Depth-First Search (DFS) & Iterative Deepening Search (IDS) — [STUDENT TODO]
# =============================================================================
def depth_first_search(env: DeepCoreEnv, max_expansions: Optional[int] = None) -> Optional[SearchMetrics]:
    """Depth-First Graph Search using a LIFO Stack with duplicate/cycle checking."""
    metrics = SearchMetrics(algorithm_name="DFS")
    start = env.get_initial_state()
    if env.is_goal(start):
        metrics.solved = True
        return metrics


    frontier: List[DeepCoreState] = [start]
    frontier_set: Set[DeepCoreState] = {start}
    explored: Set[DeepCoreState] = set()
    parent: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]] = {start: (None, None)}
    cost_so_far: Dict[DeepCoreState, int] = {start: 0}

    while frontier:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
        metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))


        if max_expansions is not None and metrics.nodes_expanded >= max_expansions:
            metrics.status_message = "Expansion limit reached."
            return metrics


        state = frontier.pop()
        frontier_set.discard(state)

        explored.add(state)
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        for action, next_state, step_cost in reversed(env.get_successors(state)):
            metrics.nodes_generated += 1
            if next_state not in explored and next_state not in frontier_set:
                parent[next_state] = (state, action.value)
                cost_so_far[next_state] = cost_so_far[state] + step_cost

                if env.is_goal(next_state):
                    metrics.solved = True
                    metrics.solution_actions = _reconstruct_path(parent, next_state)
                    metrics.total_cost = cost_so_far[next_state]
                    return metrics

                frontier.append(next_state)
                frontier_set.add(next_state)
                metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
                metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))

    metrics.status_message = "No solution found."
    return metrics
def iterative_deepening_search(
    env: DeepCoreEnv,
    max_depth: Optional[int] = None,
    max_expansions: Optional[int] = None,
) -> Optional[SearchMetrics]:

    metrics = SearchMetrics(algorithm_name="IDS")
    start = env.get_initial_state()
    if env.is_goal(start):
        metrics.solved = True
        return metrics

    min_depth: Dict[DeepCoreState, int] = {start: 0}
    dead_states: Set[DeepCoreState] = set()
    limit = 0
    cutoff_occurred = False

    while max_depth is None or limit <= max_depth:
        visited_in_iter: Set[DeepCoreState] = {start}
        cutoff_occurred = False

        stack: List[list] = [[start, 0, 0, None, None, 0, False]]

        while stack:
            metrics.max_frontier_size = max(metrics.max_frontier_size, len(stack))
            metrics.max_memory_size = max(metrics.max_memory_size, len(stack) + len(min_depth))
            frame = stack[-1]
            state, depth, cost, _, succs, idx, _ = frame

            if succs is None:
                if depth == limit:
                    cutoff_occurred = True
                    frame[6] = True
                    stack.pop()
                    if stack:
                        stack[-1][6] = True
                    continue
                if max_expansions is not None and metrics.nodes_expanded >= max_expansions:
                    metrics.status_message = "Expansion limit reached."
                    return metrics
                metrics.nodes_expanded += 1
                metrics.explored_order.append(state.bot_pos)
                succs = env.get_successors(state)
                metrics.nodes_generated += len(succs)
                frame[4] = succs

            if idx < len(succs):
                action, next_state, step_cost = succs[idx]
                frame[5] = idx + 1
                next_depth = depth + 1

                if next_state in min_depth:
                    if next_depth > min_depth[next_state] or next_state in dead_states:
                        continue
                    if next_state in visited_in_iter:
                        frame[6] = True
                        continue
                else:
                    min_depth[next_state] = next_depth

                if env.is_goal(next_state):
                    actions = [f[3] for f in stack[1:]] + [action.value]
                    metrics.solved = True
                    metrics.solution_actions = actions
                    metrics.total_cost = cost + step_cost
                    return metrics

                visited_in_iter.add(next_state)
                if next_depth < limit:
                    stack.append([next_state, next_depth, cost + step_cost, action.value, None, 0, False])
                else:
                    cutoff_occurred = True
                    frame[6] = True
            else:
                has_live_child = frame[6]
                stack.pop()
                if has_live_child:
                    if stack:
                        stack[-1][6] = True
                else:
                    dead_states.add(state)

        if not cutoff_occurred:
            metrics.status_message = "No solution found."
            return metrics
        limit += 1

    metrics.status_message = "Depth limit reached." if cutoff_occurred else "No solution found."
    return metrics

# =============================================================================
# 3. Bidirectional Search — [STUDENT TODO]
# =============================================================================
def bidirectional_search(env: DeepCoreEnv) -> Optional[SearchMetrics]:


    metrics = SearchMetrics(algorithm_name="Bidirectional")
    start = env.get_initial_state()
    if env.is_goal(start):
        metrics.solved = True
        return metrics


    fwd_queue: deque[DeepCoreState] = deque([start])
    fwd_parent: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]] = {start: (None, None)}
    fwd_depth: Dict[DeepCoreState, int] = {start: 0}
    fwd_cost: Dict[DeepCoreState, int] = {start: 0}


    chasm_boulder_diff = len(start.chasms) - len(start.boulders)
    bwd_queue: deque[DeepCoreState] = deque()
    bwd_next: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]] = {}
    bwd_depth: Dict[DeepCoreState, int] = {}
    bwd_cost: Dict[DeepCoreState, int] = {}

    for g_state in env.get_goal_states():

        if (
            len(g_state.chasms) - len(g_state.boulders) == chasm_boulder_diff
            and len(g_state.boulders) <= len(start.boulders)
            and len(g_state.chasms) <= len(start.chasms)
            and g_state.rubble.issubset(start.rubble)
        ):
            if g_state not in bwd_depth:
                bwd_queue.append(g_state)
                bwd_next[g_state] = (None, None)
                bwd_depth[g_state] = 0
                bwd_cost[g_state] = 0

    def _build_meeting_solution(meet_state: DeepCoreState) -> SearchMetrics:

        fwd_actions = _reconstruct_path(fwd_parent, meet_state)
        bwd_actions: List[str] = []
        curr: Optional[DeepCoreState] = meet_state
        while curr is not None:
            nxt, act = bwd_next.get(curr, (None, None))
            if act is not None:
                bwd_actions.append(act)
            curr = nxt

        metrics.solved = True
        metrics.solution_actions = fwd_actions + bwd_actions
        metrics.total_cost = fwd_cost[meet_state] + bwd_cost[meet_state]
        return metrics

    while fwd_queue:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(fwd_queue) + len(bwd_queue))
        metrics.max_memory_size = max(metrics.max_memory_size, len(fwd_depth) + len(bwd_depth))

        if not bwd_queue or len(fwd_queue) <= len(bwd_queue):
            layer_size = len(fwd_queue)
            for _ in range(layer_size):
                state = fwd_queue.popleft()
                metrics.nodes_expanded += 1
                metrics.explored_order.append(state.bot_pos)

                for action, next_state, step_cost in env.get_successors(state):
                    metrics.nodes_generated += 1
                    if next_state not in fwd_depth:
                        fwd_parent[next_state] = (state, action.value)
                        fwd_depth[next_state] = fwd_depth[state] + 1
                        fwd_cost[next_state] = fwd_cost[state] + step_cost

                        if env.is_goal(next_state):
                            metrics.solved = True
                            metrics.solution_actions = _reconstruct_path(fwd_parent, next_state)
                            metrics.total_cost = fwd_cost[next_state]
                            return metrics

                        if next_state in bwd_depth:
                            return _build_meeting_solution(next_state)

                        fwd_queue.append(next_state)
                        metrics.max_frontier_size = max(metrics.max_frontier_size, len(fwd_queue) + len(bwd_queue))
                        metrics.max_memory_size = max(metrics.max_memory_size, len(fwd_depth) + len(bwd_depth))
        else:
            layer_size = len(bwd_queue)
            for _ in range(layer_size):
                state = bwd_queue.popleft()
                metrics.nodes_expanded += 1
                metrics.explored_order.append(state.bot_pos)

                for action, prev_state, step_cost in env.get_predecessors(state):
                    metrics.nodes_generated += 1
                    if prev_state not in bwd_depth:
                        bwd_next[prev_state] = (state, action.value)
                        bwd_depth[prev_state] = bwd_depth[state] + 1
                        bwd_cost[prev_state] = bwd_cost[state] + step_cost

                        if prev_state in fwd_depth:
                            return _build_meeting_solution(prev_state)

                        bwd_queue.append(prev_state)
                        metrics.max_frontier_size = max(metrics.max_frontier_size, len(fwd_queue) + len(bwd_queue))
                        metrics.max_memory_size = max(metrics.max_memory_size, len(fwd_depth) + len(bwd_depth))

    metrics.status_message = "No solution found."
    return metrics


# =============================================================================
# 4. Uniform Cost Search (UCS) — [STUDENT TODO]
# =============================================================================
def uniform_cost_search(env: DeepCoreEnv) -> Optional[SearchMetrics]:

    metrics = SearchMetrics(algorithm_name="UCS")
    start = env.get_initial_state()
    if env.is_goal(start):
        metrics.solved = True
        return metrics

    counter = 0

    frontier: List[Tuple[int, int, DeepCoreState]] = [(0, counter, start)]
    best_g: Dict[DeepCoreState, int] = {start: 0}
    parent: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]] = {start: (None, None)}
    explored: Set[DeepCoreState] = set()

    while frontier:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
        metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))


        g_cost, _, state = heapq.heappop(frontier)


        if state in explored:
            continue

        explored.add(state)
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)


        if env.is_goal(state):
            metrics.solved = True
            metrics.solution_actions = _reconstruct_path(parent, state)
            metrics.total_cost = g_cost
            return metrics

        for action, next_state, step_cost in env.get_successors(state):
            metrics.nodes_generated += 1
            new_g = g_cost + step_cost


            if next_state not in explored and new_g < best_g.get(next_state, math.inf):
                best_g[next_state] = new_g
                parent[next_state] = (state, action.value)
                counter += 1
                heapq.heappush(frontier, (new_g, counter, next_state))
                metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
                metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))

    metrics.status_message = "No solution found."
    return metrics

# =============================================================================
# 5. Greedy Best-First Search (GBFS) — [STUDENT TODO]
# =============================================================================
def greedy_best_first_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float]
) -> Optional[SearchMetrics]:

    metrics = SearchMetrics(algorithm_name="GBFS")

    start = env.get_initial_state()

    if env.is_goal(start):
        metrics.solved = True
        return metrics

    counter = 0

    frontier = [
        (heuristic(start, env), counter, 0, start)
    ]

    best_g = {start: 0}

    expanded_g = {}

    parent = {start: (None, None)}

    while frontier:

        metrics.max_frontier_size = max(
            metrics.max_frontier_size, len(frontier)
        )

        state_h, _, g_cost, state = heapq.heappop(frontier)

        if g_cost != best_g.get(state):
            continue

        if expanded_g.get(state, math.inf) <= g_cost:
            continue

        expanded_g[state] = g_cost

        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        if env.is_goal(state):
            metrics.solved = True
            metrics.solution_actions = _reconstruct_path(parent, state)
            metrics.total_cost = g_cost
            return metrics

        for action, next_state, step_cost in env.get_successors(state):

            metrics.nodes_generated += 1

            new_g = g_cost + step_cost

            if new_g < best_g.get(next_state, math.inf):

                best_g[next_state] = new_g

                parent[next_state] = (state, action.value)

                counter += 1

                h_value = heuristic(next_state, env)

                heapq.heappush(
                    frontier,
                    (h_value, counter, new_g, next_state)
                )

        metrics.max_memory_size = max(
            metrics.max_memory_size,
            len(frontier) + len(expanded_g)
        )

    metrics.status_message = "No solution found."

    return metrics


# =============================================================================
# 6. A* Search — [STUDENT TODO]
# =============================================================================
def a_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float]
) -> Optional[SearchMetrics]:

    metrics = SearchMetrics(algorithm_name="A*")

    start = env.get_initial_state()

    if env.is_goal(start):
        metrics.solved = True
        return metrics

    counter = 0

    frontier = [
        (heuristic(start, env), counter, 0, start)
    ]

    best_g = {start: 0}

    expanded_g = {}

    parent = {start: (None, None)}

    while frontier:

        metrics.max_frontier_size = max(
            metrics.max_frontier_size, len(frontier)
        )

        f_cost, _, g_cost, state = heapq.heappop(frontier)

        if g_cost != best_g.get(state):
            continue

        if expanded_g.get(state, math.inf) <= g_cost:
            continue

        expanded_g[state] = g_cost

        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        if env.is_goal(state):

            metrics.solved = True

            metrics.solution_actions = _reconstruct_path(parent, state)

            metrics.total_cost = g_cost

            return metrics

        for action, next_state, step_cost in env.get_successors(state):

            metrics.nodes_generated += 1

            new_g = g_cost + step_cost

            if new_g < best_g.get(next_state, math.inf):

                best_g[next_state] = new_g

                parent[next_state] = (state, action.value)

                h_value = heuristic(next_state, env)

                new_f = new_g + h_value

                counter += 1

                heapq.heappush(
                    frontier,
                    (new_f, counter, new_g, next_state)
                )

        metrics.max_memory_size = max(
            metrics.max_memory_size,
            len(frontier) + len(expanded_g)
        )

    metrics.status_message = "No solution found."

    return metrics


# =============================================================================
# BONUS 1: Weighted A* Search — [STUDENT BONUS TODO]
# =============================================================================
def weighted_a_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
    weight: float = 1.5
) -> Optional[SearchMetrics]:

    if not math.isfinite(weight) or weight < 1.0:
        raise ValueError("Weight must be finite and >= 1.")

    metrics = SearchMetrics(algorithm_name="Weighted A*")

    start = env.get_initial_state()

    if env.is_goal(start):
        metrics.solved = True
        return metrics

    counter = 0

    frontier = [
        (weight * heuristic(start, env), counter, 0, start)
    ]

    best_g = {start: 0}

    expanded_g = {}

    parent = {start: (None, None)}

    while frontier:

        metrics.max_frontier_size = max(
            metrics.max_frontier_size, len(frontier)
        )

        f_cost, _, g_cost, state = heapq.heappop(frontier)

        if g_cost != best_g.get(state):
            continue

        if expanded_g.get(state, math.inf) <= g_cost:
            continue

        expanded_g[state] = g_cost

        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        if env.is_goal(state):

            metrics.solved = True

            metrics.solution_actions = _reconstruct_path(parent, state)

            metrics.total_cost = g_cost

            return metrics

        for action, next_state, step_cost in env.get_successors(state):

            metrics.nodes_generated += 1

            new_g = g_cost + step_cost

            if new_g < best_g.get(next_state, math.inf):

                best_g[next_state] = new_g

                parent[next_state] = (state, action.value)

                h_value = heuristic(next_state, env)

                new_f = new_g + weight * h_value

                counter += 1

                heapq.heappush(
                    frontier,
                    (new_f, counter, new_g, next_state)
                )

        metrics.max_memory_size = max(
            metrics.max_memory_size,
            len(frontier) + len(expanded_g)
        )

    metrics.status_message = "No solution found."

    return metrics


# =============================================================================
# BONUS 2: Iterative Deepening A* (IDA*) — [STUDENT BONUS TODO]
# =============================================================================
def ida_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
    max_expansions: Optional[int] = None
) -> Optional[SearchMetrics]:

    metrics = SearchMetrics(algorithm_name="IDA*")

    start = env.get_initial_state()

    if env.is_goal(start):
        metrics.solved = True
        return metrics

    heuristic_cache = {}

    def h(state):

        if state not in heuristic_cache:

            heuristic_cache[state] = float(
                heuristic(state, env)
            )

        return heuristic_cache[state]

    threshold = h(start)

    while True:

        best_g = {}

        path_states = {start}
        path_actions = []

        solution_actions = None
        solution_cost = 0

        aborted = False

        def dfs(state, g_cost):

            nonlocal solution_actions
            nonlocal solution_cost
            nonlocal aborted

            f_cost = g_cost + h(state)

            if f_cost > threshold:
                return f_cost

            if env.is_goal(state):

                solution_actions = path_actions.copy()
                solution_cost = g_cost

                return True

            previous_g = best_g.get(state)

            if previous_g is not None and g_cost >= previous_g:
                return math.inf

            best_g[state] = g_cost
            if (
                max_expansions is not None
                and metrics.nodes_expanded >= max_expansions
            ):

                aborted = True
                return math.inf

            metrics.nodes_expanded += 1

            metrics.explored_order.append(state.bot_pos)

            successors = env.get_successors(state)

            metrics.nodes_generated += len(successors)

            min_exceeded = math.inf

            for action, next_state, step_cost in successors:

                if next_state in path_states:
                    continue

                path_states.add(next_state)
                path_actions.append(action.value)

                result = dfs(
                    next_state,
                    g_cost + step_cost
                )

                if result is True:
                    return True
                path_actions.pop()
                path_states.remove(next_state)

                if aborted:
                    return math.inf

                min_exceeded = min(
                    min_exceeded,
                    result
                )

                metrics.max_frontier_size = max(
                    metrics.max_frontier_size,
                    len(path_states)
                )

                metrics.max_memory_size = max(
                    metrics.max_memory_size,
                    len(best_g) + len(path_states)
                )

            return min_exceeded
        result = dfs(start, 0)

        if result is True:

            metrics.solved = True
            metrics.solution_actions = solution_actions
            metrics.total_cost = solution_cost

            return metrics

        if aborted:

            metrics.status_message = "Expansion limit reached."

            return metrics

        if result == math.inf:

            metrics.status_message = "No solution found."

            return metrics
        threshold = result


# =============================================================================
# BONUS 3: Simplified Memory-Bounded A* (SMA*) — [STUDENT BONUS TODO]
# =============================================================================
def sma_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
    memory_limit: Optional[int] = None,
) -> Optional[SearchMetrics]:

    metrics = SearchMetrics(algorithm_name="SMA*")

    if memory_limit is None:
        memory_limit = 10000

    if not isinstance(memory_limit, int) or isinstance(memory_limit, bool):
        raise ValueError("Memory limit must be an integer.")

    if memory_limit < 2:
        raise ValueError("Memory limit must be at least 2.")

    start = env.get_initial_state()

    if env.is_goal(start):
        metrics.solved = True
        return metrics

    nodes = {}

    next_id = 1

    root_h = float(heuristic(start, env))

    nodes[0] = {
        "state": start,
        "g": 0,
        "base_f": root_h,
        "f": root_h,
        "parent": None,
        "action": None,
        "depth": 0,
        "children": set(),
        "forgotten": {},
        "expanded": False,
        "expanding": False,
        "dead": False,
    }

    metrics.max_memory_size = 1
    metrics.max_frontier_size = 1

    def is_ancestor(node_id, state):

        current = node_id

        while current is not None:

            if nodes[current]["state"] == state:
                return True

            current = nodes[current]["parent"]

        return False

    def get_leaves():

        result = []

        for node_id, node in nodes.items():

            if node["dead"] or node["expanding"]:
                continue

            if node["children"]:
                continue

            if not node["expanded"] or node["forgotten"]:
                result.append(node_id)

        return result

    def refresh_up(node_id):

        current = node_id

        while current is not None and current in nodes:

            node = nodes[current]

            estimates = [
                nodes[child]["f"]
                for child in node["children"]
                if child in nodes
            ]

            estimates.extend(node["forgotten"].values())

            if estimates:

                node["f"] = max(
                    node["f"],
                    node["base_f"],
                    min(estimates)
                )

                current = node["parent"]

            elif node["expanded"] and not node["expanding"]:

                node["dead"] = True

                parent_id = node["parent"]

                if parent_id is None:
                    return

                nodes[parent_id]["children"].discard(current)

                del nodes[current]

                current = parent_id

            else:
                return

    def remove_leaf(node_id, remember=True):

        node = nodes[node_id]

        parent_id = node["parent"]

        if parent_id is None:
            node["dead"] = True
            return

        parent = nodes[parent_id]

        parent["children"].discard(node_id)

        if remember:

            action = node["action"]

            parent["forgotten"][action] = max(
                parent["forgotten"].get(action, float("-inf")),
                node["f"]
            )

        del nodes[node_id]

        refresh_up(parent_id)

    def enforce_memory_limit():

        while len(nodes) > memory_limit:

            leaves = get_leaves()

            if not leaves:
                break

            worst_leaf = max(
                leaves,
                key=lambda n: (
                    nodes[n]["f"],
                    -nodes[n]["depth"],
                    n
                )
            )

            remove_leaf(worst_leaf, remember=True)

        metrics.max_memory_size = max(
            metrics.max_memory_size,
            len(nodes)
        )

    while True:

        leaves = get_leaves()

        if not leaves:
            break

        metrics.max_frontier_size = max(
            metrics.max_frontier_size,
            len(leaves)
        )

        metrics.max_memory_size = max(
            metrics.max_memory_size,
            len(nodes)
        )

        best_node_id = min(
            leaves,
            key=lambda n: (
                nodes[n]["f"],
                -nodes[n]["depth"],
                n
            )
        )

        node = nodes[best_node_id]

        state = node["state"]

        if env.is_goal(state):

            actions = []

            current = best_node_id

            while nodes[current]["parent"] is not None:

                actions.append(
                    nodes[current]["action"]
                )

                current = nodes[current]["parent"]

            actions.reverse()

            metrics.solved = True

            metrics.solution_actions = actions

            metrics.total_cost = node["g"]

            return metrics

        if len(nodes) >= memory_limit and len(leaves) == 1:

            remove_leaf(best_node_id, remember=False)

            continue

        if not node["expanded"]:

            node["expanded"] = True

            node["expanding"] = True

            successors = env.get_successors(state)

            metrics.nodes_expanded += 1

            metrics.nodes_generated += len(successors)

            candidates = []

            for action, next_state, step_cost in successors:

                if is_ancestor(best_node_id, next_state):
                    continue

                new_g = node["g"] + step_cost

                new_h = float(
                    heuristic(next_state, env)
                )

                new_f = max(
                    node["f"],
                    new_g + new_h
                )

                candidates.append(
                    (new_f, action.value, new_g, next_state)
                )

            candidates.sort(key=lambda x: (x[0], x[1]))

            for new_f, action_name, new_g, next_state in candidates:

                child_id = next_id
                next_id += 1

                nodes[child_id] = {
                    "state": next_state,
                    "g": new_g,
                    "base_f": new_g + float(
                        heuristic(next_state, env)
                    ),
                    "f": new_f,
                    "parent": best_node_id,
                    "action": action_name,
                    "depth": node["depth"] + 1,
                    "children": set(),
                    "forgotten": {},
                    "expanded": False,
                    "expanding": False,
                    "dead": False,
                }

                node["children"].add(child_id)

                enforce_memory_limit()

            if best_node_id in nodes:

                nodes[best_node_id]["expanding"] = False

                refresh_up(best_node_id)

        else:

            action_name, saved_f = min(
                node["forgotten"].items(),
                key=lambda item: (item[1], item[0])
            )

            del node["forgotten"][action_name]

            successors = env.get_successors(state)

            metrics.nodes_expanded += 1

            metrics.nodes_generated += len(successors)

            restored = None

            for action, next_state, step_cost in successors:

                if action.value == action_name:

                    if not is_ancestor(best_node_id, next_state):

                        restored = (
                            action,
                            next_state,
                            step_cost
                        )

                    break

            if restored is None:

                refresh_up(best_node_id)

                continue

            action, next_state, step_cost = restored

            new_g = node["g"] + step_cost

            new_h = float(
                heuristic(next_state, env)
            )

            new_f = max(
                node["f"],
                saved_f,
                new_g + new_h
            )

            child_id = next_id
            next_id += 1

            node["expanding"] = True

            nodes[child_id] = {
                "state": next_state,
                "g": new_g,
                "base_f": new_g + new_h,
                "f": new_f,
                "parent": best_node_id,
                "action": action.value,
                "depth": node["depth"] + 1,
                "children": set(),
                "forgotten": {},
                "expanded": False,
                "expanding": False,
                "dead": False,
            }

            node["children"].add(child_id)

            enforce_memory_limit()

            if best_node_id in nodes:

                nodes[best_node_id]["expanding"] = False

                refresh_up(best_node_id)

    metrics.status_message = "No solution found within the memory limit."

    return metrics