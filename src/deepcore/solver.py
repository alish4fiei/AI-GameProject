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
# 5. Greedy Best-First Search (GBFS)
# =============================================================================
def greedy_best_first_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
) -> Optional[SearchMetrics]:
    """Greedy Best-First Search ordering the priority queue strictly by h(n)."""
    metrics = SearchMetrics(algorithm_name="GBFS")
    start = env.get_initial_state()
    if env.is_goal(start):
        metrics.solved = True
        return metrics

    h_start = float(heuristic(start, env))
    if math.isinf(h_start):
        metrics.status_message = "No solution found."
        return metrics

    counter = 0
    frontier: List[Tuple[float, int, int, DeepCoreState]] = [(h_start, counter, 0, start)]
    seen: Set[DeepCoreState] = {start}
    explored: Set[DeepCoreState] = set()
    parent: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]] = {start: (None, None)}

    while frontier:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
        metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))

        _, _, g_cost, state = heapq.heappop(frontier)
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
            if next_state not in seen:
                h_value = float(heuristic(next_state, env))
                if math.isinf(h_value):
                    continue
                seen.add(next_state)
                parent[next_state] = (state, action.value)
                counter += 1
                heapq.heappush(frontier, (h_value, counter, g_cost + step_cost, next_state))
                metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
                metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))

    metrics.status_message = "No solution found."
    return metrics


# =============================================================================
# 6. A* Search
# =============================================================================
def a_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
) -> Optional[SearchMetrics]:
    """A* Graph Search ordering the priority queue by f(n) = g(n) + h(n) with reopening."""
    metrics = SearchMetrics(algorithm_name="A*")
    start = env.get_initial_state()
    if env.is_goal(start):
        metrics.solved = True
        return metrics

    h_start = float(heuristic(start, env))
    if math.isinf(h_start):
        metrics.status_message = "No solution found."
        return metrics

    counter = 0
    frontier: List[Tuple[float, float, int, int, DeepCoreState]] = [(h_start, h_start, counter, 0, start)]
    best_g: Dict[DeepCoreState, int] = {start: 0}
    explored: Set[DeepCoreState] = set()
    parent: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]] = {start: (None, None)}

    while frontier:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
        metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))

        _, _, _, g_cost, state = heapq.heappop(frontier)
        if g_cost > best_g.get(state, math.inf):
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
            if new_g < best_g.get(next_state, math.inf):
                h_value = float(heuristic(next_state, env))
                if math.isinf(h_value):
                    continue
                best_g[next_state] = new_g
                parent[next_state] = (state, action.value)
                explored.discard(next_state)
                counter += 1
                heapq.heappush(frontier, (new_g + h_value, h_value, counter, new_g, next_state))
                metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
                metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))

    metrics.status_message = "No solution found."
    return metrics


# =============================================================================
# BONUS 1: Weighted A* Search
# =============================================================================
def weighted_a_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
    weight: float = 1.5,
) -> Optional[SearchMetrics]:
    """Weighted A* ordering the frontier by f(n) = g(n) + W * h(n)."""
    if not math.isfinite(weight) or weight < 1.0:
        raise ValueError("Weight must be finite and >= 1.")

    metrics = SearchMetrics(algorithm_name="Weighted A*")
    start = env.get_initial_state()
    if env.is_goal(start):
        metrics.solved = True
        return metrics

    h_start = float(heuristic(start, env))
    if math.isinf(h_start):
        metrics.status_message = "No solution found."
        return metrics

    counter = 0
    frontier: List[Tuple[float, float, int, int, DeepCoreState]] = [
        (weight * h_start, h_start, counter, 0, start)
    ]
    best_g: Dict[DeepCoreState, int] = {start: 0}
    explored: Set[DeepCoreState] = set()
    parent: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]] = {start: (None, None)}

    while frontier:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
        metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))

        _, _, _, g_cost, state = heapq.heappop(frontier)
        if g_cost > best_g.get(state, math.inf):
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
            if new_g < best_g.get(next_state, math.inf):
                h_value = float(heuristic(next_state, env))
                if math.isinf(h_value):
                    continue
                best_g[next_state] = new_g
                parent[next_state] = (state, action.value)
                explored.discard(next_state)
                counter += 1
                heapq.heappush(
                    frontier,
                    (new_g + weight * h_value, h_value, counter, new_g, next_state),
                )
                metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
                metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))

    metrics.status_message = "No solution found."
    return metrics


# =============================================================================
# BONUS 2: Iterative Deepening A* (IDA*)
# =============================================================================
def ida_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
    max_expansions: Optional[int] = None,
) -> Optional[SearchMetrics]:
    """Iterative Deepening A* (IDA*) using f-cost threshold contours."""
    metrics = SearchMetrics(algorithm_name="IDA*")
    start = env.get_initial_state()
    if env.is_goal(start):
        metrics.solved = True
        return metrics

    heuristic_cache: Dict[DeepCoreState, float] = {}

    def h(state: DeepCoreState) -> float:
        if state not in heuristic_cache:
            heuristic_cache[state] = float(heuristic(state, env))
        return heuristic_cache[state]

    threshold = h(start)
    if math.isinf(threshold):
        metrics.status_message = "No solution found."
        return metrics

    metrics.max_frontier_size = 1
    metrics.max_memory_size = 1

    while True:
        best_g: Dict[DeepCoreState, int] = {}
        path_states: Set[DeepCoreState] = {start}
        path_actions: List[str] = []
        solution_actions: Optional[List[str]] = None
        solution_cost = 0
        aborted = False

        def dfs(state: DeepCoreState, g_cost: int):
            nonlocal solution_actions, solution_cost, aborted

            metrics.max_frontier_size = max(metrics.max_frontier_size, len(path_states))
            metrics.max_memory_size = max(metrics.max_memory_size, len(best_g) + len(path_states))

            h_val = h(state)
            if math.isinf(h_val):
                return math.inf

            f_cost = g_cost + h_val
            if f_cost > threshold:
                return f_cost

            if env.is_goal(state):
                solution_actions = path_actions.copy()
                solution_cost = g_cost
                return True

            previous_g = best_g.get(state)
            if previous_g is not None and g_cost >= previous_g:
                return math.inf

            if max_expansions is not None and metrics.nodes_expanded >= max_expansions:
                aborted = True
                return math.inf

            best_g[state] = g_cost
            metrics.nodes_expanded += 1
            metrics.explored_order.append(state.bot_pos)

            successors = env.get_successors(state)
            metrics.nodes_generated += len(successors)

            min_exceeded = math.inf
            for action, next_state, step_cost in successors:
                if next_state in path_states:
                    continue
                if math.isinf(h(next_state)):
                    continue

                path_states.add(next_state)
                path_actions.append(action.value)
                result = dfs(next_state, g_cost + step_cost)

                if result is True:
                    return True

                path_actions.pop()
                path_states.remove(next_state)

                if aborted:
                    return math.inf

                if result < min_exceeded:
                    min_exceeded = result

            return min_exceeded

        result = dfs(start, 0)

        if result is True:
            metrics.solved = True
            metrics.solution_actions = solution_actions or []
            metrics.total_cost = solution_cost
            return metrics

        if aborted:
            metrics.status_message = "Expansion limit reached."
            return metrics

        if math.isinf(result):
            metrics.status_message = "No solution found."
            return metrics

        threshold = result


# =============================================================================
# BONUS 3: Simplified Memory-Bounded A* (SMA*)
# =============================================================================
def sma_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
    memory_limit: Optional[int] = None,
) -> Optional[SearchMetrics]:
    """Simplified Memory-Bounded A* (SMA*) with O(log N) heaps and backed-up f-values."""
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

    root_h = float(heuristic(start, env))
    if math.isinf(root_h):
        metrics.status_message = "No solution found."
        return metrics

    nodes: Dict[int, dict] = {
        0: {
            "state": start,
            "g": 0,
            "base_f": root_h,
            "f": root_h,
            "parent": None,
            "action": None,
            "depth": 0,
            "children": {},       # action_name -> child_id
            "forgotten": {},      # action_name -> backed-up f
            "expanded": False,
            "expanding": False,
            "dead": False,
        }
    }
    state_nodes: Dict[DeepCoreState, Set[int]] = {start: {0}}
    in_frontier: Set[int] = {0}
    next_id = 1

    # Min-heap برای بهترین گره قابل بسط: (priority_f, -depth, nid)
    expand_heap: List[Tuple[float, int, int]] = [(root_h, 0, 0)]
    # Max-heap (با منفی کردن f) برای کم‌امیدترین برگ جهت حذف: (-f, depth, nid)
    leaf_heap: List[Tuple[float, int, int]] = []

    metrics.max_memory_size = 1
    metrics.max_frontier_size = 1

    def is_ancestor(node_id: int, state: DeepCoreState) -> bool:
        curr: Optional[int] = node_id
        while curr is not None and curr in nodes:
            if nodes[curr]["state"] == state:
                return True
            curr = nodes[curr]["parent"]
        return False

    def has_better_or_equal_in_memory(state: DeepCoreState, new_g: int) -> bool:
        ids = state_nodes.get(state)
        if not ids:
            return False
        for nid in ids:
            nd = nodes.get(nid)
            if nd is not None and not nd["dead"] and nd["g"] <= new_g:
                return True
        return False

    def push_expandable(nid: int) -> None:
        node = nodes.get(nid)
        if node is None or node["dead"]:
            in_frontier.discard(nid)
            return
        if not node["expanded"]:
            if math.isfinite(node["f"]):
                in_frontier.add(nid)
                heapq.heappush(expand_heap, (node["f"], -node["depth"], nid))
            else:
                in_frontier.discard(nid)
        elif node["forgotten"]:
            best_forg = min(node["forgotten"].values())
            if math.isfinite(best_forg):
                in_frontier.add(nid)
                heapq.heappush(expand_heap, (best_forg, -node["depth"], nid))
            else:
                in_frontier.discard(nid)
        else:
            in_frontier.discard(nid)

    def push_leaf(nid: int) -> None:
        if nid == 0:
            return
        node = nodes.get(nid)
        if node is None or node["dead"] or node["expanding"] or node["children"]:
            return
        heapq.heappush(leaf_heap, (-node["f"], node["depth"], nid))

    def delete_node_record(nid: int) -> dict:
        node = nodes.pop(nid)
        in_frontier.discard(nid)
        st = node["state"]
        s_set = state_nodes.get(st)
        if s_set is not None:
            s_set.discard(nid)
            if not s_set:
                del state_nodes[st]
        return node

    def backup_up(nid: Optional[int]) -> None:
        curr = nid
        while curr is not None and curr in nodes:
            node = nodes[curr]
            if not node["expanded"]:
                push_expandable(curr)
                push_leaf(curr)
                break

            estimates = [nodes[c]["f"] for c in node["children"].values() if c in nodes]
            estimates.extend(node["forgotten"].values())
            new_f = max(node["base_f"], min(estimates)) if estimates else math.inf
            changed = new_f != node["f"]
            node["f"] = new_f

            if math.isinf(new_f) and not node["children"] and not node["expanding"]:
                parent_id = node["parent"]
                act = node["action"]
                delete_node_record(curr)
                if parent_id is None or parent_id not in nodes:
                    return
                parent = nodes[parent_id]
                if act is not None:
                    parent["children"].pop(act, None)
                    parent["forgotten"].pop(act, None)
                curr = parent_id
                continue

            push_expandable(curr)
            push_leaf(curr)
            if not changed:
                break
            curr = node["parent"]

    def peek_worst_leaf() -> Optional[int]:
        while leaf_heap:
            neg_f, depth, nid = leaf_heap[0]
            node = nodes.get(nid)
            if (
                node is None
                or nid == 0
                or node["dead"]
                or node["expanding"]
                or node["children"]
                or node["f"] != -neg_f
                or node["depth"] != depth
            ):
                heapq.heappop(leaf_heap)
                continue
            return nid
        return None

    def prune_leaf(nid: int, remember: bool = True) -> None:
        node = delete_node_record(nid)
        parent_id = node["parent"]
        if parent_id is None or parent_id not in nodes:
            return
        parent = nodes[parent_id]
        act = node["action"]
        if act is not None:
            parent["children"].pop(act, None)
            if remember and math.isfinite(node["f"]):
                prev_f = parent["forgotten"].get(act, -math.inf)
                parent["forgotten"][act] = max(prev_f, node["f"])
        backup_up(parent_id)

    while expand_heap:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(in_frontier))
        metrics.max_memory_size = max(metrics.max_memory_size, len(nodes))

        pop_f, neg_depth, best_node_id = heapq.heappop(expand_heap)
        node = nodes.get(best_node_id)
        if node is None or node["dead"] or node["depth"] != -neg_depth:
            continue

        if not node["expanded"]:
            if pop_f != node["f"]:
                continue
        else:
            if not node["forgotten"]:
                continue
            best_forg = min(node["forgotten"].values())
            if not math.isfinite(best_forg) or pop_f != best_forg:
                continue

        if math.isinf(pop_f):
            break

        state = node["state"]

        if not node["expanded"] and env.is_goal(state):
            actions: List[str] = []
            curr: Optional[int] = best_node_id
            while curr is not None and nodes[curr]["parent"] is not None:
                actions.append(nodes[curr]["action"])
                curr = nodes[curr]["parent"]
            actions.reverse()
            metrics.solved = True
            metrics.solution_actions = actions
            metrics.total_cost = node["g"]
            return metrics

        if not node["expanded"]:
            node["expanded"] = True
            node["expanding"] = True
            in_frontier.discard(best_node_id)

            successors = env.get_successors(state)
            metrics.nodes_expanded += 1
            metrics.explored_order.append(state.bot_pos)
            metrics.nodes_generated += len(successors)

            candidates: List[Tuple[float, str, int, float, DeepCoreState]] = []
            for action, next_state, step_cost in successors:
                new_g = node["g"] + step_cost
                if is_ancestor(best_node_id, next_state) or has_better_or_equal_in_memory(next_state, new_g):
                    continue
                new_h = float(heuristic(next_state, env))
                if math.isinf(new_h):
                    continue
                base_f = new_g + new_h
                child_f = max(node["f"], base_f)
                candidates.append((child_f, action.value, new_g, base_f, next_state))

            candidates.sort(key=lambda x: (x[0], x[1]))

            for child_f, action_name, new_g, base_f, next_state in candidates:
                if len(nodes) >= memory_limit:
                    worst_id = peek_worst_leaf()
                    if worst_id is None:
                        continue
                    if nodes[worst_id]["f"] < child_f:
                        node["forgotten"][action_name] = child_f
                        continue
                    heapq.heappop(leaf_heap)
                    prune_leaf(worst_id, remember=True)

                child_id = next_id
                next_id += 1
                nodes[child_id] = {
                    "state": next_state,
                    "g": new_g,
                    "base_f": base_f,
                    "f": child_f,
                    "parent": best_node_id,
                    "action": action_name,
                    "depth": node["depth"] + 1,
                    "children": {},
                    "forgotten": {},
                    "expanded": False,
                    "expanding": False,
                    "dead": False,
                }
                state_nodes.setdefault(next_state, set()).add(child_id)
                node["children"][action_name] = child_id
                push_expandable(child_id)
                push_leaf(child_id)
                metrics.max_memory_size = max(metrics.max_memory_size, len(nodes))
                metrics.max_frontier_size = max(metrics.max_frontier_size, len(in_frontier))

            if best_node_id in nodes:
                nodes[best_node_id]["expanding"] = False
                backup_up(best_node_id)
        else:
            action_name, saved_f = min(
                node["forgotten"].items(),
                key=lambda item: (item[1], item[0]),
            )
            del node["forgotten"][action_name]
            node["expanding"] = True

            successors = env.get_successors(state)
            metrics.nodes_expanded += 1
            metrics.explored_order.append(state.bot_pos)
            metrics.nodes_generated += len(successors)

            restored = None
            for action, next_state, step_cost in successors:
                if action.value == action_name:
                    new_g = node["g"] + step_cost
                    if not is_ancestor(best_node_id, next_state) and not has_better_or_equal_in_memory(next_state, new_g):
                        restored = (action, next_state, step_cost)
                    break

            if restored is None:
                node["expanding"] = False
                backup_up(best_node_id)
                continue

            action, next_state, step_cost = restored
            new_g = node["g"] + step_cost
            new_h = float(heuristic(next_state, env))
            if math.isinf(new_h):
                node["expanding"] = False
                backup_up(best_node_id)
                continue

            base_f = new_g + new_h
            child_f = max(saved_f, base_f)

            if len(nodes) >= memory_limit:
                worst_id = peek_worst_leaf()
                if worst_id is None or nodes[worst_id]["f"] < child_f:
                    node["forgotten"][action.value] = child_f
                    node["expanding"] = False
                    backup_up(best_node_id)
                    continue
                heapq.heappop(leaf_heap)
                prune_leaf(worst_id, remember=True)

            child_id = next_id
            next_id += 1
            nodes[child_id] = {
                "state": next_state,
                "g": new_g,
                "base_f": base_f,
                "f": child_f,
                "parent": best_node_id,
                "action": action.value,
                "depth": node["depth"] + 1,
                "children": {},
                "forgotten": {},
                "expanded": False,
                "expanding": False,
                "dead": False,
            }
            state_nodes.setdefault(next_state, set()).add(child_id)
            node["children"][action.value] = child_id
            push_expandable(child_id)
            push_leaf(child_id)
            metrics.max_memory_size = max(metrics.max_memory_size, len(nodes))
            metrics.max_frontier_size = max(metrics.max_frontier_size, len(in_frontier))

            if best_node_id in nodes:
                nodes[best_node_id]["expanding"] = False
                backup_up(best_node_id)

    metrics.status_message = "No solution found within the memory limit."
    return metrics