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
    weight: float = 1.5,
) -> Optional[SearchMetrics]:
    """TODO (Bonus): Implement Weighted A* ordering the frontier by f(n) = g(n) + W * h(n)."""
    raise NotImplementedError


# =============================================================================
# BONUS 2: Iterative Deepening A* (IDA*) — [STUDENT BONUS TODO]
# =============================================================================
def ida_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
    max_expansions: Optional[int] = None,
) -> Optional[SearchMetrics]:
    """TODO (Bonus): Implement Iterative Deepening A* (IDA*) using f-cost threshold contours."""
    raise NotImplementedError


# =============================================================================
# BONUS 3: Simplified Memory-Bounded A* (SMA*) — [STUDENT BONUS TODO]
# =============================================================================
def sma_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
    memory_limit: Optional[int] = None,
) -> Optional[SearchMetrics]:
    """TODO (Bonus): Implement Simplified Memory-Bounded A* (SMA*)."""
    raise NotImplementedError
