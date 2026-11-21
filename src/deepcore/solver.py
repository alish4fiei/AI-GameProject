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
    """
    TODO: Implement Bidirectional Search expanding forward from `env.get_initial_state()`
    via `env.get_successors(s)` and backward from `env.get_goal_states()` via `env.get_predecessors(s)`.
    """
    raise NotImplementedError


# =============================================================================
# 4. Uniform Cost Search (UCS) — [STUDENT TODO]
# =============================================================================
def uniform_cost_search(env: DeepCoreEnv) -> Optional[SearchMetrics]:
    """TODO: Implement Uniform Cost Search ordering the priority queue by path cost g(n)."""
    raise NotImplementedError


# =============================================================================
# 5. Greedy Best-First Search (GBFS) — [STUDENT TODO]
# =============================================================================
def greedy_best_first_search(
    env: DeepCoreEnv, heuristic: Callable[[DeepCoreState, DeepCoreEnv], float]
) -> Optional[SearchMetrics]:
    """TODO: Implement Greedy Best-First Search ordering the priority queue strictly by h(n)."""
    raise NotImplementedError


# =============================================================================
# 6. A* Search — [STUDENT TODO]
# =============================================================================
def a_star_search(
    env: DeepCoreEnv, heuristic: Callable[[DeepCoreState, DeepCoreEnv], float]
) -> Optional[SearchMetrics]:
    """TODO: Implement A* Graph Search ordering the priority queue by f(n) = g(n) + h(n)."""
    raise NotImplementedError


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
