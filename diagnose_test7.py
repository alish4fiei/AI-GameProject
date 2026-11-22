from deepcore.environment import DeepCoreEnv
from deepcore import solver, heuristics

import time
import traceback


algorithms = [
    ("IDS", lambda env: solver.iterative_deepening_search(env)),
    ("Bidirectional", lambda env: solver.bidirectional_search(env)),
    ("IDA*", lambda env: solver.ida_star_search(
        env, heuristics.h2_advanced
    )),
]


for level in ("level2", "level3", "level4", "level5"):

    print(f"\n{'=' * 45}")
    print(f"TESTING {level.upper()}")
    print("=" * 45)

    for name, algorithm in algorithms:

        print(f"\nRunning {name}...", flush=True)

        env = DeepCoreEnv.from_file_or_builtin(level)

        start_time = time.perf_counter()

        try:

            result = algorithm(env)

            elapsed = time.perf_counter() - start_time

            print("Solved:", result.solved)
            print("Status:", result.status_message)
            print("Nodes expanded:", result.nodes_expanded)
            print("Nodes generated:", result.nodes_generated)
            print("Time:", round(elapsed, 3), "seconds")

            if result.solved:

                state = env.get_initial_state()

                for action in result.solution_actions:
                    state, _ = env.transition(state, action)

                print("Valid solution:", env.is_goal(state))
                print("Solution steps:", len(result.solution_actions))

            else:
                print("FAILED:", name, "on", level)

        except Exception:

            traceback.print_exc()

            print("ERROR:", name, "on", level)