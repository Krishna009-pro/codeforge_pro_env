import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from server.codeforge_pro_environment import CodeForgeProEnvironment
from models import CodeForgeAction, ActionType, Difficulty

def test_rl_env():
    env = CodeForgeProEnvironment()
    
    # 1. Test Reset
    obs = env.reset(task_id="medium_triage")
    print(f"RESET: task={obs.task_id}, msg={obs.message}")
    
    # 2. Sequential Action: Triage (Subgoal 1)
    # Hard Mode: Reward = +0.1 (subgoal) - 0.03 (step) = 0.07
    action = CodeForgeAction(action_type=ActionType.TRIAGE_BUG, payload={"found": "IndexError at loader.py:12"})
    obs = env.step(action)
    print(f"STEP 1 (Subgoal): reward={obs.reward:.4f}, done={obs.done}, completed={obs.metadata['completed_subgoals']}")
    
    # 3. Redundant Action (Penalty)
    # Hard Mode: Reward = -0.1 (redundance) - 0.03 (step) = -0.13
    action_redundant = CodeForgeAction(action_type=ActionType.TRIAGE_BUG, payload={"found": "again"})
    obs_redundant = env.step(action_redundant)
    print(f"STEP 2 (Redundant): reward={obs_redundant.reward:.4f}, msg={obs_redundant.message}")
    
    # 4. Next Subgoal + Completion (Requires ALL subgoals)
    # Hard Mode: Reward = +0.1 (subgoal) + 0.05 (fix text) - 0.03 (step) = 0.12
    action_solve = CodeForgeAction(action_type=ActionType.SUBMIT_FIX, payload={"fix": "isolated data source"})
    obs_final = env.step(action_solve)
    print(f"STEP 3 (Complete): reward={obs_final.reward:.4f}, done={obs_final.done}, progress={obs_final.progress:.2f}")
    if obs_final.done:
        print(f"FINAL GRADER SCORE: {obs_final.metadata['grader_score']:.4f}")
    
    # 5. Verify Safety (Harsher)
    # Hard Mode: Reward = -1.0 (safety) - 0.03 (step) = -1.03
    action_danger = CodeForgeAction(action_type=ActionType.RUN_TEST, payload={"cmd": "rm -rf /"})
    obs_danger = env.step(action_danger)
    print(f"STEP 4 (Danger): reward={obs_danger.reward:.4f}")

if __name__ == "__main__":
    test_rl_env()
