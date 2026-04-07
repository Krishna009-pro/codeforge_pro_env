import asyncio
import argparse
import random
from datetime import datetime
try:
    from codeforge_pro_env import CodeForgeProEnv, CodeForgeAction, ActionType
except ImportError:
    # Fallback for local run without installation
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from client import CodeForgeProEnv
    from models import CodeForgeAction, ActionType

async def run_episode(env: CodeForgeProEnv, task_id: str = None):
    """
    Run a single episode with a basic heuristic policy.
    """
    obs = await env.reset(task_id=task_id)
    total_reward = 0
    steps = 0
    done = False
    
    print(f"\n--- Episode Start: {obs.task_id} ({obs.difficulty}) ---")
    print(f"Goal: {obs.message}")

    while not done and steps < 20: # Limit steps for inference test
        # Basic heuristic: if tests fail, triage; if fix is ready, submit
        if "FAIL" in obs.console_output or "IndexError" in obs.console_output:
            action_type = ActionType.TRIAGE_BUG
            payload = {"found": "IndexError at loader.py:12"}
        elif obs.progress > 0.8:
            action_type = ActionType.SUBMIT_FIX
            payload = {"fix": "Applied conversion and verified tests."}
        else:
            action_type = ActionType.REVIEW_CODE
            payload = {"comment": f"Analyzing {obs.current_file}"}
            
        action = CodeForgeAction(action_type=action_type, payload=payload)
        
        # Step the environment
        result = await env.step(action)
        obs = result.observation
        total_reward += result.reward
        done = result.done
        steps += 1
        
        print(f"Step {steps}: Action={action_type.value}, Reward={result.reward:.4f}, Progress={obs.progress:.2f}")

    return {
        "task_id": obs.task_id,
        "reward": total_reward,
        "steps": steps,
        "success": obs.progress >= 1.0
    }

async def main():
    parser = argparse.ArgumentParser(description="CodeForge Pro Inference Baseline")
    parser.add_argument("--url", default="http://localhost:8000", help="Env server URL")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--task", default=None, help="Specific task ID to test")
    args = parser.parse_args()

    print(f"Starting inference against {args.url} at {datetime.now()}")
    
    async with CodeForgeProEnv(base_url=args.url) as env:
        results = []
        for i in range(args.episodes):
            res = await run_episode(env, task_id=args.task)
            results.append(res)
        
        # Aggregate statistics
        success_count = sum(1 for r in results if r["success"])
        avg_reward = sum(r["reward"] for r in results) / len(results)
        
        print("\n" + "="*40)
        print("📊 INFERENCE SUMMARY")
        print("="*40)
        print(f"Total Episodes: {len(results)}")
        print(f"Success Rate:   {success_count/len(results)*100:.1f}%")
        print(f"Avg. Reward:    {avg_reward:.4f}")
        print("="*40)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Ensure the server is running or the URL is correct.")
