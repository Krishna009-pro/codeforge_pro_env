import asyncio
import argparse
import os
import json
from datetime import datetime

# Import robustly for both local and installed runs
try:
    from codeforge_pro_env import CodeForgeProEnv, CodeForgeAction, ActionType
    from openenv.core.llm_client import OpenAIClient, create_llm_client
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from client import CodeForgeProEnv
    from models import CodeForgeAction, ActionType
    # Try to import openenv core even if not installed
    try:
        from openenv.core.llm_client import OpenAIClient, create_llm_client
    except ImportError:
        print("Error: openenv-core not found. Install it with 'pip install openenv-core'.")
        sys.exit(1)

# Configuration for the LLM
SYSTEM_PROMPT = """You are a senior software engineering agent. Your goal is to solve the task provided.
You will receive an observation containing file snapshots, console output, and progress.
You must return a JSON response in the following format:
{
  "thought": "Your reasoning here",
  "action_type": "one of the available action enums",
  "payload": { ... }
}

Action Types: review_code, triage_bug, run_test, refactor_code, submit_fix, git_commit.
Example:
{
  "thought": "The tests failed with an IndexError in loader.py. I'll triage the bug.",
  "action_type": "triage_bug",
  "payload": {"found": "IndexError at loader.py:12"}
}
"""

async def agent_policy(llm: OpenAIClient, obs):
    """
    Query the LLM to decide the next action.
    """
    prompt = f"""Task: {obs.task_id} ({obs.difficulty})
Goal: {obs.message}
Current File: {obs.current_file}
Progress: {obs.progress:.2f}

File Snapshot:
{obs.file_snapshot}

Console Output:
{obs.console_output}

Decide your next action. Use JSON format."""

    try:
        response_text = await llm.complete(prompt)
        # Parse JSON from response
        # (Handling potential LLM noise like markdown blocks)
        clean_json = response_text.strip()
        if "```json" in clean_json:
            clean_json = clean_json.split("```json")[-1].split("```")[0].strip()
        elif "```" in clean_json:
            clean_json = clean_json.split("```")[-1].split("```")[0].strip()
            
        data = json.loads(clean_json)
        print(f"Agent Thought: {data.get('thought', 'No reasoning provided')}")
        return CodeForgeAction(
            action_type=ActionType(data["action_type"]),
            payload=data.get("payload", {})
        )
    except Exception as e:
        print(f"Warning: LLM reasoning failed ({e}). Falling back to style review.")
        return CodeForgeAction(action_type=ActionType.REVIEW_CODE, payload={"comment": "Analyzing structure..."})

async def run_episode(env: CodeForgeProEnv, llm: OpenAIClient, task_id: str = None):
    print(f"\n--- Episode Start ---")
    result = await env.reset(task_id=task_id)
    obs = result.observation
    total_reward = 0
    steps = 0
    done = result.done
    
    print(f"Goal: {obs.message}")

    while not done and steps < 10: # Limit steps for inference baseline
        action = await agent_policy(llm, obs)
        result = await env.step(action)
        obs = result.observation
        total_reward += result.reward
        done = result.done
        steps += 1
        
        print(f"Step {steps}: Action={action.action_type.value}, Reward={result.reward:.4f}, Progress={obs.progress:.2f}")

    return {
        "task_id": obs.task_id,
        "reward": total_reward,
        "steps": steps,
        "success": obs.progress >= 1.0
    }

async def main():
    parser = argparse.ArgumentParser(description="CodeForge Pro Baseline Inference")
    parser.add_argument("--url", default="http://localhost:8000", help="Env server URL")
    parser.add_argument("--provider", default="huggingface", choices=["huggingface", "openai"], help="LLM provider")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model name")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    args = parser.parse_args()

    # Read API Key
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Missing API key. Please set HF_TOKEN or OPENAI_API_KEY environment variable.")
        return

    # Initialize LLM Client
    if args.provider == "huggingface":
        # Hugging Face Inference API is OpenAI-compatible at this endpoint
        llm = OpenAIClient(
            endpoint="https://api-inference.huggingface.co",
            port=443,
            model=args.model,
            api_key=api_key,
            system_prompt=SYSTEM_PROMPT
        )
    else:
        llm = create_llm_client("openai", model=args.model, api_key=api_key, system_prompt=SYSTEM_PROMPT)

    print(f"Starting LLM-based inference against {args.url}")
    print(f"Provider: {args.provider}, Model: {args.model}")

    async with CodeForgeProEnv(base_url=args.url) as env:
        results = []
        for i in range(args.episodes):
            res = await run_episode(env, llm)
            results.append(res)
        
        # Aggregate statistics
        success_count = sum(1 for r in results if r["success"])
        avg_reward = sum(r["reward"] for r in results) / len(results)
        
        print("\n" + "="*40)
        print("BASELINE INFERENCE SUMMARY")
        print("="*40)
        print(f"Total Episodes: {len(results)}")
        print(f"Success Rate:   {success_count/len(results)*100:.1f}%")
        print(f"Avg. Reward:    {avg_reward:.4f}")
        print("="*40)

if __name__ == "__main__":
    asyncio.run(main())
