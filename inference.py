import asyncio
import sys
import argparse
import os
import json
from datetime import datetime

# Import robustly for both local and installed runs
try:
    from codeforge_pro_env import CodeForgeProEnv, CodeForgeAction, ActionType
    from openenv.core.llm_client import OpenAIClient, create_llm_client
except ImportError:
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
        print(f"Agent Thought: {data.get('thought', 'No reasoning provided')}", file=sys.stderr, flush=True)
        return CodeForgeAction(
            action_type=ActionType(data["action_type"]),
            payload=data.get("payload", {})
        )
    except Exception as e:
        print(f"Warning: LLM reasoning failed ({e}). Falling back to style review.", file=sys.stderr, flush=True)
        return CodeForgeAction(action_type=ActionType.REVIEW_CODE, payload={"comment": "Analyzing structure..."})

async def run_episode(env: CodeForgeProEnv, llm: OpenAIClient, task_id: str = None):
    print(f"\n--- Episode Start ---", file=sys.stderr, flush=True)
    result = await env.reset(task_id=task_id)
    obs = result.observation
    total_reward = 0
    steps = 0
    done = result.done
    
    print(f"[START] task={obs.task_id}", flush=True)
    print(f"Goal: {obs.message}", file=sys.stderr, flush=True)

    while not done and steps < 10: # Limit steps for inference baseline
        action = await agent_policy(llm, obs)
        result = await env.step(action)
        obs = result.observation
        total_reward += result.reward
        done = result.done
        steps += 1
        
        print(f"Step {steps}: Action={action.action_type.value}, Reward={result.reward:.4f}, Progress={obs.progress:.2f}", file=sys.stderr, flush=True)
        print(f"[STEP] step={steps} reward={result.reward:.4f}", flush=True)

    # Ensure score is strictly between 0 and 1 per validator requirements
    final_score = max(0.0001, min(0.9999, float(obs.progress)))
    
    print(f"[END] task={obs.task_id} score={final_score:.4f} steps={steps}", flush=True)

    return {
        "task_id": obs.task_id,
        "reward": total_reward,
        "steps": steps,
        "success": obs.progress >= 1.0,
        "score": final_score
    }

async def main():
    parser = argparse.ArgumentParser(description="CodeForge Pro Baseline Inference")
    parser.add_argument("--url", default="http://localhost:8000", help="Env server URL")
    parser.add_argument("--provider", default="huggingface", choices=["huggingface", "openai"], help="LLM provider")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model name")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    args = parser.parse_args()

    # Read API environment variables
    api_base = os.getenv("API_BASE_URL")
    api_key = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Error: Missing API key. Please set API_KEY, HF_TOKEN, or OPENAI_API_KEY.", file=sys.stderr)
        return

    # Initialize LLM Client
    if api_base:
        # Hackathon Proxy Mode
        print(f"Using Hackathon Proxy: {api_base}", file=sys.stderr)
        # We manually configure OpenAIClient to use the exact provided base URL
        llm = OpenAIClient(
            endpoint="http://temp", # Placeholder, will be overridden
            port=80,
            model=args.model,
            api_key=api_key,
            system_prompt=SYSTEM_PROMPT
        )
        llm._client.base_url = api_base
    elif args.provider == "huggingface":
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

    print(f"Starting LLM-based inference against {args.url}", file=sys.stderr)
    print(f"Provider: {args.provider if not api_base else 'hackathon-proxy'}, Model: {args.model}", file=sys.stderr)

    async with CodeForgeProEnv(base_url=args.url) as env:
        results = []
        for i in range(args.episodes):
            res = await run_episode(env, llm)
            results.append(res)
        
        # Aggregate statistics
        success_count = sum(1 for r in results if r["success"])
        avg_reward = sum(r["reward"] for r in results) / len(results)
        
        print("\n" + "="*40, file=sys.stderr, flush=True)
        print("BASELINE INFERENCE SUMMARY", file=sys.stderr, flush=True)
        print("="*40, file=sys.stderr, flush=True)
        print(f"Total Episodes: {len(results)}", file=sys.stderr, flush=True)
        print(f"Success Rate:   {success_count/len(results)*100:.1f}%", file=sys.stderr, flush=True)
        print(f"Avg. Reward:    {avg_reward:.4f}", file=sys.stderr, flush=True)
        print("="*40, file=sys.stderr, flush=True)

if __name__ == "__main__":
    asyncio.run(main())
