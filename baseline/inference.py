import os, random
from openai import OpenAI
from openenv import HTTPEnvClient
from codeforge_pro_env.models import CodeForgeAction, ActionType

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
env = HTTPEnvClient("http://localhost:8000")  # change to HF Space URL after deploy

scores = []
for episode in range(10):
    obs = env.reset(task_id=random.choice(["easy_review", "medium_triage", "hard_pipeline"]))
    total_reward = 0.0
    for _ in range(40):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"{obs.message}\nAction format: JSON with action_type and payload"}]
        )
        # Simple parser (expand as needed)
        action_dict = {"action_type": ActionType.REVIEW_CODE.value, "payload": {}}  # placeholder
        action = CodeForgeAction(**action_dict)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done: break
    scores.append(info.get("grader_score", 0.0))
    print(f"Episode {episode+1} Grader Score: {info.get('grader_score', 0.0)}")

print(f"Average Grader Score (10 episodes): {sum(scores)/len(scores):.3f}")
