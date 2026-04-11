from fastapi import FastAPI
try:
    from openenv.core.env_server import create_app
except ImportError:
    from openenv.core.env_server.http_server import create_app

try:
    from ..models import CodeForgeAction, CodeForgeObservation, CodeForgeState
    from .codeforge_pro_environment import CodeForgeProEnvironment
except (ImportError, ValueError):
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    import models
    from server.codeforge_pro_environment import CodeForgeProEnvironment
    CodeForgeAction = models.CodeForgeAction
    CodeForgeObservation = models.CodeForgeObservation
    CodeForgeState = models.CodeForgeState

# Create the app with web interface and README integration
app = create_app(
    CodeForgeProEnvironment,
    CodeForgeAction,
    CodeForgeObservation,
    env_name="codeforge_pro_env",
    max_concurrent_envs=1,
)

# --- MANUAL TASK DISCOVERY ENDPOINTS ---
# Registered to resolve "Not enough tasks with graders" validator errors
@app.get("/tasks")
async def get_all_tasks():
    """Explicitly expose tasks to the platform validator."""
    return CodeForgeProEnvironment.list_tasks()

@app.get("/v1/tasks")
async def get_v1_tasks():
    return CodeForgeProEnvironment.list_tasks()

# --- PLATFORM COMPATIBILITY ALIASES ---
@app.post("/openenv/reset")
async def openenv_reset(request: dict = None):
    """Alias for platform compliance."""
    from openenv.core.env_server.types import ResetRequest
    from openenv.core.env_server.serialization import serialize_observation
    req = ResetRequest.model_validate(request or {})
    env = CodeForgeProEnvironment()
    try:
        obs = env.reset(**req.model_dump(exclude_unset=True))
        return serialize_observation(obs)
    finally:
        env.close()

@app.post("/openenv/step")
async def openenv_step(request: dict):
    """Alias for platform compliance."""
    from openenv.core.env_server.types import StepRequest
    from openenv.core.env_server.serialization import deserialize_action, serialize_observation
    req = StepRequest.model_validate(request)
    action = deserialize_action(req.action, CodeForgeAction)
    env = CodeForgeProEnvironment()
    try:
        obs = env.step(action) 
        return serialize_observation(obs)
    finally:
        env.close()

# --- STARTUP DIAGNOSTICS ---
@app.on_event("startup")
async def startup_event():
    try:
        # Print all registered routes for debugging
        print("--- REGISTERED ROUTES ---")
        for route in app.routes:
            if hasattr(route, "path"):
                print(f" - {route.path}")
                
        tasks = CodeForgeProEnvironment.list_tasks()
        print(f"DISCOVERY SUCCESS: Detected {len(tasks)} tasks.")
    except Exception as e:
        print(f"DISCOVERY FAILED: {e}")


def main():
    """Entry point for direct execution."""
    import uvicorn
    import argparse
    import os

    parser = argparse.ArgumentParser(description="CodeForge Pro Environment Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8000)), help="Port to bind to")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
