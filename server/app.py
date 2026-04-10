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


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
