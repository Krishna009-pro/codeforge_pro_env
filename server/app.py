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


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
