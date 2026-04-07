from openenv.core import EnvClient
try:
    from .models import CodeForgeAction, CodeForgeObservation, CodeForgeState
except (ImportError, ValueError):
    from models import CodeForgeAction, CodeForgeObservation, CodeForgeState

class CodeForgeProEnv(EnvClient[CodeForgeAction, CodeForgeObservation, CodeForgeState]):
    """Client for interacting with CodeForgeProEnvironment"""
    pass  # openenv core handles HTTP + sync/async wrappers automatically
