from typing import Any, Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
try:
    from .models import CodeForgeAction, CodeForgeObservation, CodeForgeState
except (ImportError, ValueError):
    from models import CodeForgeAction, CodeForgeObservation, CodeForgeState

class CodeForgeProEnv(EnvClient[CodeForgeAction, CodeForgeObservation, CodeForgeState]):
    """Client for interacting with CodeForgeProEnvironment"""

    def _step_payload(self, action: CodeForgeAction) -> Dict[str, Any]:
        return {"action_type": action.action_type.value, "payload": action.payload}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[CodeForgeObservation]:
        obs_data = payload.get("observation", {})
        observation = CodeForgeObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False)
        )

    def _parse_state(self, payload: Dict[str, Any]) -> CodeForgeState:
        return CodeForgeState(**payload)
