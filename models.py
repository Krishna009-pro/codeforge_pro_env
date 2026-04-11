from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, List, Literal, Optional
from enum import Enum
import uuid
from openenv.core import Action, Observation, State

class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"

class ActionType(str, Enum):
    REVIEW_CODE = "review_code"
    TRIAGE_BUG = "triage_bug"
    DEBUG_PIPELINE = "debug_pipeline"
    REFACTOR_CODE = "refactor_code"
    SUBMIT_FIX = "submit_fix"
    RUN_TEST = "run_test"
    GIT_COMMIT = "git_commit"

class CodeForgeAction(Action):
    """
    Action to be taken in the environment.
    Example: {"action_type": "review_code", "payload": {"comments": ["Check style."]}}
    """
    action_type: ActionType = Field(..., description="The type of action to execute")
    payload: Dict = Field(default_factory=dict, description="A dictionary containing parameters like 'comments', 'file_id', etc.")

class CodeForgeObservation(Observation):
    task_id: str
    difficulty: Difficulty
    message: str
    current_file: Optional[str] = None
    file_snapshot: str = ""
    console_output: str = ""
    progress: float = Field(0.0, ge=0.0, le=1.0)
    reward: float = 0.0
    done: bool = False
    available_actions: List[str]
    step_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class CodeForgeState(State):
    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    difficulty: Difficulty
    steps_taken: int = 0
    score: float = 0.0
    history: List[Dict] = Field(default_factory=list)
    completed_subgoals: List[str] = Field(default_factory=list, description="List of sub-goals reached in this episode")
    last_action_type: Optional[ActionType] = None
