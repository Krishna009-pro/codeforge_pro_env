from pydantic import BaseModel
try:
    from .models import Difficulty
except (ImportError, ValueError):
    from models import Difficulty
from typing import Optional

class CodeForgeConfig(BaseModel):
    max_steps: int = 40
    difficulty: Difficulty = Difficulty.MEDIUM
    reward_scale: float = 1.0
    seed: Optional[int] = None
    enable_safety_penalties: bool = True
    progress_shaping: bool = True
    num_parallel_tasks: int = 1
