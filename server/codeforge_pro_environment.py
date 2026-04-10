from openenv.core import Environment, Observation
try:
    from ..models import CodeForgeAction, CodeForgeObservation, CodeForgeState, Difficulty, ActionType
    from ..config import CodeForgeConfig
except (ImportError, ValueError):
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    import models
    import config
    CodeForgeAction = models.CodeForgeAction
    CodeForgeObservation = models.CodeForgeObservation
    CodeForgeState = models.CodeForgeState
    Difficulty = models.Difficulty
    ActionType = models.ActionType
    CodeForgeConfig = config.CodeForgeConfig
import random, json
try:
    from . import graders
except (ImportError, ValueError):
    import graders

TASK_DATA = {
    "easy_review": {
        "name": "Easy: Code Review",
        "difficulty": Difficulty.EASY,
        "desc": "Review small PR for style/security",
        "subgoals": ["identified_security_issue", "added_style_comment"],
        "files": {"main.py": "def upload(f): pass # TODO: check secure\nprint('hello')"},
        "gt": {"comments": 3, "security": True},
        "enabled": True
    },
    "medium_triage": {
        "name": "Medium: Bug Triage",
        "difficulty": Difficulty.MEDIUM,
        "desc": "Triage IndexError in data loader",
        "subgoals": ["found_index_error_line", "isolated_data_source"],
        "files": {"loader.py": "data = [1,2,3]\ndef get(i): return data[i]"},
        "gt": {"root_cause": "off_by_one"},
        "enabled": True
    },
    "hard_pipeline": {
        "name": "Hard: Data Pipeline",
        "difficulty": Difficulty.HARD,
        "desc": "Fix failing ETL pipeline (CSV/JSON)",
        "subgoals": ["parsed_csv", "detected_encoding_error", "fixed_transformation"],
        "files": {"pipeline.py": "def process(): read_csv('data.csv', encoding='ascii')"},
        "gt": {"fixed_files": 2},
        "enabled": True
    },
    "expert_refactor": {
        "name": "Expert: Code Refactor",
        "difficulty": Difficulty.EXPERT,
        "desc": "Refactor legacy async code + add tests",
        "subgoals": ["converted_to_async", "handled_concurrency", "tests_passing"],
        "files": {"legacy.py": "def slow(): time.sleep(1)\ndef main(): [slow() for _ in range(5)]"},
        "gt": {"tests_pass": True},
        "enabled": True
    },
    "pro_deploy": {
        "name": "Expert: Safe Deployment",
        "difficulty": Difficulty.EXPERT,
        "desc": "Simulate safe deployment with rollback",
        "subgoals": ["checked_health", "staged_blue_green", "verified_metrics"],
        "files": {"deploy.sh": "helm upgrade --install my-app ./charts"},
        "gt": {"safe_deploy": True},
        "enabled": True
    },
    "security_audit": {
        "name": "Medium: Security Audit",
        "difficulty": Difficulty.MEDIUM,
        "desc": "Audit config for hardcoded secrets",
        "subgoals": ["identified_secret", "removed_plaintext", "updated_env_var"],
        "files": {"config.json": '{"db_pass": "admin123", "api_key": "sk-12345"}'},
        "gt": {"secrets_found": 2},
        "enabled": True
    },
    "ci_pipeline_fix": {
        "name": "Hard: CI Pipeline",
        "difficulty": Difficulty.HARD,
        "desc": "Fix broken GitHub Action YAML syntax",
        "subgoals": ["validated_yaml", "fixed_indentation", "set_correct_node_version"],
        "files": {".github/workflows/main.yml": "name: CI\non: push\njobs:\n  build: runs-on: ubuntu-latest\n  steps: - uses: actions/checkout@v2"},
        "gt": {"valid_yaml": True},
        "enabled": True
    },
    "api_migration": {
        "name": "Expert: API Migration",
        "difficulty": Difficulty.EXPERT,
        "desc": "Migrate from v1 to v2 internal API",
        "subgoals": ["found_legacy_calls", "implemented_v2_wrapper", "tested_rollback"],
        "files": {"app.py": "from internal_v1 import Client\nc = Client()\nc.fetch_data()"},
        "gt": {"migrated_to_v2": True},
        "enabled": True
    },
    "doc_update": {
        "name": "Easy: Documentation Update",
        "difficulty": Difficulty.EASY,
        "desc": "Fix broken links and outdated versions in README",
        "subgoals": ["identified_broken_link", "fixed_broken_link"],
        "files": {"README.md": "# Docs\nCheck out [this link](http://broken-link.com) for details. Version 0.1"},
        "gt": {"broken_link_fixed": True},
        "enabled": True
    },
}

class CodeForgeProEnvironment(Environment[CodeForgeAction, CodeForgeObservation, CodeForgeState]):
    def __init__(self, config: CodeForgeConfig | None = None):
        self.config = config or CodeForgeConfig()
        if self.config.seed is not None:
            random.seed(self.config.seed)
        self.episode_id = None
        self.current_task = None
        self._state = None
        self.max_steps = self.config.max_steps

    def reset(self, task_id: str | None = None, **kwargs) -> CodeForgeObservation:
        if task_id is None or task_id not in TASK_DATA:
            task_id = random.choice(list(TASK_DATA.keys()))
        self.episode_id = f"episode_{random.randint(1000,9999)}"
        self.current_task = task_id
        task_data = TASK_DATA[task_id]
        self._state = CodeForgeState(episode_id=self.episode_id, task_id=task_id, difficulty=task_data["difficulty"])
        return CodeForgeObservation(
            task_id=task_id,
            difficulty=task_data["difficulty"],
            message=f"RL GOAL: {task_data['desc']}",
            current_file=list(task_data["files"].keys())[0],
            file_snapshot=json.dumps(task_data["files"]),
            console_output="System ready. Start your task.",
            progress=0.0,
            available_actions=[a.value for a in ActionType],
            step_count=0
        )

    def step(self, action: CodeForgeAction, **kwargs) -> CodeForgeObservation:
        if not self._state: raise ValueError("Reset required.")
        
        # 1. Immediate Efficiency Penalty
        step_penalty = -0.01 * self.config.reward_scale
        
        # 2. Redundancy Penalty
        redundancy_penalty = -0.05 if action.action_type == self._state.last_action_type else 0.0
        
        # 3. Dense Rewards (Action-based)
        base_reward = self._compute_dense_reward(action)
        
        # 4. Sub-goal tracking & rewards
        sub_goal_reward = self._check_and_reward_subgoals(action)
        
        total_reward = (base_reward + sub_goal_reward + step_penalty + redundancy_penalty) * self.config.reward_scale
        
        self._state.steps_taken += 1
        self._state.score += total_reward
        self._state.last_action_type = action.action_type
        self._state.history.append({"step": self._state.steps_taken, "action": action.action_type.value, "reward": total_reward})
        
        done = (self._state.steps_taken >= self.max_steps) or self._is_complete(action)
        obs = self._next_observation(action)
        
        grader_score = self._grader(self.current_task, action) if done else 0.0
        
        obs.reward = total_reward
        obs.done = done
        obs.metadata = {
            "grader_score": grader_score,
            "episode_id": self.episode_id,
            "completed_subgoals": self._state.completed_subgoals
        }
        
        return obs

    def _check_and_reward_subgoals(self, action: CodeForgeAction) -> float:
        task_info = TASK_DATA[self.current_task]
        subgoals = task_info["subgoals"]
        payload_str = json.dumps(action.payload).lower()
        reward = 0.0
        
        # Heuristic subgoal detection
        for sg in subgoals:
            if sg not in self._state.completed_subgoals:
                # Mock logic: check for keywords in action payload
                keyword = sg.split("_")[0] 
                if keyword in payload_str or (action.action_type == ActionType.RUN_TEST and "test" in sg):
                    self._state.completed_subgoals.append(sg)
                    reward += 0.2  # Significant reward for sub-goal
        return reward

    def _compute_dense_reward(self, action: CodeForgeAction) -> float:
        base = 0.02 # Lowered base to favor subgoals
        if action.action_type in [ActionType.REVIEW_CODE, ActionType.RUN_TEST]: base += 0.03
        payload_str = json.dumps(action.payload).lower()
        if self.config.enable_safety_penalties and any(x in payload_str for x in ["rm -rf", "delete", "os.remove", "exec("]): return -0.5
        if "fix" in payload_str or "corrected" in payload_str: base += 0.05
        return base

    def _grader(self, task_id: str, final_action: CodeForgeAction) -> float:
        # Map task IDs to standalone grader functions
        grader_map = {
            "easy_review": graders.grade_easy_review,
            "medium_triage": graders.grade_medium_triage,
            "security_audit": graders.grade_security_audit,
            "hard_pipeline": graders.grade_hard_pipeline,
            "ci_pipeline_fix": graders.grade_ci_pipeline_fix,
            "expert_refactor": graders.grade_expert_refactor,
            "pro_deploy": graders.grade_pro_deploy,
            "api_migration": graders.grade_api_migration,
        }
        
        grader_func = grader_map.get(task_id)
        if not grader_func:
            return 0.0
            
        return grader_func(final_action.payload, self._state.completed_subgoals)

    def _is_complete(self, action: CodeForgeAction) -> bool:
        return (action.action_type == ActionType.SUBMIT_FIX and len(self._state.completed_subgoals) >= 2) or \
               (self._state.steps_taken >= self.max_steps)

    def _next_observation(self, action: CodeForgeAction) -> CodeForgeObservation:
        task_info = TASK_DATA[self.current_task]
        console = f"Status: {action.action_type.value} complete."
        if action.action_type == ActionType.RUN_TEST:
            console = "Tests: PASS" if len(self._state.completed_subgoals) > 1 else "Tests: FAIL (IndexError at loader.py:12)"
        
        progress = len(self._state.completed_subgoals) / len(task_info["subgoals"])
        
        return CodeForgeObservation(
            task_id=self.current_task,
            difficulty=task_info["difficulty"],
            message=f"Progress: {progress*100:.1f}%. Subgoals met: {', '.join(self._state.completed_subgoals)}",
            current_file=list(task_info["files"].keys())[0],
            file_snapshot=json.dumps(task_info["files"]),
            console_output=console,
            progress=progress,
            available_actions=[a.value for a in ActionType],
            step_count=self._state.steps_taken
        )

    def state(self) -> CodeForgeState:
        return self._state

    @property
    def current_state(self) -> CodeForgeState:
        return self._state

    @property
    def tasks(self) -> list[dict]:
        return [
            {
                "id": tid,
                "name": data.get("name", tid.replace("_", " ").title()),
                "description": data["desc"],
                "difficulty": data["difficulty"].name.lower() if hasattr(data["difficulty"], "name") else str(data["difficulty"]),
                "grader": f"server.graders:grade_{tid}"
            }
            for tid, data in TASK_DATA.items()
        ]
