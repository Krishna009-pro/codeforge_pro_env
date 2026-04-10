def base_grade(payload: dict, completed_subgoals: list[str], total_subgoals: int, gt: dict, steps: int = 0, max_steps: int = 20) -> float:
    """
    Standard grading logic for CodeForge Pro tasks.
    Now includes an efficiency penalty for RL 'Hard Mode'.
    """
    score = 0.0
    payload_str = str(payload).lower()
    gt_str = str(gt).lower()
    
    # 1. Ground Truth Overlap (40% - reduced to make room for efficiency)
    overlap = len(set(payload_str.split()) & set(gt_str.split())) / max(len(gt_str.split()), 1)
    score += 0.4 * overlap
    
    # 2. Sub-goal Completion (50%)
    if total_subgoals > 0:
        score += 0.5 * (len(completed_subgoals) / total_subgoals)
    else:
        score += 0.5
        
    # 3. Efficiency Bonus (10%)
    # Reward finishing in few steps, penalize using full budget
    if steps > 0:
        efficiency = max(0.0, 1.0 - (steps / max_steps))
        score += 0.1 * efficiency
    else:
        score += 0.05 # Baseline
        
    return min(1.0, score)

def grade_easy_review(payload: dict, completed_subgoals: list[str], **kwargs) -> float:
    gt = {"comments": 3, "security": True}
    return base_grade(payload, completed_subgoals, 2, gt, **kwargs)

def grade_medium_triage(payload: dict, completed_subgoals: list[str], **kwargs) -> float:
    gt = {"root_cause": "off_by_one"}
    return base_grade(payload, completed_subgoals, 2, gt, **kwargs)

def grade_security_audit(payload: dict, completed_subgoals: list[str], **kwargs) -> float:
    gt = {"secrets_found": 2}
    return base_grade(payload, completed_subgoals, 3, gt, **kwargs)

def grade_hard_pipeline(payload: dict, completed_subgoals: list[str], **kwargs) -> float:
    gt = {"fixed_files": 2}
    return base_grade(payload, completed_subgoals, 3, gt, **kwargs)

def grade_ci_pipeline_fix(payload: dict, completed_subgoals: list[str], **kwargs) -> float:
    gt = {"valid_yaml": True}
    return base_grade(payload, completed_subgoals, 3, gt, **kwargs)

def grade_expert_refactor(payload: dict, completed_subgoals: list[str], **kwargs) -> float:
    gt = {"tests_pass": True}
    return base_grade(payload, completed_subgoals, 3, gt, **kwargs)

def grade_pro_deploy(payload: dict, completed_subgoals: list[str], **kwargs) -> float:
    gt = {"safe_deploy": True}
    return base_grade(payload, completed_subgoals, 3, gt, **kwargs)

def grade_api_migration(payload: dict, completed_subgoals: list[str], **kwargs) -> float:
    gt = {"migrated_to_v2": True}
    return base_grade(payload, completed_subgoals, 3, gt, **kwargs)

def grade_doc_update(payload: dict, completed_subgoals: list[str], **kwargs) -> float:
    gt = {"broken_link_fixed": True}
    return base_grade(payload, completed_subgoals, 2, gt, **kwargs)
