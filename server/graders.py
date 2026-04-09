def base_grade(payload: dict, completed_subgoals: list[str], total_subgoals: int, gt: dict) -> float:
    """
    Standard grading logic for CodeForge Pro tasks.
    """
    score = 0.0
    payload_str = str(payload).lower()
    gt_str = str(gt).lower()
    
    # 1. Ground Truth Overlap (50%)
    overlap = len(set(payload_str.split()) & set(gt_str.split())) / max(len(gt_str.split()), 1)
    score += 0.5 * overlap
    
    # 2. Sub-goal Completion (50%)
    if total_subgoals > 0:
        score += 0.5 * (len(completed_subgoals) / total_subgoals)
    else:
        score += 0.5
        
    return min(1.0, score)

def grade_easy_review(payload: dict, completed_subgoals: list[str]) -> float:
    gt = {"comments": 3, "security": True}
    return base_grade(payload, completed_subgoals, 2, gt)

def grade_medium_triage(payload: dict, completed_subgoals: list[str]) -> float:
    gt = {"root_cause": "off_by_one"}
    return base_grade(payload, completed_subgoals, 2, gt)

def grade_security_audit(payload: dict, completed_subgoals: list[str]) -> float:
    gt = {"secrets_found": 2}
    return base_grade(payload, completed_subgoals, 3, gt)

def grade_hard_pipeline(payload: dict, completed_subgoals: list[str]) -> float:
    gt = {"fixed_files": 2}
    return base_grade(payload, completed_subgoals, 3, gt)

def grade_ci_pipeline_fix(payload: dict, completed_subgoals: list[str]) -> float:
    gt = {"valid_yaml": True}
    return base_grade(payload, completed_subgoals, 3, gt)

def grade_expert_refactor(payload: dict, completed_subgoals: list[str]) -> float:
    gt = {"tests_pass": True}
    return base_grade(payload, completed_subgoals, 3, gt)

def grade_pro_deploy(payload: dict, completed_subgoals: list[str]) -> float:
    gt = {"safe_deploy": True}
    return base_grade(payload, completed_subgoals, 3, gt)

def grade_api_migration(payload: dict, completed_subgoals: list[str]) -> float:
    gt = {"migrated_to_v2": True}
    return base_grade(payload, completed_subgoals, 3, gt)
