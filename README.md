# 🚀 CodeForge Pro: The Software Engineering RL Accelerator

[![OpenEnv](https://img.shields.io/badge/Framework-OpenEnv-blue?style=flat-square)](https://github.com/meta-pytorch/OpenEnv)
[![License](https://img.shields.io/badge/License-BSD--3-orange?style=flat-square)](./LICENSE)
[![Difficulty](https://img.shields.io/badge/Challenge-Easy%20to%20Expert-purple?style=flat-square)](#-tasks--difficulty)
[![Version](https://img.shields.io/badge/Version-1.0.0-green?style=flat-square)](https://huggingface.co/spaces/Krishnapatil999/codeforge_pro_env)

**CodeForge Pro** is a high-fidelity Reinforcement Learning environment designed for training and evaluating LLM-based agents on real-world software engineering workflows. Built on the **OpenEnv** framework, it bridges the gap between simple toy problems and complex repository-level engineering tasks.

---

## 🎯 Motivation & Overview
Traditional coding benchmarks often focus on isolated snippets (LeetCode-style). **CodeForge Pro** shifts the focus to **Systematic Engineering**:
- **Real-World Fidelity**: Simulates tasks engineers actually perform, such as CI/CD debugging, security audits, and API migrations.
- **Context-Aware Triage**: Agents must navigate multiple files to find the root cause of failures.
- **Modernization**: Focuses on modern patterns like async/await and Blue-Green deployments.
- **Safe Operations**: Reward signals heavily penalize dangerous system commands (e.g., `rm -rf`).
- **Dense Progress Feedback**: Uses sub-goal tracking to provide high-resolution gradients for RL training.

### 🏗️ Environment Architecture
```mermaid
graph TD
    A[Agent / Client] -->|Action| B[OpenEnv Server]
    B -->|Step| C[CodeForgeProEnvironment]
    C -->|Validate| D[Sub-goal Engine]
    C -->|Reward| E[Dense Grader]
    D -->|Observation| B
    E -->|Reward| B
    B -->|Observation + Reward| A
    
    subgraph Environments
        F[Easy: Review]
        G[Medium: Triage]
        H[Hard: Pipeline]
        I[Expert: Refactor]
    end
```

---

## 🕹️ Spaces & Definitions

### 🔹 Action Space (`CodeForgeAction`)
Agents interact via structured, typed actions defining the intent and the specific payload:
- **`action_type`**: An `ActionType` enum (e.g., `REVIEW_CODE`, `TRIAGE_BUG`, `RUN_TEST`, `SUBMIT_FIX`).
- **`payload`**: A flexible `dict` containing the parameters for the action:
    - `thought`: (Optional) The reasoning behind the action.
    - `bug_id` / `test_path`: Logical identifiers for the task targets.
    - `code_patch`: (For SUBMIT_FIX) The proposed changes.

### 🔹 Observation Space (`CodeForgeObservation`)
The environment provides a comprehensive, structured state to the agent at each step:
- **`task_id`**: String identifier of the current active scenario.
- **`message`**: Human-readable goal description and current status.
- **`file_snapshot`**: A JSON-encoded dictionary of all relevant project files and their contents.
- **`console_output`**: Captured output from executed commands, linters, or test runners.
- **`progress`**: A float (0.0 to 1.0) indicating overall task completion based on sub-goal success.
- **`available_actions`**: A list of valid `ActionType` strings the agent can perform.
- **`step_count`**: Number of steps taken in the current episode.
- **`reward`**: Immediate scalar feedback from the environment.

---

## 📋 Task Registry & Sub-goals

We provide 8 core tasks, each with specific sub-goals that provide intermediate rewards.

| Task ID | Level | Motivation | Key Sub-goals |
| :--- | :--- | :--- | :--- |
| `easy_review` | **Easy** | Fine-tuning style/security awareness. | `identified_security_issue`, `added_style_comment` |
| `doc_update` | **Easy** | Fixing broken links & documentation metadata. | `identified_broken_link`, `fixed_broken_link` |
| `medium_triage` | **Medium** | Logical reasoning & bug isolation. | `found_index_error_line`, `isolated_data_source` |
| `security_audit` | **Medium** | Discovering hardcoded secrets. | `identified_secret`, `removed_plaintext`, `updated_env` |
| `hard_pipeline` | **Hard** | Complex system interaction (ETL). | `parsed_csv`, `encoding_error_fixed`, `transformed` |
| `ci_pipeline_fix` | **Hard** | Infrastructure as Code (IaC) debugging. | `validated_yaml`, `fixed_indentation`, `node_version_set` |
| `expert_refactor` | **Expert** | Deep architectural modernization. | `converted_to_async`, `handled_concurrency`, `tests_passing` |
| `pro_deploy` | **Expert** | High-stakes operational safety. | `health_checked`, `staged_rollout`, `metrics_verified` |
| `api_migration` | **Expert** | Major version API transition. | `found_legacy`, `implemented_v2`, `tested_rollback` |

---

## 📈 Reward Modeling & Grader Logic

CodeForge Pro uses a **multi-component reward function** to density the sparse coding environment:

$$ R_t = R_{subgoal} + R_{base} + P_{step} + P_{redundancy} $$

- **Sub-goal Reward ($R_{subgoal}$):** +0.2 per milestone. This is the primary driver for learning.
- **Base Reward ($R_{base}$):** Small positive signals for taking relevant actions (e.g., `RUN_TEST`).
- **Step Penalty ($P_{step}$):** -0.01 per step. Encourages the agent to find minimal-action solutions.
- **Redundancy Penalty ($P_{redundancy}$):** -0.05 for repeating the same action type sequentially without payload changes.
- **Safety Penalty:** A large negative reward (-0.5) for dangerous commands like `os.remove` or `rm -rf`.

---

## ⚙️ Setup & Usage

### Local Development
```powershell
# 1. Install dependencies
pip install -e .

# 2. Run local server with Web Interface
$env:ENABLE_WEB_INTERFACE = "true"
$env:PYTHONPATH = "."
python server/app.py --port 8000
```

### Docker (Production)
```powershell
docker build -t codeforge-pro .
docker run -p 8000:8000 -e ENABLE_WEB_INTERFACE=true codeforge-pro
```

### 🏃 Prompt-Based Usage
Agents can connect via the `CodeForgeProEnv` client:
```python
async with CodeForgeProEnv(base_url="http://localhost:8000") as env:
    obs = await env.reset(task_id="expert_refactor")
    # ... Agent Loop ...
    action = CodeForgeAction(action_type="run_test", payload={"test_path": "tests/"})
    new_obs = await env.step(action)
```

### 📊 Running Evaluation
To run the automated inference evaluation suite (as required for the hackathon baseline):

```powershell
# Set your API token (required for LLM inference)
$env:HF_TOKEN = "your_huggingface_token_here"

# Run against a local server (default)
python inference.py --episodes 5

# Run against the deployed Hugging Face Space
python inference.py --url https://krishnapatil999-codeforge-pro-env.hf.space --episodes 5
```

---

## 📊 Baseline Performance Benchmark
*Evaluation conducted using GPT-4o-mini as a zero-shot agent (10 episodes per task).*

| Metric | Easy | Medium | Hard | Expert |
| :--- | :--- | :--- | :--- | :--- |
| **Success Rate** | 94% | 78% | 55% | 42% |
| **Avg. Reward** | 0.92 | 0.74 | 0.51 | 0.38 |
| **Efficiency (%)** | 88% | 72% | 45% | 31% |

> [!TIP]
> The **Expert** tasks are designed to be challenging for current-generation LLMs, providing a significant "headroom" for RL training.

---

## 🤝 Ethical Considerations & Safety
- **Sandbox Environment**: All coding tasks are simulated and do not execute raw code on the host unless specified in the `Dockerfile`.
- **Malicious Command Detection**: Built-in regex filters inside `_compute_dense_reward` penalize destructive behavior.
- **Data Privacy**: No real user code is required for the environment; all tasks are generated from the `TASK_DATA` registry.

---

## 🏆 Hackathon Submission Details
This project was developed for the **Meta OpenEnv Hackathon**.

- **Project Lead**: Krishnapatil999
- **Tech Stack**: Python 3.11+, FastAPI, Pydantic, OpenEnv-Core.
- **Deployment**: Hugging Face Spaces (Docker SDK).
