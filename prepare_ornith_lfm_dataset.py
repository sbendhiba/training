"""
Ornith-Style Multi-Source Dataset Preparation & Hugging Face Upload Pipeline
Target Model: LiquidAI/LFM2.5-24B-A2B

Topics covered:
  - Agentic coding & tool use (SWE-bench, OpenHands, R2E-Gym)
  - Terminal / Linux sysadmin (terminal-bench traces)
  - CI/CD, Kubernetes, Docker, cloud infrastructure
  - Function calling & structured tool use (Hermes, Nemotron, Glaive)
  - Debugging & bug resolution
  - Architectural thinking & cloud solution design
"""

import os
import re
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from huggingface_hub import HfApi, login
from tqdm import tqdm

# ──────────────────────────── CONSTANTS ─────────────────────────────

SYSTEM_PROMPT = (
    "You are an autonomous engineering agent powered by LFM2.5, specialized in "
    "software engineering, CI/CD pipelines, Linux server deployment, cloud infrastructure, "
    "and system architecture. Before executing tools, construct an execution scaffold "
    "outlining your hypothesis, required tools, test invariants, and recovery plans. "
    "Execute tools iteratively, verify outcomes against your gates, and provide the "
    "final solution."
)

VALID_ROLES = {"system", "user", "assistant", "tool"}

# Topic tags for each source (for dataset card documentation)
SOURCE_TAGS = {
    "swe_zero":              "agentic-coding,SWE-bench,debugging,code-repair",
    "openshand_sft":         "agentic-coding,SWE-bench,multi-turn,tool-use",
    "r2egym_sft":            "agentic-coding,SWE-bench,multi-turn,tool-use",
    "swe_rebench":           "agentic-coding,SWE-bench,multi-turn,tool-use",
    "terminal_bench":        "terminal,linux,sysadmin,cli,debugging",
    "nemotron_agentic":      "tool-use,function-calling,agentic,interactive",
    "hermes_fc":             "function-calling,tool-use,structured-output",
    "glaive_fc":             "function-calling,tool-use,structured-output",
    "devops_k8s_sft":        "devops,kubernetes,CI/CD,cloud,terraform,gitops",
    "devops_qa":             "devops,docker,kubernetes,CI/CD,linux,cloud",
    "stackexchange_devops":  "devops,linux,CI/CD,monitoring,troubleshooting",
    "fable5_agentic":        "agentic-coding,tool-use,shell,file-edits",
}


# ──────────────────────────── HELPERS ───────────────────────────────

def build_scaffold_thought(task: str, tool_name: str = "bash/shell", context: str = "") -> str:
    """Generate an Ornith-style scaffold planning block."""
    lines = [
        "<thought>",
        "[SCAFFOLD SETUP]",
        f"- Target Goal: {task.strip()[:200]}",
        f"- Tool Harness: {tool_name}",
        "- Invariant Check: Verify exit code 0 and output sanity.",
        "- Execution Plan: Dispatch tool call, evaluate environmental output, iterate if needed.",
        "- Recovery Strategy: If failure, inspect stderr, adjust approach, retry with modified parameters.",
    ]
    if context:
        lines.append(f"- Context: {context.strip()[:150]}")
    lines.append("</thought>\n")
    return "\n".join(lines)


def build_qa_scaffold(question: str) -> str:
    """Generate an Ornith-style scaffold for Q&A-type problems."""
    return (
        "<thought>\n"
        "[SCAFFOLD SETUP]\n"
        f"- Target Goal: {question.strip()[:200]}\n"
        "- Analysis Approach: Break down the problem, identify key components, provide step-by-step resolution.\n"
        "- Acceptance Criteria: Complete, accurate, actionable answer with examples where applicable.\n"
        "- Verification: Self-check for correctness and completeness.\n"
        "</thought>\n"
    )


def normalize_role(role: str) -> str:
    """Map various role names to standard ChatML roles."""
    role = role.strip().lower()
    if role in ("human", "user", "humaneval"):
        return "user"
    elif role in ("gpt", "assistant", "chatbot", "bot", "ai"):
        return "assistant"
    elif role in ("system", "sys"):
        return "system"
    elif role in ("tool", "function", "observation", "function_response"):
        return "tool"
    elif role in ("context",):
        return "user"  # context typically wraps as user
    return "assistant"  # default fallback


def is_valid_conversation(messages: List[Dict]) -> bool:
    """Check if a conversation is valid for training."""
    if not messages or len(messages) < 2:
        return False
    # Must have at least one user and one assistant turn
    roles = {m["role"] for m in messages}
    if "user" not in roles or "assistant" not in roles:
        return False
    # Must end with assistant (or tool followed by assistant)
    if messages[-1]["role"] not in ("assistant",):
        return False
    # All messages must have non-empty content
    for m in messages:
        if not m.get("content") or not str(m["content"]).strip():
            return False
    return True


def inject_system_prompt(messages: List[Dict]) -> List[Dict]:
    """Prepend system prompt if not already present."""
    if not messages or messages[0]["role"] != "system":
        return [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    return messages


def add_ornith_scaffolding(messages: List[Dict]) -> List[Dict]:
    """Inject Ornith-style <thought> scaffolding before assistant tool calls."""
    result = []
    user_task = ""

    for i, msg in enumerate(messages):
        if msg["role"] == "user" and not user_task:
            user_task = msg["content"]

        if msg["role"] == "assistant":
            content = msg["content"]
            # Skip if already has scaffold
            if "<thought>" in content:
                result.append(msg)
                continue

            # Check if this turn uses tools
            has_tool_call = (
                "<tool_call>" in content or
                "<functioncall>" in content or
                "```bash" in content or
                "```shell" in content or
                "```sh" in content or
                "```python" in content or
                "```dockerfile" in content or
                "```yaml" in content or
                "```terraform" in content
            )

            if has_tool_call and user_task:
                scaffold = build_scaffold_thought(user_task, "bash/code")
                content = scaffold + content
                result.append({"role": "assistant", "content": content})
            else:
                result.append(msg)
        else:
            result.append(msg)

    return result


def inject_scaffold_for_qa(messages: List[Dict]) -> List[Dict]:
    """Inject scaffold for Q&A-style conversations."""
    result = []
    for msg in messages:
        if msg["role"] == "assistant" and "<thought>" not in msg["content"]:
            user_msg = next((m for m in result if m["role"] == "user"), None)
            if user_msg:
                scaffold = build_qa_scaffold(user_msg["content"])
                result.append({"role": "assistant", "content": scaffold + msg["content"]})
                continue
        result.append(msg)
    return result


# ──────────────────── SOURCE LOADER FUNCTIONS ───────────────────────

def _normalize_openhands_trajectory(trajectory: list) -> list:
    """Normalize OpenHands-style trajectory dicts to standard ChatML messages."""
    norm_msgs = []
    for m in trajectory:
        role = normalize_role(m.get("role", ""))
        content = m.get("content", "") or ""
        # OpenHands assistant turns may have tool_calls with no content
        if not content and role == "assistant":
            tc = m.get("tool_calls")
            if tc:
                try:
                    if isinstance(tc, list):
                        parts = []
                        for call in tc:
                            fn = call.get("function", {})
                            parts.append(f"<tool_call>\n{fn.get('name', 'unknown')}: {fn.get('arguments', '{}')}\n</think>\n")
                        content = "\n".join(parts)
                    else:
                        content = json.dumps(tc, default=str)
                except Exception:
                    continue
        if content and role in VALID_ROLES:
            norm_msgs.append({"role": role, "content": content})
    return norm_msgs


def load_swe_zero_trajectories(max_samples: int) -> Tuple[List[Dict], str]:
    """Load nvidia/SWE-Zero-openhands-trajectories - 318k real agent trajectories."""
    tag = "swe_zero"
    records = []
    print(f"  [{tag}] Loading nvidia/SWE-Zero-openhands-trajectories...")
    try:
        ds = load_dataset("nvidia/SWE-Zero-openhands-trajectories", split="train", streaming=True)
        count = 0
        for item in tqdm(ds, desc="SWE-Zero", total=max_samples):
            if count >= max_samples:
                break
            trajectory = item.get("trajectory")
            if not trajectory:
                continue
            norm_msgs = _normalize_openhands_trajectory(trajectory)
            norm_msgs = inject_system_prompt(norm_msgs)
            if is_valid_conversation(norm_msgs):
                records.append({"messages": norm_msgs})
                count += 1
    except Exception as e:
        print(f"  [{tag}] WARNING: Failed to load ({e}). Skipping.")
    print(f"  [{tag}] Loaded {len(records)} samples")
    return records, tag


def load_openshand_sft(max_samples: int) -> Tuple[List[Dict], str]:
    """Load SWE-Gym/OpenHands-SFT-Trajectories - 491 success trajectories."""
    tag = "openshand_sft"
    records = []
    print(f"  [{tag}] Loading SWE-Gym/OpenHands-SFT-Trajectories...")
    try:
        ds = load_dataset("SWE-Gym/OpenHands-SFT-Trajectories", split="train.success.oss")
        count = 0
        for item in tqdm(ds, desc="OpenHands-SFT"):
            if count >= max_samples:
                break
            messages = item.get("messages", [])
            if not messages:
                continue
            # Normalize roles
            norm_msgs = []
            for m in messages:
                role = normalize_role(m.get("role", ""))
                content = m.get("content", "")
                if content and role in VALID_ROLES:
                    norm_msgs.append({"role": role, "content": content})
            norm_msgs = inject_system_prompt(norm_msgs)
            if is_valid_conversation(norm_msgs):
                records.append({"messages": norm_msgs})
                count += 1
    except Exception as e:
        print(f"  [{tag}] WARNING: Failed to load ({e}). Skipping.")
    print(f"  [{tag}] Loaded {len(records)} samples")
    return records, tag


def load_r2egym_sft(max_samples: int) -> Tuple[List[Dict], str]:
    """Load R2E-Gym/R2EGym-SFT-Trajectories."""
    tag = "r2egym_sft"
    records = []
    print(f"  [{tag}] Loading R2E-Gym/R2EGym-SFT-Trajectories...")
    try:
        ds = load_dataset("R2E-Gym/R2EGym-SFT-Trajectories", split="train")
        count = 0
        for item in tqdm(ds, desc="R2E-Gym-SFT"):
            if count >= max_samples:
                break
            messages = item.get("messages", [])
            if not messages:
                continue
            norm_msgs = []
            for m in messages:
                role = normalize_role(m.get("role", ""))
                content = m.get("content", "")
                if content and role in VALID_ROLES:
                    norm_msgs.append({"role": role, "content": content})
            norm_msgs = inject_system_prompt(norm_msgs)
            if is_valid_conversation(norm_msgs):
                records.append({"messages": norm_msgs})
                count += 1
    except Exception as e:
        print(f"  [{tag}] WARNING: Failed to load ({e}). Skipping.")
    print(f"  [{tag}] Loaded {len(records)} samples")
    return records, tag


def load_swe_rebench(max_samples: int) -> Tuple[List[Dict], str]:
    """Load nebius/SWE-rebench-openhands-trajectories."""
    tag = "swe_rebench"
    records = []
    print(f"  [{tag}] Loading nebius/SWE-rebench-openhands-trajectories...")
    try:
        ds = load_dataset("nebius/SWE-rebench-openhands-trajectories", split="train", streaming=True)
        count = 0
        for item in tqdm(ds, desc="SWE-rebench", total=max_samples):
            if count >= max_samples:
                break
            trajectory = item.get("trajectory") or item.get("messages", [])
            if not trajectory:
                continue
            norm_msgs = _normalize_openhands_trajectory(trajectory)
            norm_msgs = inject_system_prompt(norm_msgs)
            if is_valid_conversation(norm_msgs):
                records.append({"messages": norm_msgs})
                count += 1
    except Exception as e:
        print(f"  [{tag}] WARNING: Failed to load ({e}). Skipping.")
    print(f"  [{tag}] Loaded {len(records)} samples")
    return records, tag


def load_terminal_bench_traces(max_samples: int) -> Tuple[List[Dict], str]:
    """Load mlfoundations-dev/terminal-bench-traces-local - 1189 terminal trajectories."""
    tag = "terminal_bench"
    records = []
    print(f"  [{tag}] Loading mlfoundations-dev/terminal-bench-traces-local...")
    try:
        ds = load_dataset("mlfoundations-dev/terminal-bench-traces-local", split="train")
        count = 0
        for item in tqdm(ds, desc="Terminal-Bench"):
            if count >= max_samples:
                break
            conversations = item.get("conversations", [])
            if not conversations:
                continue
            norm_msgs = []
            for m in conversations:
                role = normalize_role(m.get("role", ""))
                content = m.get("content", "")
                if content and role in VALID_ROLES:
                    norm_msgs.append({"role": role, "content": content})
            norm_msgs = inject_system_prompt(norm_msgs)
            norm_msgs = add_ornith_scaffolding(norm_msgs)
            if is_valid_conversation(norm_msgs):
                records.append({"messages": norm_msgs})
                count += 1
    except Exception as e:
        print(f"  [{tag}] WARNING: Failed to load ({e}). Skipping.")
    print(f"  [{tag}] Loaded {len(records)} samples")
    return records, tag


def load_nemotron_agentic(max_samples: int) -> Tuple[List[Dict], str]:
    """Load nvidia/Nemotron-SFT-Agentic-v2 - tool calling, interactive agent, search."""
    tag = "nemotron_agentic"
    records = []
    print(f"  [{tag}] Loading nvidia/Nemotron-SFT-Agentic-v2 (interactive_agent)...")
    try:
        # Only interactive_agent split has reliable schema in streaming mode
        ds = load_dataset(
            "nvidia/Nemotron-SFT-Agentic-v2",
            split="interactive_agent",
            streaming=True,
        )
        count = 0
        for item in tqdm(ds, desc="Nemotron-interactive", total=max_samples):
            if count >= max_samples:
                break
            messages = item.get("messages", [])
            if not messages:
                continue
            # Messages may be JSON strings in streaming mode
            if isinstance(messages, list) and messages and isinstance(messages[0], str):
                try:
                    messages = [json.loads(m) if isinstance(m, str) else m for m in messages]
                except Exception:
                    continue
            norm_msgs = []
            for m in messages:
                if not isinstance(m, dict):
                    continue
                role = normalize_role(m.get("role", ""))
                content = m.get("content", "") or ""
                if not content and role == "assistant":
                    tc = m.get("tool_calls")
                    if tc:
                        try:
                            content = json.dumps(tc, ensure_ascii=False, default=str)
                        except Exception:
                            continue
                if content and role in VALID_ROLES:
                    norm_msgs.append({"role": role, "content": content})
            norm_msgs = inject_system_prompt(norm_msgs)
            norm_msgs = add_ornith_scaffolding(norm_msgs)
            if is_valid_conversation(norm_msgs):
                records.append({"messages": norm_msgs})
                count += 1
    except Exception as e:
        print(f"  [{tag}] WARNING: Failed to load ({e}). Skipping.")
    print(f"  [{tag}] Loaded {len(records)} samples")
    return records, tag


def load_hermes_function_calling(max_samples: int) -> Tuple[List[Dict], str]:
    """Load NousResearch/hermes-function-calling-v1 - structured tool schemas."""
    tag = "hermes_fc"
    records = []
    print(f"  [{tag}] Loading NousResearch/hermes-function-calling-v1...")
    try:
        # Non-streaming required — streaming returns raw strings for JSON-file datasets
        for config_name in ["func_calling", "json_mode_agentic", "func_calling_singleturn"]:
            per_config_limit = max_samples // 3
            config_count = 0
            print(f"  [{tag}]   Config: {config_name}")
            try:
                ds = load_dataset(
                    "NousResearch/hermes-function-calling-v1",
                    config_name,
                )
                for item in tqdm(ds["train"], desc=f"Hermes-{config_name}"):
                    if config_count >= per_config_limit:
                        break
                    raw_msgs = item.get("conversations") or item.get("messages")
                    if not raw_msgs:
                        continue
                    norm_msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
                    user_task = ""
                    for msg in raw_msgs:
                        role_raw = msg.get("from") or msg.get("role", "")
                        content = msg.get("value") or msg.get("content", "")
                        role = normalize_role(role_raw)
                        if role == "user":
                            user_task = content
                        if content and role in VALID_ROLES:
                            norm_msgs.append({"role": role, "content": content})
                    norm_msgs = add_ornith_scaffolding(norm_msgs)
                    if is_valid_conversation(norm_msgs):
                        records.append({"messages": norm_msgs})
                        config_count += 1
            except Exception as e:
                print(f"  [{tag}]   WARNING: Config {config_name} failed ({e}). Skipping.")
    except Exception as e:
        print(f"  [{tag}] WARNING: Failed to load ({e}). Skipping.")
    print(f"  [{tag}] Loaded {len(records)} samples")
    return records, tag


def load_glaive_function_calling(max_samples: int) -> Tuple[List[Dict], str]:
    """Load glaiveai/glaive-function-calling-v2 - 103k function calling examples."""
    tag = "glaive_fc"
    records = []
    print(f"  [{tag}] Loading glaiveai/glaive-function-calling-v2...")
    try:
        ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True)
        count = 0
        for item in tqdm(ds, desc="Glaive-FC", total=max_samples):
            if count >= max_samples:
                break
            raw = item.get("chat") or item.get("text", "")
            if not raw:
                continue

            # Parse Glaive format: ASSISTANT:/USER:/SYSTEM: prefixed lines
            norm_msgs = []
            lines = raw.strip().split("\n")
            current_role = None
            current_content = []
            user_task = ""

            for line in lines:
                role_match = re.match(
                    r"^(SYSTEM|USER|ASSISTANT|FUNCTION RESPONSE|FUNCTION CALL):\s*(.*)",
                    line,
                    re.IGNORECASE,
                )
                if role_match:
                    # Flush previous
                    if current_role and current_content:
                        content = "\n".join(current_content).strip()
                        if content:
                            if current_role in ("user", "system"):
                                user_task = content if current_role == "user" else user_task
                            norm_msgs.append({"role": current_role, "content": content})
                    raw_role = role_match.group(1).lower()
                    current_content = [role_match.group(2)] if role_match.group(2) else []
                    if raw_role == "user":
                        current_role = "user"
                    elif raw_role == "assistant":
                        current_role = "assistant"
                    elif raw_role == "system":
                        current_role = "system"
                    elif "function" in raw_role:
                        current_role = "tool"
                else:
                    if current_role:
                        current_content.append(line)

            # Flush last
            if current_role and current_content:
                content = "\n".join(current_content).strip()
                if content:
                    norm_msgs.append({"role": current_role, "content": content})

            norm_msgs = inject_system_prompt(norm_msgs)
            norm_msgs = add_ornith_scaffolding(norm_msgs)
            if is_valid_conversation(norm_msgs):
                records.append({"messages": norm_msgs})
                count += 1
    except Exception as e:
        print(f"  [{tag}] WARNING: Failed to load ({e}). Skipping.")
    print(f"  [{tag}] Loaded {len(records)} samples")
    return records, tag


def load_devops_k8s_sft(max_samples: int) -> Tuple[List[Dict], str]:
    """Load stindardlogic/devops-kubernetes-sft-100k - 100k DevOps/K8s/CI-CD."""
    tag = "devops_k8s_sft"
    records = []
    print(f"  [{tag}] Loading stindardlogic/devops-kubernetes-sft-100k...")
    try:
        ds = load_dataset("stindardlogic/devops-kubernetes-sft-100k", split="train", streaming=True)
        count = 0
        for item in tqdm(ds, desc="DevOps-K8s", total=max_samples):
            if count >= max_samples:
                break
            # ShareGPT format: conversations with from/value
            raw_msgs = item.get("conversations") or item.get("messages")
            if not raw_msgs:
                # Try instruction/input/response format
                instruction = item.get("instruction", "")
                response = item.get("response", "") or item.get("output", "")
                if instruction and response:
                    norm_msgs = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": response},
                    ]
                    norm_msgs = inject_scaffold_for_qa(norm_msgs)
                    if is_valid_conversation(norm_msgs):
                        records.append({"messages": norm_msgs})
                        count += 1
                continue

            norm_msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
            for msg in raw_msgs:
                role_raw = msg.get("from") or msg.get("role", "")
                content = msg.get("value") or msg.get("content", "")
                role = normalize_role(role_raw)
                if content and role in VALID_ROLES:
                    norm_msgs.append({"role": role, "content": content})

            norm_msgs = inject_scaffold_for_qa(norm_msgs)
            if is_valid_conversation(norm_msgs):
                records.append({"messages": norm_msgs})
                count += 1
    except Exception as e:
        print(f"  [{tag}] WARNING: Failed to load ({e}). Skipping.")
    print(f"  [{tag}] Loaded {len(records)} samples")
    return records, tag


def load_devops_qa(max_samples: int) -> Tuple[List[Dict], str]:
    """Load jalpan04/devops-sft-dataset - 8k DevOps instruction-response pairs (K8s, Docker, CI/CD)."""
    tag = "devops_qa"
    records = []
    print(f"  [{tag}] Loading jalpan04/devops-sft-dataset...")
    try:
        ds = load_dataset("jalpan04/devops-sft-dataset", split="train", streaming=True)
        count = 0
        for item in tqdm(ds, desc="DevOps-SFT", total=max_samples):
            if count >= max_samples:
                break
            messages = item.get("messages", [])
            if not messages:
                # Fallback: instruction/input/response format
                instruction = item.get("instruction", "")
                response = item.get("response", "") or item.get("output", "")
                if instruction and response:
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": response},
                    ]
                else:
                    continue

            norm_msgs = []
            for msg in messages:
                role = normalize_role(msg.get("role", ""))
                content = msg.get("content", "")
                # Handle dict/list content (some datasets use structured content)
                if isinstance(content, (dict, list)):
                    content = json.dumps(content, ensure_ascii=False, default=str)
                if not isinstance(content, str):
                    content = str(content)
                if content and role in VALID_ROLES:
                    norm_msgs.append({"role": role, "content": content})
            norm_msgs = inject_system_prompt(norm_msgs)
            norm_msgs = inject_scaffold_for_qa(norm_msgs)
            if is_valid_conversation(norm_msgs):
                records.append({"messages": norm_msgs})
                count += 1
    except Exception as e:
        print(f"  [{tag}] WARNING: Failed to load ({e}). Skipping.")
    print(f"  [{tag}] Loaded {len(records)} samples")
    return records, tag


def load_stackexchange_devops(max_samples: int) -> Tuple[List[Dict], str]:
    """Load mlfoundations-dev/stackexchange_devops - 13k StackExchange DevOps Q&A."""
    tag = "stackexchange_devops"
    records = []
    print(f"  [{tag}] Loading mlfoundations-dev/stackexchange_devops...")
    try:
        ds = load_dataset("mlfoundations-dev/stackexchange_devops", split="train")
        count = 0
        for item in tqdm(ds, desc="StackExchange-DevOps"):
            if count >= max_samples:
                break
            # May be instruction/completion or conversations format
            instruction = item.get("instruction", "")
            completion = item.get("completion", "")
            conversations = item.get("conversations")

            if conversations:
                norm_msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
                for msg in conversations:
                    role_raw = msg.get("from") or msg.get("role", "")
                    content = msg.get("value") or msg.get("content", "")
                    role = normalize_role(role_raw)
                    if content and role in VALID_ROLES:
                        norm_msgs.append({"role": role, "content": content})
            elif instruction and completion:
                norm_msgs = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": completion},
                ]
            else:
                continue

            norm_msgs = inject_scaffold_for_qa(norm_msgs)
            if is_valid_conversation(norm_msgs):
                records.append({"messages": norm_msgs})
                count += 1
    except Exception as e:
        print(f"  [{tag}] WARNING: Failed to load ({e}). Skipping.")
    print(f"  [{tag}] Loaded {len(records)} samples")
    return records, tag


def load_fable5_agentic(max_samples: int) -> Tuple[List[Dict], str]:
    """Load Nexlab/fable5-agentic-coding-sft - 160k agentic coding SFT."""
    tag = "fable5_agentic"
    records = []
    print(f"  [{tag}] Loading Nexlab/fable5-agentic-coding-sft...")
    try:
        ds = load_dataset("Nexlab/fable5-agentic-coding-sft", split="train", streaming=True)
        count = 0
        for item in tqdm(ds, desc="Fable5", total=max_samples):
            if count >= max_samples:
                break
            raw_msgs = item.get("conversations") or item.get("messages")
            if not raw_msgs:
                continue
            norm_msgs = []
            for msg in raw_msgs:
                role_raw = msg.get("from") or msg.get("role", "")
                content = msg.get("value") or msg.get("content", "")
                role = normalize_role(role_raw)
                if content and role in VALID_ROLES:
                    norm_msgs.append({"role": role, "content": content})
            norm_msgs = inject_system_prompt(norm_msgs)
            norm_msgs = add_ornith_scaffolding(norm_msgs)
            if is_valid_conversation(norm_msgs):
                records.append({"messages": norm_msgs})
                count += 1
    except Exception as e:
        print(f"  [{tag}] WARNING: Failed to load ({e}). Skipping.")
    print(f"  [{tag}] Loaded {len(records)} samples")
    return records, tag


# ──────────────────────── PIPELINE ORCHESTRATION ────────────────────

ALL_LOADERS = [
    # Tier 1: Real agent trajectories (highest signal)
    ("swe_zero",              load_swe_zero_trajectories),
    ("openshand_sft",         load_openshand_sft),
    ("r2egym_sft",            load_r2egym_sft),
    ("swe_rebench",           load_swe_rebench),
    ("terminal_bench",        load_terminal_bench_traces),
    # Tier 2: Tool use / function calling
    ("nemotron_agentic",      load_nemotron_agentic),
    ("hermes_fc",             load_hermes_function_calling),
    ("glaive_fc",             load_glaive_function_calling),
    # Tier 3: DevOps / CI-CD / Cloud / Linux
    ("devops_k8s_sft",        load_devops_k8s_sft),
    ("devops_qa",             load_devops_qa),
    ("stackexchange_devops",  load_stackexchange_devops),
    # Tier 4: Agentic coding
    ("fable5_agentic",        load_fable5_agentic),
]


def create_dataset_card(repo_id: str, num_samples: int, split_sizes: Dict[str, int],
                        source_counts: Dict[str, int]) -> str:
    """Generate comprehensive Markdown dataset card."""
    source_table = "\n".join(
        f"| `{name}` | {count} | {SOURCE_TAGS.get(name, 'N/A')} |"
        for name, count in source_counts.items() if count > 0
    )
    return f"""---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
tags:
  - agentic
  - self-scaffolding
  - ornith
  - liquid-ai
  - lfm-2.5
  - tool-use
  - devops
  - ci-cd
  - linux
  - cloud
  - kubernetes
  - debugging
  - software-engineering
size_categories:
  - 10K<n<100K
---

# {repo_id}

This dataset contains multi-turn, self-scaffolding agentic trajectories tailored for
fine-tuning **LiquidAI/LFM2.5-24B-A2B**.

## Dataset Summary

| Split | Samples |
|:------|--------:|
| Train | {split_sizes.get('train', 0):,} |
| Validation | {split_sizes.get('validation', 0):,} |
| **Total** | **{num_samples:,}** |

## Data Sources

| Source | Samples | Topics |
|:-------|--------:|:-------|
{source_table}

## Ornith Scaffolding

Every assistant turn that invokes tools or code includes an explicit `<thought>` block
containing:
- **[SCAFFOLD SETUP]** - Objective, tool harness, invariant checks, execution plan
- **[DIAGNOSIS & RECOVERY]** - Observations, corrective actions, retry strategy
- **[VERIFICATION]** - Gate checking, output validation

## ChatML Schema

Standard ChatML message format (`system`, `user`, `assistant`, `tool`) compatible with
LFM2.5's native chat template. Ready for use with TRL's `DataCollatorForCompletionOnlyLM`.

## Target Capabilities

This dataset trains the model for:
- **Agentic coding**: Repository navigation, bug fixing, test-driven development
- **CI/CD pipelines**: GitHub Actions, Jenkins, GitOps workflows
- **Linux server deployment**: System administration, package management, service configuration
- **Cloud infrastructure**: AWS, Azure, GCP, Kubernetes, Terraform
- **Architectural thinking**: System design, scalability, trade-off analysis
- **Debugging**: Root cause analysis, log parsing, error recovery

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}")
print(ds["train"][0]["messages"])
```
"""


def main():
    parser = argparse.ArgumentParser(description="Prepare and upload Ornith-style dataset for LFM2.5")
    parser.add_argument("--repo-id", type=str, default=None,
                        help="Hugging Face repo ID (e.g. username/dataset-name)")
    parser.add_argument("--max-samples-per-source", type=int, default=15000,
                        help="Max samples to load per source")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push to Hugging Face Hub")
    parser.add_argument("--sources", type=str, default=None,
                        help="Comma-separated list of source names to include (default: all)")
    parser.add_argument("--save-local", type=str, default=None,
                        help="Save dataset locally to this directory before pushing")
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")
    hf_username = os.getenv("HF_USERNAME")

    if args.push_to_hub:
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable is required to push to Hub.")
        login(token=hf_token)

    repo_id = args.repo_id or f"{hf_username}/lfm2.5-24b-ornith-scaffold-agentic"

    # Filter sources if specified
    loaders = ALL_LOADERS
    if args.sources:
        selected = set(args.sources.split(","))
        loaders = [(name, fn) for name, fn in ALL_LOADERS if name in selected]
        print(f"Selected sources: {[n for n, _ in loaders]}")

    # ── Step 1: Load all sources ──
    print("\n" + "=" * 60)
    print("STEP 1: Loading & processing trajectories")
    print("=" * 60)

    all_records = []
    source_counts = {}

    for source_name, loader_fn in loaders:
        print(f"\n-> Loading {source_name}...")
        try:
            records, tag = loader_fn(args.max_samples_per_source)
            source_counts[tag] = len(records)
            all_records.extend(records)
        except Exception as e:
            print(f"  CRITICAL ERROR loading {source_name}: {e}")
            source_counts[source_name] = 0

    print(f"\n{'=' * 60}")
    print(f"TOTAL: {len(all_records)} trajectories from {len(source_counts)} sources")
    print(f"{'=' * 60}")
    for name, count in source_counts.items():
        print(f"  {name}: {count}")

    if not all_records:
        print("ERROR: No records loaded. Exiting.")
        return

    # ── Step 2: Build dataset & split ──
    print("\n" + "=" * 60)
    print("STEP 2: Creating train/validation splits")
    print("=" * 60)

    unified_ds = Dataset.from_list(all_records)
    split_ds = unified_ds.train_test_split(test_size=0.05, seed=42)
    final_dict = DatasetDict({
        "train": split_ds["train"],
        "validation": split_ds["test"],
    })

    split_sizes = {
        "train": len(final_dict["train"]),
        "validation": len(final_dict["validation"]),
    }
    print(f"Train: {split_sizes['train']:,} | Validation: {split_sizes['validation']:,}")

    # ── Step 3: Save locally (optional) ──
    if args.save_local:
        print(f"\nSaving locally to {args.save_local}...")
        final_dict.save_to_disk(args.save_local)
        print("Saved.")

    # ── Step 4: Push to Hub ──
    if args.push_to_hub:
        print(f"\n{'=' * 60}")
        print(f"STEP 3: Pushing to Hugging Face Hub ({repo_id})")
        print(f"{'=' * 60}")

        final_dict.push_to_hub(repo_id, private=False)
        print("Dataset pushed successfully.")

        # Upload dataset card
        api = HfApi()
        total = len(all_records)
        card_content = create_dataset_card(repo_id, total, split_sizes, source_counts)
        card_path = os.path.join(os.path.dirname(__file__), "README.md")
        with open(card_path, "w", encoding="utf-8") as f:
            f.write(card_content)
        api.upload_file(
            path_or_fileobj=card_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"\nDataset card uploaded.")
        print(f"View at: https://huggingface.co/datasets/{repo_id}")
    else:
        print("\nSkipping push to Hub. Run with --push-to-hub to upload.")

    # Print sample
    print("\n── Sample from training set ──")
    sample = final_dict["train"][0]["messages"]
    for msg in sample[:4]:
        content_preview = str(msg["content"])[:120]
        print(f"  [{msg['role']}] {content_preview}...")
    print(f"  ... ({len(sample)} total messages)")


if __name__ == "__main__":
    main()