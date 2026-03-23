import os
import json
import re
from dotenv import load_dotenv

load_dotenv()

PROOF_SYSTEM_PROMPT = """You are a rigorous mathematical proof assistant with expertise across all areas of mathematics.

Given a theorem and context, you will:
1. Identify the most appropriate proof strategy
2. Decompose the proof into clear, numbered steps
3. For each step, provide: the claim, the mathematical justification, and any definitions used
4. Identify any gaps, assumptions, or sub-lemmas that need separate proofs
5. Assess your confidence in the proof's correctness

Return your response as a JSON object with this exact structure:
{
  "theorem": "<restate the theorem precisely>",
  "strategy": "<chosen proof strategy and why>",
  "key_ideas": ["<main insight 1>", "<main insight 2>"],
  "steps": [
    {
      "step_num": 1,
      "claim": "<what we show in this step>",
      "justification": "<mathematical reasoning>",
      "definitions_used": ["<def 1>", "<def 2>"]
    }
  ],
  "gaps": ["<any unproven assumptions or gaps>"],
  "needed_lemmas": ["<lemma 1 that needs separate proof>"],
  "confidence": "high|medium|low",
  "notes": "<any additional mathematical insights>"
}

Be rigorous. Clearly distinguish between what is proven and what is assumed."""


def proof_assist(
    theorem: str,
    context: str = "",
    strategy: str = "auto",
    mode: str = "detailed",
) -> dict:
    from llm import get_client

    proof_model = os.environ.get("PROOF_MODEL", "").strip() or None
    client = get_client(model_override=proof_model)

    mode_instruction = {
        "outline": "Provide a high-level outline with 3-6 key steps only. Keep steps brief.",
        "detailed": "Provide a complete, detailed proof with full justifications for each step.",
        "lemmas": "Focus on identifying all lemmas and sub-results needed. List them with brief descriptions.",
    }.get(mode, "Provide a complete detailed proof.")

    strategy_instruction = (
        f"Use the '{strategy}' proof strategy."
        if strategy != "auto"
        else "Choose the most elegant and appropriate proof strategy."
    )

    user_content = f"""Theorem: {theorem}

Mathematical context/known results:
{context or "Standard mathematical axioms and well-known theorems may be used freely."}

Instructions:
- {strategy_instruction}
- {mode_instruction}
- Return your response as valid JSON matching the specified structure."""

    result = client.chat(
        system=PROOF_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
        max_tokens=4096,
    )

    # Extract text from content blocks
    text = ""
    for block in result["content_blocks"]:
        if block["type"] == "text":
            text += block["text"]

    # Try to parse JSON from response
    try:
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            proof_data = json.loads(json_match.group(1))
        else:
            proof_data = json.loads(text)
    except (json.JSONDecodeError, AttributeError):
        proof_data = {
            "theorem": theorem,
            "strategy": strategy,
            "raw_proof": text,
            "confidence": "unknown",
        }

    return proof_data
