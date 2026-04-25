import json
import re

from edge_rag.inference.base import InferenceClient

FAITHFULNESS_SYSTEM_PROMPT = """You are a metric extraction system. Your task is to evaluate the faithfulness of an answer given its context.
Extract all claims from the generated answer. Next, check if each claim is supported by the context snippets.
Output your response STRICTLY as a raw JSON object and nothing else, with exactly this format:
{
  "total_claims": <int>,
  "supported_claims": <int>
}
"""

def evaluate_faithfulness(inference_client: InferenceClient, query: str, answer: str, context: str) -> tuple[float, int, int]:
    if not answer or answer.strip() == "I do not know." or not context:
        return 0.0, 0, 0
        
    prompt = (
        f"Question:\n{query}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:\n{answer}\n\n"
        "Evaluate the faithfulness of the answer. Remember to only output valid JSON."
    )
    
    try:
        result = inference_client.generate(
            prompt=prompt,
            system_prompt=FAITHFULNESS_SYSTEM_PROMPT,
            max_tokens=1024,
            temperature=0.0
        )
        
        # Try finding JSON block
        text = result.text.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group()
            
        data = json.loads(text)
        total = int(data.get("total_claims", 0))
        supported = int(data.get("supported_claims", 0))
        
        if total == 0:
            return 1.0, 0, 0
            
        score = float(supported) / total
        return score, supported, total
    except Exception as e:
        print(f"Failed to evaluate faithfulness: {e}")
        return 0.0, 0, 0
