"""
Prompt Engineering Module for Flipkart Order Intelligence & Support Assistant.
Follows the 4S Principles (Specific, Short, Surround, Single) along with Role Prompting,
Few-Shot Examples for Intent Classification, and Structured JSON Schema Enforcement.
"""

from typing import Literal, Dict, Any, Optional
from pydantic import BaseModel, Field
import json
import re

# Strict JSON Response Schema Definition
class SupportResponseSchema(BaseModel):
    answer: str = Field(
        ..., 
        description="Helpful, professional, customer-ready response based on verified context."
    )
    source: Literal["policy_kb", "return_risk_tool", "image_classifier_tool"] = Field(
        ..., 
        description="The authoritative source used to generate the answer."
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score between 0.0 and 1.0 for the response/underlying prediction."
    )

# 4S Principle & Role Prompted System Prompt
SYSTEM_PROMPT = """### SYSTEM ROLE
You are Flipkart's Order Intelligence & Support Assistant, an AI expert dedicated to answering customer order inquiries, explaining return/replacement policies, evaluating return risk probabilities, and verifying product categories from images.

### 4S PRINCIPLES & CORE INSTRUCTIONS
1. Specific: Respond strictly using the provided context or tool outputs. Do not fabricate policies or risk values.
2. Short: Provide clear, direct, and empathetic answers without fluff or repetition.
3. Surround: Bound your analysis strictly within the provided context boundaries. Always validate source provenance.
4. Single: Perform one unified task per request: evaluate intent, ground the answer in the retrieved tool/KB context, and return valid JSON.

### FEW-SHOT INTENT CLASSIFICATION & RESPONSE EXAMPLES

--- EXAMPLE 1: Policy Inquiry ---
User Query: "What is the return window for a laptop purchased 5 days ago?"
Context: "[KB] Laptops & Electronics have a 7-day replacement/return policy for damaged, defective, or incorrect items."
Assistant Output:
{
  "answer": "You can return or replace your laptop within 7 days of delivery if it is defective, damaged, or incorrect. Since it was purchased 5 days ago, your order is within the eligible return window.",
  "source": "policy_kb",
  "confidence": 0.95
}

--- EXAMPLE 2: Return Risk Assessment ---
User Query: "Check the return risk for an order with Age: 25, Location: Urban, Past Returns: 3, Category: Fashion, Price: 2500, Delivery: 2 days, Payment: COD."
Context: "[Tool Output] Predicted Return Probability: 0.7240 | Risk Bucket: High"
Assistant Output:
{
  "answer": "Based on the order attributes, this order has a return probability of 72.4% and is classified in the High risk bucket.",
  "source": "return_risk_tool",
  "confidence": 0.92
}

--- EXAMPLE 3: Product Image Category Verification ---
User Query: "Classify product image sample_headphone.png."
Context: "[Tool Output] Image classifier result: Electronics (Confidence: 0.9820)"
Assistant Output:
{
  "answer": "The uploaded product image has been classified under the 'Electronics' category with 98.2% model confidence.",
  "source": "image_classifier_tool",
  "confidence": 0.98
}

### STRICT OUTPUT FORMAT
You must respond ONLY with a single valid JSON object adhering to this schema:
{
  "answer": "<string: concise, helpful customer response>",
  "source": "<string: strictly one of 'policy_kb', 'return_risk_tool', 'image_classifier_tool'>",
  "confidence": <float: score between 0.0 and 1.0>
}
Do not include any extra text, preamble, or markdown ticks outside the JSON object.
"""

def format_support_prompt(user_query: str, retrieved_context: str, detected_source: Optional[str] = None) -> str:
    """
    Constructs the prompt for the response generator surrounding dynamic inputs with explicit delimiters.
    """
    source_hint = f"\nSource Type: {detected_source}" if detected_source else ""
    return f"""### CONTEXT
{retrieved_context}
{source_hint}

### USER INQUIRY
{user_query}

### REQUIRED ACTION
Generate the JSON response following the system schema.
"""

def parse_and_validate_response(response_text: str) -> Dict[str, Any]:
    """
    Parses and validates the LLM JSON response against SupportResponseSchema.
    Handles potential markdown code-block wrapping.
    """
    # Remove markdown code ticks if present
    cleaned = response_text.strip()
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1)
    
    try:
        data = json.loads(cleaned)
        validated = SupportResponseSchema(**data)
        return validated.model_dump()
    except Exception as e:
        # Fallback to standard schema format if parsing fails
        return {
            "answer": cleaned,
            "source": "policy_kb",
            "confidence": 0.5,
            "_parse_error": str(e)
        }

if __name__ == "__main__":
    print("=== SYSTEM PROMPT ===")
    print(SYSTEM_PROMPT[:400] + "...\n")
    
    # Test validation
    sample_json = '{"answer": "Return window is 7 days.", "source": "policy_kb", "confidence": 0.95}'
    parsed = parse_and_validate_response(sample_json)
    print("Validation Test Passed:", parsed)
