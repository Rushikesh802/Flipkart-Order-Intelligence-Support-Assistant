"""
MOCK_LLM Deterministic Mode Module for Flipkart Order Intelligence & Support Assistant.
Provides deterministic, rule-based/template response generation with zero external network calls
and zero API keys. Adheres strictly to the fixed JSON schema: answer, source, confidence.
"""

import os
import sys
import json
from typing import Dict, Any, Optional

# Add root directory to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from support_agent.prompts import SupportResponseSchema

# Default deterministic execution mode flag (defaults to True for offline evaluation/grading)
MOCK_LLM_ENABLED = os.getenv("MOCK_LLM", "true").lower() in ("1", "true", "yes", "y")


def generate_deterministic_response(
    intent: str,
    query: str,
    context: str = "",
    tool_output: Optional[Dict[str, Any]] = None,
    order_context: Optional[Dict[str, Any]] = None
) -> SupportResponseSchema:
    """
    Deterministically composes the final structured response based on intent,
    retrieved KB chunks, and/or tool outputs with zero network calls and zero API keys.
    
    Args:
        intent (str): The classified intent ('policy', 'return_risk', 'product_category', 'general').
        query (str): The raw user query.
        context (str): Retrieved KB chunk text or status message.
        tool_output (dict, optional): Direct output dictionary from tools.
        order_context (dict, optional): Contextual order features.
        
    Returns:
        SupportResponseSchema: Validated Pydantic model with fields (answer, source, confidence).
    """
    intent = (intent or "general").lower()
    
    # 1. Policy Knowledge Base Response Generation
    if intent in ("policy", "policy_kb"):
        if not context or "no relevant policy found" in context.lower() or context.strip() == "":
            answer = (
                "I could not locate a specific policy covering your query in our knowledge base. "
                "Generally, Flipkart offers a 7-day to 10-day return or replacement window on eligible items. "
                "Please check your order details or contact customer care for item-specific guidelines."
            )
            confidence = 0.40
        else:
            cleaned_context = context.strip().replace("\n", " ")
            # Construct a clear, professional summary answer from retrieved KB chunks
            answer = (
                f"Based on Flipkart's official policy: {cleaned_context} "
                "You can initiate a return or replacement request directly from the 'My Orders' section."
            )
            confidence = 0.95
            
        return SupportResponseSchema(
            answer=answer,
            source="policy_kb",
            confidence=confidence
        )

    # 2. Return Risk Tool Response Generation
    elif intent in ("return_risk", "return_risk_tool"):
        if tool_output and "predicted_probability" in tool_output:
            prob = tool_output["predicted_probability"]
            bucket = tool_output.get("risk_bucket", "Medium")
            answer = (
                f"Order return risk evaluation complete. The estimated return probability is "
                f"{prob * 100:.1f}%, classifying this order into the '{bucket}' risk bucket."
            )
            confidence = 0.92
        elif "probability=" in context.lower() or "bucket=" in context.lower():
            answer = f"Order return risk analysis: {context.strip()}"
            confidence = 0.92
        elif not order_context:
            answer = (
                "Please provide the order details (such as price, category, payment method, "
                "delivery days, and past return history) to evaluate the return risk."
            )
            confidence = 0.60
        else:
            answer = f"Return risk evaluation status: {context.strip()}"
            confidence = 0.80

        return SupportResponseSchema(
            answer=answer,
            source="return_risk_tool",
            confidence=confidence
        )

    # 3. Product Image Categoriser Tool Response Generation
    elif intent in ("product_category", "image_classifier_tool"):
        if tool_output and "predicted_category" in tool_output:
            cat = tool_output["predicted_category"]
            conf = float(tool_output.get("confidence", 0.90))
            answer = (
                f"The product image has been analyzed and classified under the '{cat}' category "
                f"with {conf * 100:.2f}% model confidence."
            )
            confidence = round(conf, 4)
        elif "result for" in context.lower() or "confidence:" in context.lower():
            answer = f"Product image classification result: {context.strip()}"
            confidence = 0.90
        elif "not found" in context.lower():
            answer = f"The requested image file could not be found. {context.strip()}"
            confidence = 0.30
        else:
            answer = (
                "Please provide a valid image filename (e.g., sample_headphone.png) to classify "
                "the product category."
            )
            confidence = 0.50

        return SupportResponseSchema(
            answer=answer,
            source="image_classifier_tool",
            confidence=confidence
        )

    # 4. Fallback / General Inquiry Response Generation
    else:
        answer = (
            "Hello! I am Flipkart's Order Intelligence & Support Assistant. "
            "I can assist you with:\n"
            "1. Checking Flipkart return, replacement, and refund policies\n"
            "2. Assessing order return risk and probability\n"
            "3. Verifying and classifying product categories from images"
        )
        return SupportResponseSchema(
            answer=answer,
            source="policy_kb",
            confidence=0.50
        )


def get_deterministic_json_response(
    intent: str,
    query: str,
    context: str = "",
    tool_output: Optional[Dict[str, Any]] = None,
    order_context: Optional[Dict[str, Any]] = None,
    indent: int = 2
) -> str:
    """
    Returns the deterministic response as a validated, formatted JSON string.
    """
    schema_obj = generate_deterministic_response(
        intent=intent,
        query=query,
        context=context,
        tool_output=tool_output,
        order_context=order_context
    )
    return json.dumps(schema_obj.model_dump(), indent=indent)


if __name__ == "__main__":
    print("=== Testing MOCK_LLM Deterministic Mode ===")
    
    # 1. Test Policy KB
    res_kb = get_deterministic_json_response(
        intent="policy",
        query="What is the return window for electronics?",
        context="Electronics items have a 7-day replacement guarantee for technical defects."
    )
    print("\n[Policy KB Output]:")
    print(res_kb)
    
    # 2. Test Return Risk Tool
    res_risk = get_deterministic_json_response(
        intent="return_risk",
        query="Check risk for my order",
        tool_output={"predicted_probability": 0.442, "risk_bucket": "Medium"}
    )
    print("\n[Return Risk Tool Output]:")
    print(res_risk)
    
    # 3. Test Image Classifier Tool
    res_img = get_deterministic_json_response(
        intent="product_category",
        query="Classify image shoe.png",
        tool_output={"predicted_category": "Sneaker", "confidence": 0.9654}
    )
    print("\n[Image Classifier Tool Output]:")
    print(res_img)
