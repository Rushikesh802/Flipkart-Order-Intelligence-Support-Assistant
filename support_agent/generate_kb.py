import json
import os

documents = [
    {
        "doc_id": "doc_1",
        "title": "Apparel and Footwear Returns",
        "text": "Apparel and footwear items can be returned within 14 days of delivery. The items must be unused, unwashed, and have all original tags intact. If the tags are missing, the return request will be automatically rejected."
    },
    {
        "doc_id": "doc_2",
        "title": "Electronics Returns",
        "text": "Electronics such as smartphones and laptops have a 7-day return window from the date of delivery. Returns are only accepted if the product is defective or dead on arrival. Change of mind returns are not applicable for this category."
    },
    {
        "doc_id": "doc_3",
        "title": "Home and Furniture Returns",
        "text": "Home and furniture items can be returned within 10 days of delivery. The product must be unassembled and in its original packaging. A technician will inspect the item before the return is approved."
    },
    {
        "doc_id": "doc_4",
        "title": "COD Refund Timeline",
        "text": "For Cash on Delivery (COD) orders, refunds will be initiated once the returned item reaches our warehouse. The refund will be processed to the provided bank account within 3 to 5 business days. Customers must ensure their bank details are updated in their profile."
    },
    {
        "doc_id": "doc_5",
        "title": "Prepaid Refund Timeline",
        "text": "For prepaid orders, the refund will be credited back to the original payment method. This process usually takes 5 to 7 business days after the return is picked up. Credit card refunds may take an additional billing cycle to reflect on your statement."
    },
    {
        "doc_id": "doc_6",
        "title": "Flipkart Plus Delivery SLA",
        "text": "Flipkart Plus members are eligible for free next-day delivery on all F-Assured items. This service is guaranteed across all major metropolitan cities. If the SLA is breached, members will receive compensation in the form of SuperCoins."
    },
    {
        "doc_id": "doc_7",
        "title": "Standard Delivery SLA",
        "text": "Standard delivery for non-Plus members typically takes 3 to 5 business days depending on the pin code. Shipping charges may apply for orders below 500 INR. Delivery dates are estimates and can be affected by public holidays or unforeseen logistics delays."
    },
    {
        "doc_id": "doc_8",
        "title": "Reverse Pickup Eligibility",
        "text": "Reverse pickup is available for most pin codes across India. If a pin code is unserviceable for reverse pickup, the customer must self-ship the item to our designated warehouse. Shipping costs for self-shipped returns will be reimbursed up to 100 INR."
    },
    {
        "doc_id": "doc_9",
        "title": "Missing Item Policy",
        "text": "If an item is missing from your order, you must raise a complaint within 48 hours of delivery. A thorough investigation will be conducted involving our logistics partner. We may require unboxing videos or images of the outer packaging to process your claim."
    },
    {
        "doc_id": "doc_10",
        "title": "Damaged Product Policy",
        "text": "Damaged products must be reported immediately upon delivery with photographic evidence. The outer box and all packaging materials must be retained for verification. Replacements are subject to stock availability at the time of the claim."
    },
    {
        "doc_id": "doc_11",
        "title": "Exchange Policy",
        "text": "Customers can opt for an exchange instead of a refund if the desired size or color is available. Exchange requests must be placed within the applicable category return window. A single item can only be exchanged once."
    },
    {
        "doc_id": "doc_12",
        "title": "Open Box Delivery Policy",
        "text": "Open Box Delivery is mandatory for high-value electronics and appliances. The delivery agent will open the package in front of you to verify the contents and physical condition. If any damage or missing parts are found, you must reject the delivery immediately."
    }
]

chunks = []
chunk_counter = 1

for doc in documents:
    # Sentence wise chunking (naive approach for this specific text)
    sentences = [s.strip() + "." for s in doc["text"].split(".") if s.strip()]
    for sentence in sentences:
        chunks.append({
            "chunk_id": f"chunk_{chunk_counter}",
            "doc_id": doc["doc_id"],
            "text": sentence
        })
        chunk_counter += 1

queries = [
    {
        "query_id": "q1",
        "query": "Can I return a t-shirt if I took off the tags?",
        "relevant_doc_ids": ["doc_1"]
    },
    {
        "query_id": "q2",
        "query": "How long does it take to get my money back for a COD order?",
        "relevant_doc_ids": ["doc_4"]
    },
    {
        "query_id": "q3",
        "query": "Do I have to pay for shipping if I don't have Flipkart Plus?",
        "relevant_doc_ids": ["doc_7"]
    },
    {
        "query_id": "q4",
        "query": "What should I do if the delivery guy asks me to open my new phone box?",
        "relevant_doc_ids": ["doc_12"]
    },
    {
        "query_id": "q5",
        "query": "My pin code doesn't support reverse pickup, how do I send the item back?",
        "relevant_doc_ids": ["doc_8"]
    }
]

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
output_dir = os.path.join(ROOT_DIR, "data", "knowledge_base")
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "documents.json"), "w") as f:
    json.dump(documents, f, indent=4)

with open(os.path.join(output_dir, "chunks.json"), "w") as f:
    json.dump(chunks, f, indent=4)

with open(os.path.join(output_dir, "queries_eval.json"), "w") as f:
    json.dump(queries, f, indent=4)

print(f"Knowledge base created successfully in {output_dir}")

