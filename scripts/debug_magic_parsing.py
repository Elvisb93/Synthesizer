import json
import logging

# Mockting the cleaning logic from core/llm_client.py
def parse_response(raw_response):
    print(f"Raw length: {len(raw_response)}")
    clean_json = raw_response.strip()
    
    start_idx = clean_json.find('[')
    end_idx = clean_json.rfind(']')
    
    print(f"Start: {start_idx}, End: {end_idx}")
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        clean_json = clean_json[start_idx:end_idx+1]
        print("Extracted substring via indices.")
    else:
        # Fallback cleanups
        if clean_json.startswith("```json"):
            clean_json = clean_json[7:]
        if clean_json.startswith("```"):
            clean_json = clean_json[3:]
        if clean_json.endswith("```"):
            clean_json = clean_json[:-3]
            
    clean_json = clean_json.strip()
    
    try:
        schema = json.loads(clean_json)
        print("JSON Parsed Successfully!")
        print(f"Item count: {len(schema)}")
        return schema
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print(f"Snippet near error: {clean_json[max(0, e.pos-20):min(len(clean_json), e.pos+20)]}")
        return []

# The content string provided by the user
user_content = """[
  {
    "name": "id",
    "type": "Auto Increment (ID)",
    "prompt_instruction": "",
    "constraints": {}
  },
  {
    "name": "first_name",
    "type": "Faker / Deterministic",
    "prompt_instruction": "",
    "constraints": {"faker_provider": "name.first_name"}
  },
  {
    "name": "last_name",
    "type": "Faker / Deterministic",
    "prompt_instruction": "",
    "constraints": {"faker_provider": "name.last_name"}
  },
  {
    "name": "email",
    "type": "Short Text",
    "prompt_instruction": "@[first_name].@[last_name]@amazon.com",
    "constraints": {}
  },
  {
    "name": "phone_number",
    "type": "Short Text",
    "prompt_instruction": "(###) ###-####",
    "constraints": {}
  },
  {
    "name": "street_address",
    "type": "Faker / Deterministic",
    "prompt_instruction": "",
    "constraints": {"faker_provider": "address.street_address"}
  },
  {
    "name": "city",
    "type": "Short Text",
    "prompt_instruction": "City in @[country]",
    "constraints": {}
  },
  {
    "name": "state_province",
    "type": "Short Text",
    "prompt_instruction": "State/Province for @[city] in @[country]",
    "constraints": {}
  },
  {
    "name": "zip_code",
    "type": "Short Text",
    "prompt_instruction": "ZIP code for @[state_province] in @[country]",
    "constraints": {}
  },
  {
    "name": "country",
    "type": "Categorical",
    "prompt_instruction": "",
    "constraints": {"options": ["United States", "Canada", "Mexico"]}
  },
  {
    "name": "date_of_birth",
    "type": "Short Text",
    "prompt_instruction": "Date of birth in YYYY-MM-DD",
    "constraints": {}
  },
  {
    "name": "gender",
    "type": "Categorical",
    "prompt_instruction": "",
    "constraints": {"options": ["Male", "Female", "Non-binary"]}
  },
  {
    "name": "loyalty_program_member",
    "type": "Boolean",
    "prompt_instruction": "",
    "constraints": {}
  },
  {
    "name": "membership_status",
    "type": "Categorical",
    "prompt_instruction": "",
    "constraints": {"options": ["None", "Silver", "Gold", "Platinum"]}
  },
  {
    "name": "join_date",
    "type": "Short Text",
    "prompt_instruction": "Date when customer joined Amazon in YYYY-MM-DD",
    "constraints": {}
  },
  {
    "name": "last_purchase_date",
    "type": "Short Text",
    "prompt_instruction": "Last purchase date after @[join_date]",
    "constraints": {}
  },
  {
    "name": "total_spent",
    "type": "Numeric",
    "prompt_instruction": "",
    "constraints": {"min_value": 0, "max_value": 10000}
  },
  {
    "name": "preferred_payment_method",
    "type": "Categorical",
    "prompt_instruction": "",
    "constraints": {"options": ["Credit Card", "Debit Card", "Amazon Pay", "PayPal"]}
  },
  {
    "name": "shipping_preference",
    "type": "Categorical",
    "prompt_instruction": "",
    "constraints": {"options": ["Standard Shipping", "Expedited Shipping", "Pickup at Store"]}
  }
]"""

if __name__ == "__main__":
    parse_response(user_content)
