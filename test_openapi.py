"""
Quick test to verify OpenAPI spec looks good
"""
import json
from app import app

# Get the OpenAPI schema
openapi_schema = app.openapi()

# Print key endpoints with their descriptions
print("=" * 80)
print("OPENAPI ENDPOINTS & DESCRIPTIONS")
print("=" * 80)

for path, methods in openapi_schema["paths"].items():
    for method, details in methods.items():
        if method in ["post", "get", "delete"]:
            print(f"\n{method.upper()} {path}")
            print(f"Summary: {details.get('summary', 'N/A')}")
            desc = details.get('description', 'N/A')
            # Print first 200 chars of description
            print(f"Description: {desc[:200]}...")
            print("-" * 80)

# Save full spec to file
with open("openapi_spec.json", "w") as f:
    json.dump(openapi_schema, f, indent=2)
    print("\n✅ Full OpenAPI spec saved to openapi_spec.json")
