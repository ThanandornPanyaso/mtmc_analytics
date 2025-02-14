import json

# Initialize a set to store unique IDs
unique_ids = set()

# Read the file line by line
file_path = "messages-cross1.json"  # Replace with your actual file path
with open(file_path, "r") as file:
    for line in file:
        try:
            # Parse each line as a JSON object
            data = json.loads(line.strip())
            # Check if "objects" key exists in the JSON object
            if "objects" in data:
                for obj in data["objects"]:
                    # Split by '|' and take the first part
                    if any("Crossed" in obj):
                        object_id = obj.split('|')[0]
                        unique_ids.add(object_id)
        except json.JSONDecodeError:
            # Skip lines that are not valid JSON
            continue

# Output the unique IDs
print("Unique Object IDs:", sorted(unique_ids))  # Sorted for readability

# Optional: Save the IDs to a file if needed
with open("unique_ids.txt", "w") as output_file:
    output_file.write("\n".join(sorted(unique_ids)))
