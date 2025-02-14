import json

# Step 1: Extract unique IDs (use the output of the previous program or load from a file)
unique_ids_file = "unique_ids.txt"  # File containing extracted unique IDs
with open(unique_ids_file, "r") as f:
    unique_ids = set(f.read().splitlines())  # Load unique IDs into a set

# Step 2: Filter JSON objects from another file
input_json_file = "messages-all1.json"  # Replace with the file to filter from
filtered_objects = []

with open(input_json_file, "r") as file:
    for line in file:
        try:
            # Parse each line as a JSON object
            data = json.loads(line.strip())
            # Check if "objects" key exists and matches any unique ID
            if "objects" in data:
                for obj in data["objects"]:
                    object_id = obj.split('|')[0]
                    if object_id in unique_ids:
                        filtered_objects.append(data)
                        break  # Stop checking other objects in the current JSON object
        except json.JSONDecodeError:
            # Skip lines that are not valid JSON
            continue

# Step 3: Save filtered objects to a new file
output_json_file = "filtered_data.json"
with open(output_json_file, "w") as output_file:
    for obj in filtered_objects:
        output_file.write(json.dumps(obj) + "\n")

print(f"Filtered {len(filtered_objects)} objects. Saved to '{output_json_file}'.")
