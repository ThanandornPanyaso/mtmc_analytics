import json
import numpy as np
from scipy.io import savemat
from typing import List, Dict, Tuple, Optional, Any
def normalize_vector(vector: List[float]) -> Optional[np.array]:
    """
    Normalizes a vector

    :param List[float] vector: vector
    :return: normalized vector or None
    :rtype: Optional[List[float]]
    """
    if len(vector) > 0:
        vector_norm = np.linalg.norm(vector)
        if vector_norm > 0.:
            return vector / vector_norm
        else:
            print("WARNING: The norm of the input vector for normalization is zero. None is returned.")
    return None

# Initialize lists to store data
query_f = []
person_ids = []
frame_ids = []
camera_ids = []
# Open the file and process line by line
with open("messages-cross-renew.json", "r") as file:
    for line in file:
        # Parse the line as JSON
        data = json.loads(line.strip())

        # Iterate over objects
        for obj_str in data.get("objects", []):
            if not "Crossed" in obj_str :
                continue
            object_tokens = obj_str.split("|")
            
            # # Only consider "Person" objects
            # object_type = object_tokens[5]
            # if object_type != "Person":
            #     continue
            frame_id = int(data.get("id", ""))
            frame_ids.append(frame_id)
            camera_id = data.get("sensorId", "")
            camera_ids.append(camera_id)
            # Extract the person ID (first number in the object string)
            person_id = int(object_tokens[0])
            person_ids.append(person_id)

            # Extract the embedding
            embedding = None
            try:
                embedding_idx = object_tokens.index("embedding")
                embedding = object_tokens[embedding_idx + 1]
                embedding = [float(x) for x in embedding.split(",")]
                embedding = normalize_vector(embedding)  # Use your normalize_vector function
                if embedding is not None:
                    query_f.append(embedding)
                else:
                    print(f"WARNING: Normalized embedding is None for object: {obj_str}")
            except ValueError:
                print(f"WARNING: Embedding not found or invalid for object: {obj_str}")
                continue


# Convert lists to NumPy arrays for MATLAB compatibility
query_f = np.array(query_f)
gallery_f = np.array(query_f)  # Duplicate of query_f
person_ids = np.array(person_ids)
frame_ids = np.array(frame_ids)
camera_ids = np.array(camera_ids)
# Save to a .mat file
savemat("features_deep_renew.mat", {
    "query_f": query_f,
    "gallery_f": gallery_f,
    "person_Id": person_ids,
    "frame_ids": frame_ids,
    "camera_ids": camera_ids
})

print("Created features.mat successfully!")
