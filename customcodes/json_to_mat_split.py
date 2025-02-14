import json
import numpy as np
from scipy.io import savemat
from typing import List, Dict, Tuple, Optional, Any
import os
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

# Parameters
cam0_name = "stream1"
cam1_name = "stream2"
time_index = "800"
input_filename = f"messages-cross-renew{time_index}.json"
input_folder= f"/home/big/Github/dataset_prepare/get_img_cosine_renew{time_index}"
output_filename = f"features_deep_renew_split{time_index}.mat"
output_folder = "/home/big/Github/Person_reID_baseline_pytorch/"
# Initialize lists to store data
query_f = []
query_ids = []
query_frames = []
gallery_f = []
gallery_ids = []
gallery_frames = []
#query_cams = []
# Open the file and process line by line
with open(os.path.join(input_folder,input_filename), "r") as file:
    for line in file:
        # Parse the line as JSON
        data = json.loads(line.strip())

        # Iterate over objects
        for obj_str in data.get("objects", []):
            if not "Crossed" in obj_str :
                continue
            object_tokens = obj_str.split("|")
            
            frame_id = int(data.get("id", ""))
            
            camera_id = data.get("sensorId", "")
            #query_cams.append(camera_id)

            # Extract the person ID (first number in the object string)
            person_id = int(object_tokens[0])
            

            # Extract the embedding
            embedding = None
            try:
                embedding_idx = object_tokens.index("embedding")
                embedding = object_tokens[embedding_idx + 1]
                embedding = [float(x) for x in embedding.split(",")]
                embedding = normalize_vector(embedding)  # Use your normalize_vector function
                if embedding is not None:
                    if cam0_name in camera_id :
                        gallery_f.append(embedding)
                        gallery_ids.append(person_id)
                        gallery_frames.append(frame_id)
                    elif cam1_name in camera_id:
                        query_f.append(embedding)
                        query_ids.append(person_id)
                        query_frames.append(frame_id)
                    else:
                        print(f"WARNING: No match camera names for object: {obj_str}")
                else:
                    print(f"WARNING: Normalized embedding is None for object: {obj_str}")
            except ValueError:
                print(f"WARNING: Embedding not found or invalid for object: {obj_str}")
                continue


# Convert lists to NumPy arrays for MATLAB compatibility
query_f = np.array(query_f)
query_ids = np.array(query_ids)
query_frames = np.array(query_frames)
gallery_f = np.array(gallery_f)  
gallery_ids = np.array(gallery_ids)
gallery_frames = np.array(gallery_frames)
# Save to a .mat file
savemat(os.path.join(output_folder,output_filename), {
    "query_f": query_f,
    "query_ids": query_ids,
    "query_frames": query_frames,
    "gallery_f": gallery_f,
    "gallery_ids": gallery_ids,
    "gallery_frames": gallery_frames,
})

print(f"Created {output_filename} successfully!")
