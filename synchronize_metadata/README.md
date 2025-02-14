# MTMC batch processing with sequential metadata generation

Guide to sequentially run perception processing on each stream then synchronize those metadata as input for MTMC batch processing. This may be tried as a baseline in case processing all streams simultaneously on a particular HW configuration is not feasible or optimal.

Messages can be logged in 2 formats (either JSON or Protobuf). 
The final output synchronized file for JSON is stored as a .json file & for protobuf is stored .txt file.
Instructions are provided for both.

## Requirements 
- docker 
- nvidia-docker
- python3

## Assumptions
- Default configs from the docker container will be used for generating metadata
- Video file name will be used as `sensorId` in the metadata. This means for a video named `sensor1.mp4`, the below instructions will create a raw data file called `sensor1.log` populating the `sensorId` field with `sensor1`
- For SV3DT, we use the synthetic tracker config file & the messages are logged in protobuf.


# I. Generate metadata per stream (sensor) 

## Step 1: Start Zookeeper and kafka

Perform this step in the 1st terminal. 

```
cd standalone-deployment/modules/multi-camera-tracking/synchronize_metadata
sudo docker-compose up
```

## Step 2: Export variables 

Perform this step in the 2nd terminal.

```
export DOCKER_IMAGE=<insert-mdx-perception-docker-image-and-tag>
export GENERATE_DS_METADATA_FILE=/path/to/generate_ds_metadata.py

# For JSON
export CONSUMER_FILE=/path/to/kafka_consumer.py 

# For Protobuf
export CONSUMER_FILE=/path/to/kafka_pb_consumer.py
export SCHEMA_FILE=/path/to/schema_pb2.py

export INPUT_VIDEOS_DIR=/path/to/video_files_dir/
export OUTPUT_LOGS_DIR=/path/to/output_logs_dir/
```

If you want to run the mdx-perception container with SV3DT, prepare the camera matrix files as mentioned [here](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html#single-view-3d-tracking-alpha). Each camera should have its own camera matrix file as a yaml file, saved as 'sensor_name.yml'. Example camera matrix files are for the Retail Synthetic dataset are present under ```examples/Retail_Synthetic_Dataset``` folder.
These files should be stored in a folder in your host machine. Once saved, use the below export command so the files are mounted. Running SV3DT is optional. 

```
export SV3DT_FILES=/path/to/camera_matrix_folder/

example:
export SV3DT_FILES=/path/to/examples/Retail_Synthetic_Dataset/
```

Note: ```OUTPUT_LOGS_DIR``` should be an empty directory.

## Step 3: Launch mdx-perception docker container 

Perform this step in the 2nd terminal.


```
# For JSON
nvidia-docker run --rm -it -v $GENERATE_DS_METADATA_FILE:/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-fewshot-learning-app/generate_ds_metadata.py -v $CONSUMER_FILE:/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-fewshot-learning-app/kafka_consumer.py  -v $INPUT_VIDEOS_DIR:/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-fewshot-learning-app/samples/ -v $OUTPUT_LOGS_DIR:/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-fewshot-learning-app/output_logs --gpus all --network=host $DOCKER_IMAGE

# For Protobuf
nvidia-docker run --rm -it -v $GENERATE_DS_METADATA_FILE:/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-fewshot-learning-app/generate_ds_metadata.py -v $CONSUMER_FILE:/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-fewshot-learning-app/kafka_consumer.py -v $SCHEMA_FILE:/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-fewshot-learning-app/schema_pb2.py -v $INPUT_VIDEOS_DIR:/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-fewshot-learning-app/samples/ -v $OUTPUT_LOGS_DIR:/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-fewshot-learning-app/output_logs  --gpus all --network=host $DOCKER_IMAGE

# If SV3DT is used, messages are logged in protobuf:
nvidia-docker run --rm -it -v $GENERATE_DS_METADATA_FILE:/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-fewshot-learning-app/generate_ds_metadata.py -v $CONSUMER_FILE:/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-fewshot-learning-app/kafka_consumer.py -v $SCHEMA_FILE:/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-fewshot-learning-app/schema_pb2.py -v $INPUT_VIDEOS_DIR:/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-fewshot-learning-app/samples/ -v $OUTPUT_LOGS_DIR:/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-fewshot-learning-app/output_logs -v $SV3DT_FILES:/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-fewshot-learning-app/configs/mtmc/sv3dt_files --gpus all --network=host $DOCKER_IMAGE

```

Use a directory with your own videos for ```$INPUT_VIDEOS_DIR```. The example above used the 7 sample videos that the Multi-Camera Tracking reference app uses by default.

Once the above step is performed, you should be able to see your videos in the samples directory using the below command

``` ls /opt/nvidia/deepstream/deepstream-6.4/sources/apps/sample_apps/deepstream-fewshot-learning-app/samples
Retail_Synthetic_Cam01.mp4  Retail_Synthetic_Cam03.mp4  Retail_Synthetic_Cam05.mp4  Retail_Synthetic_Cam07.mp4
Retail_Synthetic_Cam02.mp4  Retail_Synthetic_Cam04.mp4  Retail_Synthetic_Cam06.mp4  Retail_Synthetic_Cam08.mp4
```

If you are using SV3DT, you can check if the camera matrix files are mounted correctly
```
ls /opt/nvidia/deepstream/deepstream-6.4/sources/apps/sample_apps/deepstream-fewshot-learning-app/configs/mtmc/sv3dt_files/
Retail_Synthetic_Cam01.yml  Retail_Synthetic_Cam03.yml  Retail_Synthetic_Cam05.yml  Retail_Synthetic_Cam07.yml
Retail_Synthetic_Cam02.yml  Retail_Synthetic_Cam04.yml  Retail_Synthetic_Cam06.yml  Retail_Synthetic_Cam08.yml
```


## Step 4: Run DS + kafka consumer
Perform this step in the 2nd terminal inside the docker container

```
# For json
sudo python3 generate_ds_metadata.py

You will be prompted for nvidia docker password use: nvidiafsl

# For protobuf 
sudo apt-get update

You will be prompted for nvidia docker password use: nvidiafsl

sudo apt-get install python3-pip
sudo pip install protobuf==4.24.4
sudo python3 generate_ds_metadata.py --metadata_format protobuf

# If using SV3DT, use the --use_sv3dt flag
sudo python3 generate_ds_metadata.py --metadata_format protobuf --use_sv3dt
 
```

A single log file called ```all_sensors_unsynced.txt``` will be saved inside the docker container & in the mounted_dir.

## Step 5: Create separate sensor logs from the single log file.
In a new terminal, perform the following step outside the container:

```
# For both json & protobuf
python3 convert_single_log_to_multi_cam_logs.py -i path/to/all_sensors_unsynced.txt  -o /path/to/output_dir
```


# II. Synchronize metadata across streams
Once all the above steps are complete, run the following in a new terminal:

```
pip3 install utils
pip3 install pytz
```

Usage for json - (Metadata format argument is set to json by default): 
```
python3 synchronize_raw_timestamps.py --input_folder input_logs_dir --output_folder output_dir  --fps FPS
```

Usage for protobuf: 
```
python3 synchronize_raw_timestamps.py --input_folder input_logs_dir --output_folder output_dir  --fps FPS --metadata_format protobuf
```

Edit ```input_log_dir, output_dir, fps, metadata_format``` according to your setup.


The output is a JSON file of synchronized raw metadata (equivalent to data sent to `mdx-raw` Kafka topic) that can be used as an input for batch processing. 

If the 7 sample videos were used earlier, the output JSON should be very close if not the same as the ```mtmc_buildingK_playback.json``` file.

# III. Run MTMC batch processing

Follow the last 2 bullets of “Steps for Evaluation” in the documentation section below:
https://developer.nvidia.com/docs/mdx/dev-guide/text/MDX_Multi_Camera_Tracking_KPI.html#application-end-to-end

For JSON, set `jsonDataPath` to a valid JSON file in `resources/app_config.json`.

For protobuf, set `protobufDataPath` to a valid protobuf file in `resources/app_config.json`.

More config options in this page:
https://developer.nvidia.com/docs/mdx/dev-guide/text/MDX_Multi_Camera_Tracking_MS_Configuration.html 


# IV. FAQ

## Question:
The scores of my evaluation results are lower than expected. What could be the cause?

## Solution:  
The problem is likely caused by misalignment of frame IDs in raw data and ground truth. Please adjust the `groundTruthFrameIdOffset` in `app_config.json` accordingly to align the frame IDs to correct the evaluation. To measure the offset of frame IDs, please use the visualization script and set vizMode as frames in `viz_config.json`.

If the bounding boxes are lagging behind the objects in the visualization, your raw data file is running with a fixed offset which is slower than ground truth. To resolve this, set the `groundTruthFrameIdOffset` config in `app_config.json` to a negative value to synchronize the 2 files.

Similarly, if the bounding boxes are ahead of the objects in the visualization, set the `groundTruthFrameIdOffset` config to a positive value to synchronize the 2 files.

The exact value can be obtained by visual inspection.

---

## Question:
The command `sudo docker-compose` up fails with the error message: Bind for 0.0.0.0:9092 failed: port is already allocated


## Solution 1: 
We need to kill the already existing docker processes running kafka and zookeeper. Run `docker ps` to see the existing containers

```
user@machine:~$ docker ps

CONTAINER ID   IMAGE                             COMMAND                  CREATED         STATUS         PORTS                                       NAMES


fa82d5657df0   confluentinc/cp-kafka:5.4.3       "/etc/confluent/dock…"   2 weeks ago     Up 2 days      0.0.0.0:9092->9092/tcp, :::9092->9092/tcp   few-shot-learning_kafka_1


be62a755455c   confluentinc/cp-zookeeper:5.4.3   
"/etc/confluent/dock…"   2 weeks ago     Up 2 days      2181/tcp, 2888/tcp, 3888/tcp                few-shot-learning_zookeeper_1
```

Kill the containers using:
```
user@machine:~$ docker kill <container_id>
user@machine:~$ docker kill fa82d5657df0 
user@machine:~$ docker kill be62a755455c 
```

## Solution 2: 
Kill the already existing process running on port 9092
```
user@machine:~$ sudo netstat -tlnp | grep 9092
tcp        0      0 0.0.0.0:9092            0.0.0.0:*               LISTEN      339218/docker-proxy 
tcp6       0      0 :::9092                 :::*                    LISTEN      339224/docker-proxy 
```

Example:
```
user@machine:~$ sudo kill -9  <enter the process ID here>
user@machine:~$ sudo kill -9  339224
user@machine:~$ sudo kill -9  339218
```
---

## Question: 
I want to run the mdx-perception with my own custom configs. How can I do this?

## Solution:
The `generate_ds_metadata.py` uses simple `sed` commands to read and manipulate the config files. The above python script acts as a wrapper for each `sed` command. Additional `sed` commands can be added to this python file for custom runs.

---

## Question: 
I am observing logs with sensor IDs as empty strings. (For example: sensorID: "" )

## Solution:

You can use the remove_empty_logs.py file to remove such logs.

```
Usage: python3 remove_empty_logs.py --input_file input_file.txt --output_file output_file.txt
```
---