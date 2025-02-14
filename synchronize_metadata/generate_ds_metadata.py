# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import os
import signal
import time
import argparse
import subprocess


def edit_static_configs(metadata_format, use_sv3dt):
    """ This will edit the configs in mtmc_config.txt """

    # Edit the msgconv type from protobuf to kafka
    if metadata_format == "json":
        original_message_payload_txt = "msg-conv-payload-type=2"
        modified_message_payload_txt = "msg-conv-payload-type=1"
        command = f"sed -i 's|{original_message_payload_txt}|{modified_message_payload_txt}|' configs/mtmc/mtmc_config.txt"
        process = subprocess.Popen(command, shell=True)
        process.wait()        

    # Edit the EOS value from 1 to 0
    original_eos_txt = "drop-pipeline-eos=1"
    modified_eos_txt = "drop-pipeline-eos=0"
    command = f"sed -i 's|{original_eos_txt}|{modified_eos_txt}|' configs/mtmc/mtmc_config.txt"
    process = subprocess.Popen(command, shell=True)
    process.wait()

    # Turn off osd in main config
    command = f"sed -i '/\[osd\]/,/^$/ s/enable=1/enable=0/' configs/mtmc/mtmc_config.txt"
    process = subprocess.Popen(command, shell=True)
    process.wait()

    # Turn sync=1 in sink0
    original_sink0_txt = "sync=0"
    modified_sink0_txt = "sync=1"
    command = f"sed -i 's|{original_sink0_txt}|{modified_sink0_txt}|' configs/mtmc/mtmc_config.txt"
    process = subprocess.Popen(command, shell=True)
    process.wait()

    # Set interval = 0 for pgie
    command = f"sed -i '/\[primary-gie\]/,/^\s*$/s/interval=1/interval=0/' configs/mtmc/mtmc_config.txt"
    process = subprocess.Popen(command, shell=True)
    process.wait()


    # Change tracker file if SV3DT is enabled
    if use_sv3dt:

        # Set tracker height
        original_tracker_height_txt = "tracker-height=544"
        modified_tracker_height_txt = "tracker-height=1088"
        command = f"sed -i 's|{original_tracker_height_txt}|{modified_tracker_height_txt}|' configs/mtmc/mtmc_config.txt"
        process = subprocess.Popen(command, shell=True)
        process.wait()

        # Set tracker width
        original_tracker_width_txt = "tracker-width=960"
        modified_tracker_width_txt = "tracker-width=1920"
        command = f"sed -i 's|{original_tracker_width_txt}|{modified_tracker_width_txt}|' configs/mtmc/mtmc_config.txt"
        process = subprocess.Popen(command, shell=True)
        process.wait()

        # Point to the synthetic tracker file if SV3DT is enabled
        original_tracker_file_txt = "config_tracker_NvDCF_accuracy.yml"
        modified_tracker_file_txt = "config_synthetic_tracker_NvDCF_accuracy_3D.yml"
        command = f"sed -i 's|{original_tracker_file_txt}|{modified_tracker_file_txt}|' configs/mtmc/mtmc_config.txt"
        process = subprocess.Popen(command, shell=True)
        process.wait()


def generate_ds_metadata(metadata_format, use_sv3dt):

    # Set the correct tracker file
    tracker_config_file = "config_tracker_NvDCF_accuracy.yml"
    if use_sv3dt:
        tracker_config_file = "config_synthetic_tracker_NvDCF_accuracy_3D.yml"

    # Edit the static configs in mtmc_config.txt
    edit_static_configs(metadata_format, use_sv3dt)

    # Define the input & output folders
    VIDEO_FILES_DIR="/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-fewshot-learning-app/samples/"

    # Make output dir for storing individual logs
    OUTPUT_LOG_DIR = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-fewshot-learning-app/output_logs/"

    # Default video and sensor ids names in configs. These will be searched and replaced.
    last_video_file_name = "Building_K_Cam7.mp4"
    last_sensor_name = "stream1"
    last_sensor_name_in_main_config = "UniqueSensorName1"
    last_sensor_id_in_main_config = "UniqueSensorId1"
    last_cam_matrix_in_sv3dt = "cam01_matrix_config"

    # Run consumer.py 
    absoulte_sensor_logs_file_name = os.path.join(OUTPUT_LOG_DIR, "all_sensors_unsynced.txt")
    command = f"sudo python3 kafka_consumer.py {absoulte_sensor_logs_file_name}"
    kafka_process = subprocess.Popen("exec "+ command, shell=True, preexec_fn=os.setsid)

    print("Running video...")
    video_count = 1
    for video_file_name in sorted(os.listdir(VIDEO_FILES_DIR)):

        print("Processing video file: ", video_file_name)

        # Get the sensor name based on video file
        new_sensor_name = video_file_name.split(".")[0]

        # Edit the video file name in main config file
        command = f"sed -i 's|{last_video_file_name}|{video_file_name}|' configs/mtmc/mtmc_config.txt"
        process = subprocess.Popen(command, shell=True)
        process.wait()

        # Edit the sensor name in main config file
        command = f"sed -i 's|sensor-id-list={last_sensor_id_in_main_config}|sensor-id-list={new_sensor_name}|' configs/mtmc/mtmc_config.txt"
        process = subprocess.Popen(command, shell=True)
        process.wait()

        command = f"sed -i 's|sensor-name-list={last_sensor_name_in_main_config}|sensor-name-list={new_sensor_name}|' configs/mtmc/mtmc_config.txt"
        process = subprocess.Popen(command, shell=True)
        process.wait()


        # Edit the sensor name in msgconv config file
        command = f"sed -i 's|{last_sensor_name}|{new_sensor_name}|' configs/mtmc/dstest5_msgconv_sample_config.txt"
        process = subprocess.Popen(command, shell=True)
        process.wait()

        # Edit camera matrix if sv3dt is enabled
        if use_sv3dt:
            command = f"sed -i 's|{last_cam_matrix_in_sv3dt}.yml|{new_sensor_name}.yml|' configs/mtmc/{tracker_config_file}"
            process = subprocess.Popen(command, shell=True)
            process.wait()

            if video_count == 1:
                command = f"sed -i 's|{new_sensor_name}.yml|sv3dt_files/{new_sensor_name}.yml|' configs/mtmc/{tracker_config_file}"
                process = subprocess.Popen(command, shell=True)
                process.wait()

            # Print the sv3dt config file
            command = f"cat configs/mtmc/{tracker_config_file}"
            process = subprocess.Popen(command, shell=True)
            process.wait()

        command = f"cat configs/mtmc/mtmc_config.txt"
        process = subprocess.Popen(command, shell=True)
        process.wait()

        # Run Deepstream
        command = "./deepstream-fewshot-learning-app -c configs/mtmc/mtmc_config.txt -m 1 -t 1 -l 5 --message-rate 1 --tracker-reid 1 --reid-store-age 1"
        ds_process = subprocess.Popen("exec " +command, shell=True)
        ds_process.wait()

        last_video_file_name = video_file_name
        last_sensor_name = new_sensor_name
        last_sensor_id_in_main_config = new_sensor_name
        last_sensor_name_in_main_config = new_sensor_name
        last_cam_matrix_in_sv3dt = new_sensor_name
        video_count += 1

        print("Processed video file: ", video_file_name)

    kafka_process.kill()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_format', default="json", choices=["json", "protobuf"])
    parser.add_argument('--use_sv3dt', action="store_true")

    # Parse all arguments
    args = parser.parse_args()
    generate_ds_metadata(args.metadata_format, args.use_sv3dt)