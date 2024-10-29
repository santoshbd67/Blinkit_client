import subprocess
import os
import TAPPconfig as cfg

# Path to the text file containing PID and partition information
# Get File Path from config file
# file_path = "/home/pAIges_client/src/kafka_info.txt"
client_src = cfg.getpAIgesClientSRC()
kafka_restart_file_name = cfg.get_kafka_restart_file_name()
file_path = os.path.join(client_src,kafka_restart_file_name)
# Path to the virtual environment's activate script
# Get activate script from config file
# activate_script = "/root/.cache/pypoetry/virtualenvs/poetry-test-QuTT6H4w-py3.10/bin/activate"
activate_script = cfg.get_kafka_restart_activate_script_path()

# Path to the directory containing your kafka_consumer.py script
# script_directory = "/home/pAIges_home/pAIges_client/pierian/src"


# Read the PID and partition information from the file
with open(file_path, "r") as file:
    lines = file.readlines()

# Extract PID and partition information
consumer_info = [line.strip().split() for line in lines]

# Check and restart missing consumers
for pid, partition in consumer_info:
    pid = int(pid)
    try:
        # Check if the process is running based on PID
        subprocess.check_output(["kill", "-0", str(pid)])
    except subprocess.CalledProcessError:
        print(f"Restarting Kafka consumer for partition {partition}...")
#        subprocess.Popen(["python", "kafka_consumer.py", partition])

        # Activate the virtual environment and execute the script with the partition number as an argument
        command = [
            "/bin/bash", "-c",
            f"source {activate_script} && python kafka_consumer.py {partition}"
        ]
        subprocess.Popen(command)

# Define the shell command you want to run
command = 'ps -ef | grep "python kafka_consumer.py" | grep -v "grep" | awk \'{print $2, $NF}\'  > ' + str(file_path)   
try:
    # Run the shell command
    subprocess.run(command, shell=True, check=True, executable="/bin/bash")
    print("Command executed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Command failed with error: {e}")

print("Consumer monitoring and restart completed.")