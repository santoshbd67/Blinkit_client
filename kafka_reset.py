import os
import subprocess
import json

# Read the .env file
with open(".env", "r") as env_file:
    env_data = json.load(env_file)

kafka_bootstrap_servers = env_data.get("KAFKA_SERVER")
kafka_consumer_group = env_data.get("KAFKA_CONSUMER_GROUP")
num_partitions = env_data.get("KAFKA_TOPIC_PARTITIONS")

# Check for missing partitions and restart them
for partition_number in range(num_partitions):
    # Check if the Kafka consumer for the partition is running
    command = f"pgrep -f 'python kafka_consumer.py {partition_number}( |$)'"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    
    if result.returncode != 0:
        # Restart the Kafka consumer for the missing partition
        print(f"Restarting Kafka consumer for partition {partition_number}")
        subprocess.Popen(["python", "kafka_consumer.py", str(partition_number)])
        #subprocess.Popen(["python", "kafka_consumer.py", str(partition_number), kafka_bootstrap_servers, kafka_consumer_group])

print("Partition check and recovery completed.")
