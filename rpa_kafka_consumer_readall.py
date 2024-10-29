from kafka import KafkaConsumer
from kafka import TopicPartition
from json import loads
# from time import sleep
from sys import argv
import TAPPconfig as cfg

partition = int(argv[1])
cnt = 0
serverName = cfg.getKafkaServer()
consGroup = cfg.getKafkaConsumerGroup()
topic = cfg.getKafkaTopic()

def get_waiting_messages_count(consumer, partitions):
    waiting_messages_count = 0
    # Get the current offset
    for partition in partitions:
        current_offset = consumer.position(partition)

        # Get the end offset for the partition
        end_offset = consumer.end_offsets([partition])[partition]

        # Calculate the number of waiting messages
        waiting_messages_count += max(end_offset - current_offset, 0)

    return waiting_messages_count

#Commit all the open messages when starting the kafka_consumer
print("Starting called again")
consumer = KafkaConsumer(
     bootstrap_servers=[serverName],
     auto_offset_reset='earliest',
     enable_auto_commit=True,
     group_id=consGroup,
     value_deserializer=lambda x: loads(x.decode('utf-8')))
print("Partition assigned", partition,topic)
consumer.assign([TopicPartition(topic, partition)])

# Get the assigned partitions
partitions = consumer.assignment()

waiting_messages_count = get_waiting_messages_count(consumer, partitions)
print("Waiting message count is:", waiting_messages_count, partition)
if waiting_messages_count == 0:
    print("No messages to consume. Exiting loop.") 
else:    
    for message in consumer:
        message_ = message.value
        print(message)
        cnt+=1
        if cnt == waiting_messages_count:
            break
    print("outside for loop", partition)
consumer.close()
print("exiting readall")
