# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 22:12:32 2021

@author: Hari
"""

# from time import sleep
from json import dumps
from kafka import KafkaProducer
from kafka import KafkaAdminClient
from kafka.admin import NewPartitions, NewTopic
import TAPPconfig as cfg
import traceback
from logging_module import setup_logging
import pytz
import os
from preProcUtilities import create_log_folder
from datetime import datetime

topicName = cfg.getKafkaTopic()
noPartitions = cfg.getKafkaTopicPartitions()
kafkaServer = cfg.getKafkaServer()

try:
    admin_client = KafkaAdminClient(bootstrap_servers=[kafkaServer])
    topic_list = []
    print("Create new topic: ", topicName)
    topic_list.append(NewTopic(name=topicName,
                               num_partitions = 1,
                               replication_factor=1))
    admin_client.create_topics(new_topics=topic_list,
                               validate_only=False)
    print("Create new topic created: ", topicName)
except:
    print("New topic not created:", traceback.print_exc())
    pass

try:
    print("Create new partition: ", noPartitions)
    topic_partitions = {}
    topic_partitions[topicName] = NewPartitions(total_count=noPartitions)
    admin_client.create_partitions(topic_partitions)
except:
    print(traceback.print_exc())
    pass

# producer = KafkaProducer(bootstrap_servers=[kafkaServer],
#                          value_serializer=lambda x: 
#                          dumps(x).encode('utf-8'))

# partitions = [0,1,2,3]
partitionNumber = 0

# Generate a random number between 1 and 1000 (inclusive)

def sendMessage(data):

    try:
        global partitionNumber
        
        documentId = data["documentId"] 
        auth_token = data["auth_token"]
        noOfPages = data["noOfPages"]
        # file_name = str(documentId)+"_auth_token_"+str(auth_token)+".log"
        # log_folder = os.path.join(current_directory, 'logs')
        # putil.create_log_folder(log_folder)
        # document_log_file = os.path.join(log_folder,file_name)
        # from logging_module import setup_logging
        current_directory = os.getcwd()
        file_name = str(documentId)+"_auth_token_"+str(auth_token)+".log"
        log_folder = os.path.join(current_directory, 'logs')
        create_log_folder(log_folder)
        document_log_file = os.path.join(log_folder,file_name)
        document_log = setup_logging(log_file=document_log_file,logger_name="document_log")
        
        # file_name = str(documentId)+"_kafka.log"
        # log_folder = os.path.join(current_directory, 'logs')
        # putil.create_log_folder(log_folder)
        # log_file_kafka = os.path.join(log_folder,file_name)
        # app_log = setup_logging(log_file_kafka)
        local_timezone = pytz.timezone('Asia/Kolkata')
        dt_utc = datetime.utcnow()
        dt_local = dt_utc.replace(tzinfo=pytz.utc).astimezone(local_timezone)
        document_log.info(f"In kafka_producer Sending message to kafkaProducer for document id: '{documentId}' at time {dt_local}")
        
        producer = KafkaProducer(bootstrap_servers=[kafkaServer],
                                 value_serializer=lambda x: 
                                 dumps(x).encode('utf-8'))
        import random
        random_partitionNumber = random.randint(0,1000)

        # print(random_number)
        p = random_partitionNumber % noPartitions
        # from check_process import check_partition
        # document_log.info("Checking partition restart functionality")
        # status_partition = check_partition(documentId,auth_token,p)
        # document_log.info(f"Status of restart functionality is: {status_partition}")
        # print("Checking partition restart",status_partition)
        # print("Producer send: ", data, p,topicName)
        document_log.info(f"In kafka_producer, Total Partitions are: {noPartitions} Partition is: '{p}' topic name is: '{topicName}' ")
        dt_utc = datetime.utcnow()
        dt_local = dt_utc.replace(tzinfo=pytz.utc).astimezone(local_timezone)
        document_log.info(f"In kafka_producer, producer sending message for document_id '{documentId} time is {dt_local}")
        producer.send(topicName,
                      value=data,
                      partition = p)
        # from kafka_python import send_message
        # response = send_message(message= data, documentId = documentId, auth_token = auth_token, TOPIC = topicName, partition= p)
        dt_utc = datetime.utcnow()
        dt_local = dt_utc.replace(tzinfo=pytz.utc).astimezone(local_timezone)
        document_log.info(f"In kafka_producer, message sent for document_id '{documentId} time is {dt_local}")
        partitionNumber += 1
        return True
    except Exception as e:
        document_log.error(f"In kafka_producer got Exception {e}")
        print(traceback.print_exc())
        return False


# for e in range(20):
#     data = {"number":e}
#     p = e % len(partitions)
#     producer.send('numtest',
#                   value=data,
#                   partition = p
#                   )
    # producer.send('numtest',
    #               value=data
    #               )
    # sleep(1)
