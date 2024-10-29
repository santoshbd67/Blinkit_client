#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 11:15:00 2023

@author: Parmesh
"""
from sys import argv
import traceback
from json import loads

from kafka import KafkaConsumer
from kafka import TopicPartition

# from time import sleep
import TAPPconfig as cfg
# 31 Oct 2023 Removed import of full tapp_client
# import tapp_client as cli
from tapp_client import getExtractionResults
from logging_module import setup_logging
import time
# import logging
import pytz
from datetime import datetime
local_timezone = pytz.timezone('Asia/Kolkata')
import os
from preProcUtilities import create_log_folder
# from kafka_sql import messageRead, messageCommited

serverName = cfg.getKafkaServer()
consGroup = cfg.getKafkaConsumerGroup()
topic = cfg.getKafkaTopic()
rootFolderPath = cfg.getRootFolderPath()


partition = int(argv[1])

while True:
    consumer = KafkaConsumer(
         bootstrap_servers = [serverName],
         auto_offset_reset = 'earliest',
         enable_auto_commit = True,
         group_id = consGroup,
         value_deserializer=lambda x: loads(x.decode('utf-8')))
    print("Partition assigned", partition,topic)
    consumer.assign([TopicPartition(topic, partition)])
    print("Assigned comsumer")
    for message in consumer:
        message_ = message.value
        sub_id = message_["sub_id"]
        auth_token = message_["auth_token"]
        documentId = message_["documentId"]
        s_delta = message_["s_delta"]
        callbackUrl = message_["callbackUrl"]
        print("Message received", message)
        current_directory = os.getcwd()
        # file_name = str(documentId)+"_kafka.log"
        file_name = str(documentId)+"_auth_token_"+str(auth_token)+".log"
        log_folder = os.path.join(current_directory, 'logs')
        create_log_folder(log_folder)
        document_log_file = os.path.join(log_folder,file_name)
        """logging.basicConfig(
            filename= log_file,      # Specify the log file name (optional)
            level=logging.DEBUG,     # Set the minimum log level to record (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            format='%(asctime)s - %(levelname)s - %(message)s',  # Define the log format
            datefmt='%Y-%m-%d %H:%M:%S'  # Optionally specify the date format
        )
        # Set up a specific logger for each message with a unique log file name
        logger = logging.getLogger("sa")
        """
        # document_log = setup_logging(logger_name="document_log")
        document_log = setup_logging(log_file=document_log_file,logger_name="document_log")
        # messageRead(topic=topic, documentId=documentId)
        # document_log.info(f"In kafka_consumer message read for document_id {documentId} and set read status to 1 in sqlite")
        # app_log = setup_logging(log_file_kafka)
        try:
            t1 = time.time()
            dt_utc = datetime.utcnow()
            dt_local = dt_utc.replace(tzinfo=pytz.utc).astimezone(local_timezone)
            document_log.info(f"In kafka_consumer message received for document_id {documentId} at time {dt_local}")
            document_log.info(f"In Kafka_consumer sending message to tapp_client for document_id {documentId} at time {dt_local}")
            document_log.info(f"Kafka_consumer value for Partition is {partition} and topic name is {topic}")
            extraction = getExtractionResults(auth_token,
                                                  s_delta,
                                                  documentId,
                                                  callbackUrl,
                                                  sub_id)
            # extraction = cli.getExtractionResults(auth_token,
            #                                       s_delta,
            #                                       documentId,
            #                                       callbackUrl,
            #                                       sub_id)
            # messageCommited(topic=topic, documentId=documentId)
            # document_log.info(f"In kafka_consumer message commited set to 1 in sqlite")
            dt_utc = datetime.utcnow()
            dt_local = dt_utc.replace(tzinfo=pytz.utc).astimezone(local_timezone)
            document_log.info(f"In kafka consumer getExtractionResults returned with response from tapp_client as {extraction} at time {dt_local} ")
            time_taken = time.time() - t1
            print(f"Time taken for {documentId} in extraction is {time_taken} and Extraction status {extraction}")

        except Exception as e:
            document_log.error("In kafka_consumer got exception {e}")
            print("Kafka consumer",
                  traceback.print_exc())
            document_log.error(f"Exception occurred: {str(traceback.format_exc())}")
        finally:
            document_log.debug(f"Closing the loger for docuement : {documentId}")
            # app_log.close_file_handler(log_file_kafka)
        
    consumer.close()
#     consumer.assign([TopicPartition(topic, partition)])
#     logger.info("Partition assigned: %d, Topic: %s", partition, topic)


#     consumer.close()