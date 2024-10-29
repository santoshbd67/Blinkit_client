# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 22:50:03 2023

@author: DELL
"""

import psutil
import subprocess
import os
import preProcUtilities as putil
from logging_module import setup_logging
import pytz
from datetime import datetime

def check_partition(documentId, auth_token,partition_to_check  = None):
    # documentId = data["documentId"] 
        
    # from logging_module import setup_logging
    # partition_to_check = 6
    if partition_to_check != None:
        # document_log = setup_logging(logger_name="document_log")
        current_directory = os.getcwd()
        # file_name = str(documentId)+"_kafka.log"
        file_name = str(documentId)+"_auth_token_"+str(auth_token)+".log"
        log_folder = os.path.join(current_directory, 'logs')
        putil.create_log_folder(log_folder)
        document_log_file = os.path.join(log_folder,file_name)
        document_log = setup_logging(log_file=document_log_file,logger_name="document_log")
        # app_log = setup_logging(log_file_kafka)
        local_timezone = pytz.timezone('Asia/Kolkata')
        dt_utc = datetime.utcnow()
        dt_local = dt_utc.replace(tzinfo=pytz.utc).astimezone(local_timezone)
        document_log.info(f"In check_process Request received for document id: '{documentId}' at time {dt_local}")
        found_process = False
        for proc in psutil.process_iter(['name', 'cmdline']):
            if 'python' in proc.info['name'] and 'kafka_consumer.py' in proc.cmdline() and str(partition_to_check) in proc.cmdline():
                document_log.info(f"Process for partition {partition_to_check} is already running. Skipping new subprocess.")
                print(f"Process for partition {partition_to_check} is already running. Skipping new subprocess.")
                found_process = True
                break

        if found_process:
            return 0
        else:
            document_log.info(f"In check_process No process found for partition {partition_to_check}. You can start a new subprocess here.")
            subprocess.Popen(['python', 'kafka_consumer.py', str(partition_to_check)])
            document_log.info(f"Process for partition {partition_to_check} started")
            return 1





