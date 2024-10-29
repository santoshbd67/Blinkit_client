# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 08:24:44 2021

@author: Hari
"""

import subprocess
import sys
import TAPPconfig as cfg

partitions = cfg.getKafkaTopicPartitions()

# from datetime import datetime as dt
# from kafka_sql import createTable
# import sqlite3 as sql
# db_name = "kafka.db"
# tbl_name = "kafka_client_messages"

# commit_sql = "update " + tbl_name
# commit_sql += " set committed = 2,"
# commit_sql += " committed_time = '" + str(dt.now()) + "'"
# commit_sql += " where committed = 0"

# table_created = createTable()
# if table_created:
#     con = sql.connect(db_name)
#     cur = con.cursor()
#     cur.execute(commit_sql)
#     con.commit()
#     con.close()
for i in range(partitions):
    
    read_proc = subprocess.Popen(['python',
                                  'kafka_consumer_readall.py',
                                  str(i)])

sys.exit(0)

