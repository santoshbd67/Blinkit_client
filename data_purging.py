import psycopg2
import logging
import os
import requests
import json
import datetime
# import time
import TAPPconfig as cfg
import time
import preProcUtilities as putil
import traceback
from dateutil import parser
import pandas as pd
from cryptography.fernet import Fernet
import sys

UI_SERVER = cfg.getUIServer()
DB_NAME = cfg.getDB_NAME()
# S3_UPLOAD_TABLE = cfg.getS3_UPLOAD_TABLE()
PURGE_TABLE = cfg.getPURGE_TABLE()
# statuses_to_exclude = ['REVIEW_COMPLETED', 'REVIEW', 'REASSIGN']
# PURGE_STATUS = ', '.join([f"'{status}'" for status in statuses_to_exclude])
KEY = cfg.getKEY()
# PURGE_STATUS = cfg.getPURGEStatus()
PURGE_FREQUENCY = cfg.getPURGEFrequency()
PURGE_FREQUENCY_FILESHARE = cfg.getPURGEFrequencyFileShare()
PURGE_FREQUENCY_DB = cfg.getPURGEFrequencyDb()
ENCRYPTED_CONNSTR = cfg.getEncryptedConnStr()
IMPORT_FOLDER = cfg.getImportfolder()
PREPROC_FOLDER = cfg.getPreProcfolder()
ENCRYPTED_CONNSTR = ENCRYPTED_CONNSTR.encode("utf-8")
CURRENT_DATE = datetime.date.today()

def getPurgeFiles(documentId:str,fileName:str)->list:
    files = []
    try:
        import glob
        #Delete original pdf/tiff files
        #Delete files from split folder, tifffiles folder, logs, etc.,
        # fileName = os.path.basename(fileName)
        localFiles=[]
        fileName=fileName
        PAIGES_CLIENT_SRC = cfg.getpAIgesClientSRC()
        print(PAIGES_CLIENT_SRC,"DIR")
        localFiles_1 = glob.glob(PAIGES_CLIENT_SRC + '/*/*' + documentId + '*')

        localFiles_2 = glob.glob(IMPORT_FOLDER + '/*' + documentId + '*')
        localFiles_3 = glob.glob(PREPROC_FOLDER + '/*' + documentId + '*')
        # localFiles.extend([localFiles_1,localFiles_2,localFiles_3])
        localFiles_1.extend(localFiles_2)
        localFiles_1.extend(localFiles_3)
        # localFiles = glob.glob(rootFolderPath + '/*/*' + documentId + '*')
        # localRootFolderFiles = glob.glob(rootFolderPath + '/*' + documentId + '*')
        # localFiles_filename0 = glob.glob(rootFolderPath + '/*/*' + fileName + '*')
        # localFiles_filename1 = glob.glob(rootFolderPath + '/*' + fileName + '*')
        # localFiles1 = glob.glob(PAIGES_CLIENT_SRC + '/*/*' + documentId + '*')
        # print(localFiles1,"")
        # localRootFolderFiles1 = c + '/*' + documentId + '*')
        # localFiles_filename2 = glob.glob(PAIGES_CLIENT_SRC + '/*/*' + fileName + '*')
        # localFiles_filename3 = glob.glob(PAIGES_CLIENT_SRC + '/*' + fileName + '*')
        # localFiles_ = localFiles + localRootFolderFiles +localFiles_filename0 +localFiles_filename1
        
        # print("purge files list",localFiles)
        return localFiles_1
    except Exception as ex:
        print("getPurgeFiles exception",ex)
        return files

def purgeVmFiles(files:list)->dict:
    log = ''
    for file in files:
        try:
            os.remove(file)
            # f["file"] = "Deleted"
            log = log + str(file) + "Deleted"
        except Exception as e:
            log = log + str(file) + str(e)
            #f["file"] = str(e)+"OR File already deleted"
    return log

def downloadDocResult(f_name,documentId,dirPath):
    try:
        endPiont = UI_SERVER + "/document/result/get/" + documentId
        headers = {}
        headers["Content-Type"] = "application/json"
        print("endPiont:",endPiont)
        response = requests.get(endPiont,
                                headers = headers,
                                verify = False)
        print("response ",response)
        if response.status_code != 200:
            return False
        resp = response.json()
        # print("Extracted Data :",resp)
        if str(resp.get('params').get('status')).lower() == "failed":
            return False
        #getting list of header items from doc result
        headerItems = resp.get("result").get("document").get("documentInfo")
        with open(r"fieldMapping.json","r") as f:
            fieldMapping = json.load(f)
        extracted_data = {}
        if headerItems:
            for hdr_item in headerItems:
                f_key = hdr_item.get("fieldId")
                transformed_key = fieldMapping[f_key]
                f_val = hdr_item.get("fieldValue")
                extracted_data[transformed_key] = f_val

        inv_Date = extracted_data.get("invoiceDate")
        if inv_Date:
            try:
                extracted_data["invoiceDate"] = parser.parse(extracted_data.get("invoiceDate"),
                                                                dayfirst = True).date().strftime('%d-%b-%Y')
            except:
                extracted_data["invoiceDate"] = ""
        extracted_data["fileName"] = f_name
        if len(extracted_data)>0:
            fileName = os.path.splitext(f_name)[0] +'.json'
            domp_json = putil.saveTOJson(os.path.join(dirPath,fileName),extracted_data)
            if domp_json is None:
                return False
            return True 
        return False
    except:
        print("Get document result failed exception:",traceback.print_exc())
        return False

## monogDB purging record API call
def purgeMongoDBRecords(documentId:str):
    # ulr = http://52.172.153.247:7777/document/purgeDocument
    """
    {
    "id": "api.document.purge",
    "ver": "1.0",
    "ts": 1660022252453,
    "params": {
        "resmsgid": "",
        "msgid": "e468ef69-064f-4140-ba67-8c924876627e",
        "status": "Success",
        "err": "",
        "errmsg": "",
        "reason": ""
    },
    "responseCode": "OK",
    "result": {
        "documentId": "f52af359-fc3f-43a6-bfef-bc0235299560",
        "Message": "Document Purged successfully"
    }}
    """
    endpoint = "/document/purge" 
    urls = UI_SERVER + endpoint
    #urls = "http://172.22.0.10/document/purge"
    header =  {"Content-Type": "application/json"}
    payload =   {
                    "id": "api.document.purge",
                    "ver": "1.0",
                    "ts": 1659534515,   
                    "params": {"msgid": "e468ef69-064f-4140-ba67-8c924876627e"},
                    "request": {"documentId": "docId"}
                }
    payload["request"]['documentId']= documentId
    payload = json.dumps(payload)
    try:
        # print("URL :",urls,"\nPayload :",payload)
        response = requests.post(urls,
                                 headers = header,
                                 data = payload,
                                 verify = False)
        resp = response.json()
        # print("API response code :",response.status_code,"\nResponse :",response.json())
        if response.status_code == 200:
            if str(resp.get("params").get("status")).lower() == "success":
                return {"status_code":200,"message":"Purge document successful"}
        if response.status_code == 404:
            return {"status_code":response.status_code,"message":"Not found/Already Deleted"}
            
        return {"status_code":500,"message":resp.get("params").get("errmsg")}
    except Exception as ex:
        print("mongoDB purge record exception",ex)
        return {"status_code":500,"message":"Failed with error msg ->"+str(ex)}

class PurgeData:
    # def __init__(self,host, port, database, user,password):
    def __init__(self,conn_str):
        # self.host = host
        # self.port = port
        # self.database = database
        # self.user = user
        # self.password = password
        self.conn_str = conn_str
        # Set up logging
        # logging.basicConfig(filename='./fetch_records.log', level=logging.INFO,
        #                     format='%(asctime)s:%(levelname)s:%(message)s')
    def conn_to_db(self):
        logging.info("Connecting Database...")
        try:
            connection = psycopg2.connect(
                # host=self.host,
                # port=self.port,
                # database=self.database,
                # user=self.user,
                # password=self.password,
                # connect_timeout=3,
                # options='-c statement_timeout=3000'
                self.conn_str
            )
            return connection
        except Exception as e:
            logging.error(f"Error while connecting to database: {e}")
            return None

    def get_conn(self):
        return self.conn_to_db()

    def fetch_records(self,qurery):
        # Connect to database using connection pooling with pgbouncer
        records =[]
        try :
            connection =  self.conn_to_db()
            if not connection:
                raise
            logging.info("Connection Established")
            logging.info("Fetching records")        
            # Fetch records from database
            try:
                with connection.cursor() as  cursor:
                    cursor.execute(qurery)
                    records = cursor.fetchall()
                logging.info(f"Number of records featched : {len(records)}")
                return records
            except Exception as e:
                logging.error(f"Error while fetching records from database: {e}")
                raise
        except Exception as ex:
            return records
        finally :
            if connection:
                connection.close()
                logging.info("db connection closed")

    # Update records 
    def update_records(self,query,doc_id):
        try:
            # print("Doc Id:",doc_id)
            connection = self.conn_to_db()
            # Process the records
            # for record in records:
            # Update the record here
            print("query :",query)
            try:
                with connection.cursor() as cursor:
                    cursor.execute(query)
                    connection.commit()
                    logging.info(f"Updated record : {doc_id}")
            except Exception as e:
                print("query block",e)
                logging.error(f"Error while updating processed status of record with id {doc_id}: {e}")
                pass
        except:
            logging.info("Failed! While updatnig records")
        finally:
            if connection:
                connection.close()
                logging.info("db connection closed")
     
def remove_char(text:str='')->str:
    spacial = ["{","}","'",")","(",'""',"[","]",]
    for ch in spacial:
        text = str(text).replace(ch,'')
    return text

# purging records modified on 15 jan 2024
def purge_records_old(cnn_str):
    logging.info(f"Data purging Started")
    print("Data purging")
    # tbl.status = 'REVIEW_COMPLETED'::text AND to_timestamp((tbl.lastupdated / 1000)::double precision) < (CURRENT_DATE - '3 days'::interval) AND tbl.document_id = 'UAT_2d961246-05ca-11ee-a5c7-505bc2db505c'::text
    # WHERE status='REVIEW_COMPLETED' AND lastupdated < (CURRENT_DATE - '40 days'::interval) LIMIT 10;
    print(PURGE_FREQUENCY,CURRENT_DATE)
    if PURGE_FREQUENCY == -1:
        return None
    #get_purge_rcds = f"SELECT document_id,file_name,lastupdated,submitted_on FROM {PURGE_TABLE} WHERE status='{PURGE_STATUS}' AND lastupdated < (CURRENT_DATE - '{PURGE_FREQUENCY} days'::interval) ORDER BY lastupdated ASC;"
    # get_purge_rcds = f"SELECT document_id,file_name,lastupdated,submitted_on FROM {PURGE_TABLE} WHERE status NOT IN ({PURGE_STATUS}) AND lastupdated < (CURRENT_DATE - '{PURGE_FREQUENCY} days'::interval) ORDER BY lastupdated ASC;"
    get_purge_rcds = f"SELECT document_id,file_name,lastupdated,submitted_on FROM {PURGE_TABLE} WHERE lastupdated < (CURRENT_DATE - '{PURGE_FREQUENCY} days'::interval) ORDER BY lastupdated ASC;"
    purge = PurgeData(cnn_str)
    print(purge)
    records = purge.fetch_records(get_purge_rcds)
    print("No of records :",len(records))
    docs = []
    blobs = []
    purged_status =[]
    purged_msg =[]
    for doc in records:
        doc_id = doc[0]
        file_name = doc[1]
        lastupdated = doc[2]
        submitted_on = doc[3]
        # blob_path = doc[1]
        # consumed = doc[2]
        # manual = doc[3]
        print("Purging record :",doc)
        flag = 0
        msg = ""
        t1= time.time()
        mg_status = purgeMongoDBRecords(doc_id)
        print("time taken to purge record from mongo db is:", time.time()-t1)
        # print("mg_status :",mg_status)
        msg = "Purging MongoDB records : status_code: " + str(mg_status.get("status_code")) +" message: "+ str(mg_status.get("message"))
        print(msg,"Look into the message after mongo deletion")
        if mg_status.get("status_code") in [200]:
            # if mongo record deleted setting purge flag to 1
            flag = 1
            print(flag,"Purge flat status changed")
            t2 = time.time()
            vm_files = getPurgeFiles(doc_id,file_name)#changed blob path to file name
            print("vm_files :",vm_files)
            if len(vm_files) == 0:
                msg = msg + ", Purged VM files : "+str("Files Not found/ Already Deleted")
            else :
                msg = msg + ", Purged VM files : "+ str(purgeVmFiles(vm_files))
            print("Time taken to purge vm file is:", time.time()- t2)
            upd_purge_status =f"UPDATE PUBLIC.{DB_NAME} SET purged={flag},purged_timestamp ='{datetime.datetime.now()}',purged_log='{remove_char(str(msg))}' WHERE document_id ='{doc_id}';"
        
    
            # upd_purge_status =f"UPDATE PUBLIC.{DB_NAME} SET purged={flag},purged_timestamp ='{datetime.datetime.now()}',purged_log='{remove_char(str(msg))}' WHERE documentid ='{doc_id}';"
            t3 = time.time()
            purge.update_records(upd_purge_status,doc_id)
            print("Time taken to execute update query is:", time.time()-t3)
            print("Updated log msg :",msg)
            docs.append(doc_id)
            # blobs.append(blob_path)
            purged_status.append(flag)
            purged_msg.append(msg)
        sys.stdout.flush()
        
    df = pd.DataFrame()
    df["documentid"] = docs
    # df["blobName"] = blobs
    df["purged_status"] = purged_status
    df["purged_msg"] = purged_msg
    purge_result = "./files/"+str(time.time()) + "purged_docs.csv"
    df.to_csv(purge_result)
    logging.info(f"Data purging Done")
    if os.path.isfile(purge_result):
        try:
            print("purging was successful in both vm and mongo")
            # send_email(subject="Purging Report",files=[purge_result])
        except:
            print("Sending email failed")
            logging.info(f"Sending email failed :"+str(traceback.print_exc()))

def purge_records_db():
    """
    Purge records from Db which are older than 60 days

    Returns:
        bool: True/False. Final o/p is csv file which contain file status along with msg for different doc id's
    """
    try:
        print("Data purging started for db")
        get_purge_rcds_db = f"SELECT document_id,file_name,lastupdated,submitted_on FROM {PURGE_TABLE} WHERE lastupdated < (CURRENT_DATE - '{PURGE_FREQUENCY_DB} days'::interval) ORDER BY lastupdated ASC;"
        print(PURGE_FREQUENCY_DB,CURRENT_DATE)
        if PURGE_FREQUENCY_DB == -1: 
            return None
        purge = PurgeData(cnn_str)
        print(purge)
        records_db = purge.fetch_records(get_purge_rcds_db)
        print("No of records in db deletion is:",len(records_db))
        docs = []
        purged_status_db =[]
        purged_msg_db =[]
        for doc in records_db:
            doc_id = doc[0]
            flag = 0
            print("Purging record :",doc)
            
            msg = ""
            t1= time.time()
            mg_status = purgeMongoDBRecords(doc_id)
            if mg_status.get("status_code") in [200]:
                flag = 1
                print("Purge flat status changed", flag)
            print("time taken to purge record from mongo db is:", time.time()-t1)
            # print("mg_status :",mg_status)
            msg = "Purging MongoDB records : status_code: " + str(mg_status.get("status_code")) +" message: "+ str(mg_status.get("message"))
            print(msg,"Look into the message after mongo deletion")
            docs.append(doc_id)
            # blobs.append(blob_path)
            purged_status_db.append(flag)
            purged_msg_db.append(msg)
        df_db = pd.DataFrame()
        df_db["documentid"] = docs
        # df["blobName"] = blobs
        df_db["purged_status"] = purged_status_db
        df_db["purged_msg_db"] = purged_msg_db
        purge_result_db = "./files/"+str(time.time()) + "purged_docs.csv"
        df_db.to_csv(purge_result_db)
        if os.path.isfile(purge_result_db):
            try:
                print("purging was successful in both vm and mongo")
                return True
                # send_email(subject="Purging Report",files=[purge_result])
            except Exception as e:
                print("Exception occured in deletion from file share", e)
                return False
    except Exception as e:
        print("Exception occured in purge_records_db", e)
        return False
        
def purge_records_fileshare():
    """
    Purge records from Filesahre which are older than 30 days
    Returns:
        bool: True/False. Final o/p is csv file which contain file status along with msg for different doc id's
    """
    try:
        print("Data purging started for fileshare")
        get_purge_rcds_fileshare = f"SELECT document_id,file_name,lastupdated,submitted_on FROM {PURGE_TABLE} WHERE lastupdated < (CURRENT_DATE - '{PURGE_FREQUENCY_FILESHARE} days'::interval) ORDER BY lastupdated ASC;"
        print(PURGE_FREQUENCY_FILESHARE,CURRENT_DATE)
        if PURGE_FREQUENCY_FILESHARE == -1 : 
            return None
        purge = PurgeData(cnn_str)
        print(purge)
        records_fileshare = purge.fetch_records(get_purge_rcds_fileshare)
        print("No of records in fileshare deletion is:",len(records_fileshare))
        docs_fileshare = [] 
        purged_status_fileshare = []
        purged_msg_fileshare = []
        for doc in records_fileshare:
            flag = 0
            doc_id = doc[0]
            file_name = doc[1]
            msg = ""
            t2 = time.time()
            vm_files = getPurgeFiles(doc_id,file_name)#changed blob path to file name
            print("vm_files :",vm_files)
            if len(vm_files) == 0:
                msg = msg + ", Purged VM files : "+str("Files Not found/ Already Deleted")
            else :
                msg = msg + ", Purged VM files : "+ str(purgeVmFiles(vm_files))
                flag = 1
            print("Time taken to purge vm file is:", time.time()- t2)
            docs_fileshare.append(doc_id)
            purged_status_fileshare.append(flag)
            purged_msg_fileshare.append(msg)
        df_fileshare = pd.DataFrame()
        df_fileshare["documentid"] = docs_fileshare
        df_fileshare["purged_status"] = purged_status_fileshare
        df_fileshare["purged_msg_db"] = purged_msg_fileshare
        purge_result_fileshare = "./files/"+str(time.time()) + "purged_docs.csv"
        df_fileshare.to_csv(purge_result_fileshare)
        if os.path.isfile(purge_result_fileshare):
            try:
                print("purging was successful in both vm and mongo")
                return True
                # send_email(subject="Purging Report",files=[purge_result])
            except Exception as e:
                print("Exception occured in deletion from file share", e)
                return False
    except Exception as e:
        print("Exception occured in purge_records_fileshare", e)
        return False

# purging records
def purge_records(cnn_str):
    """
    This function is used to purge the data from Database and fileshare. Both of them are having different frequencies.
    Args:
        cnn_str (_type_): Requires connection string to connect to database
    Returns:
        None
    """
    print("Data Purging Started")
    status_db = purge_records_db()
    # status_fileshare = purge_records_fileshare()
    # print(f"Data Purging completed, Status of db is {status_db}, status of fileshare is {status_fileshare}") 
    print(f"Data Purging completed, Status of db is {status_db}")           
    logging.info(f"Data purging Done")
    return None


if __name__ == "__main__":
    try:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        # log_file_path = os.path.join(ROOT_DIR, "./files/Reference_Data_Auto_Update.log")
        log_file_path = os.path.join(f"./files/{str(datetime.date.today())}_S3_upload_and_purging.log")
        log_file_folder = os.path.join("./files")
        if not os.path.exists(log_file_folder):
            print("making dir")
            os.mkdir("files")
        log_file = open(log_file_path,"w")
        sys.stdout = log_file
        sys.stderr = log_file
        print("testing log")
        # logging.basicConfig(filename=f'./files/{str(datetime.date.today())}_S3_upload_and_purging.log', level=logging.INFO,
        #                 format='%(asctime)s:%(levelname)s:%(message)s')
        # cnn_str = "host=127.0.0.1 port=6432 user=swginsta_admin@swginstapaiges dbname=swginsta_pAIges"
        # cnn_str = "host=127.0.0.1 port=5432 user=swginsta_admin@swginstapaiges dbname=swginsta_pAIges"
        #cnn_str = "host=bcppaigessql.postgres.database.azure.com port=5432 user=bcppaiges@bcppaigessql dbname=bcp-tapp password=jm|-b'T3*9,dj4%"


        # Initialize the Fernet cipher with the encryption key
        cipher = Fernet(KEY)
        
        # Decrypt the connection string
        decrypted_cnn_str = cipher.decrypt(ENCRYPTED_CONNSTR)
        
        # Convert the decrypted bytes to string
        cnn_str = decrypted_cnn_str.decode('utf-8')
        
        purge_records(cnn_str)
        # uploading files to s3 bucket
        # upd_data_to_s3(cnn_str)
    except:
        print("main function exception ",traceback.print_exc())
    finally:
        if log_file is not None:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            log_file.close()
        pass
    

