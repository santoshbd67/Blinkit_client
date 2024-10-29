import os
import sys 
import json
import traceback
import copy
import datetime

import requests
import pandas as pd

import TAPPconfig as cfg
import preProcUtilities as putil

from post_processor import wordshape
from dateutil.parser import parse

NoOfDays = 2
if len(sys.argv)>0:
    NoOfDays = int(sys.argv[1])

class captureReferenceData():
    '''

    '''

    def __init__(self) -> None:
        self .SUBSCRIBER_ID = cfg.getSubscriberId()
        self.PREPROC_SERVER = cfg.getPreprocServer()
        self.UI_SERVER = cfg.getUIServer()
        self.GET_DOCUMENT_RESULT = cfg.getDocoumentResult()
        self.FIND_DOCUMENT = cfg.getDocumentFind()
        self.RESULT_DOWNLOAD_FOLDER = cfg.getUIDownloadFolder()
        self.header  = {"Content-Type":"application/json"}
        self.NoOfDays = NoOfDays


    def get_back_date_epoc_timestamp(self)->int:
        date = datetime.datetime.now() - datetime.timedelta(days=self.NoOfDays)
        return int(datetime.datetime.timestamp(date)*1000)
    
    def convert_epoc_time_into_date_format(self, epoc_time_stamp:int)->datetime.datetime or None:
        try:
            date = datetime.datetime.fromtimestamp(int(epoc_time_stamp)/1000)
            return date.strftime("%d/%m/%y")
        except Exception as exception:
            print("convert_epoc_time_into_date_format exception",exception)
            return None

    def get_doc_filter(self,)-> dict:


        """ Returns filter request with form given num of days back """

        return {
                    "id": "api.document.find",
                    "ver": "1.0",
                    "ts": 1571813791276,
                    "params": {
                        "msgid": ""
                    },
                    "request": {
                        "token": "",
                        "filter": {
                            "status": "REVIEW_COMPLETED","stp": False,
                            "lastUpdatedOn": {">":self.get_back_date_epoc_timestamp()}
                        },
                        "offset": 0,
                        "limit": 0,
                        "page": 1
                    }
                }
    def get_doc_filter1(self,)-> dict:
        """ Returns filter request with form given num of days back """
        return {
                    "id": "api.document.find",
                    "ver": "1.0",
                    "ts": 1571813791276,
                    "params": {
                        "msgid": ""
                    },
                    "request": {
                        "token": "",
                        "filter": {
                            "status": "REVIEW_COMPLETED","stp": False,
                            "lastUpdatedOn": {">":self.get_back_date_epoc_timestamp()}
                        },
                        "offset": 0,
                        "limit": 0,
                        "page": 1
                    }
                }
    def get_doc_filter2(self,)-> dict:
        """ Returns filter request with form given num of days back """
        return {
                    "id": "api.document.find",
                    "ver": "1.0",
                    "ts": 1571813791276,
                    "params": {
                        "msgid": ""
                    },
                    "request": {
                        "token": "",
                        "filter": {
                            "status": "RPA_FAILED","stp": False,
                            "lastUpdatedOn": {">":self.get_back_date_epoc_timestamp()}
                        },
                        "offset": 0,
                        "limit": 0,
                        "page": 1
                    }
                }
    def get_doc_filter3(self,)-> dict:
        """ Returns filter request with form given num of days back """
        return {
                    "id": "api.document.find",
                    "ver": "1.0",
                    "ts": 1571813791276,
                    "params": {
                        "msgid": ""
                    },
                    "request": {
                        "token": "",
                        "filter": {
                            "status": "RPA_PROCESSED","stp": False,
                            "lastUpdatedOn": {">":self.get_back_date_epoc_timestamp()}
                        },
                        "offset": 0,
                        "limit": 0,
                        "page": 1
                    }
                }
    
    def get_filtered_docs_perpage(self,filter_request : dict)-> list:
        doc_ids = []
        page_num = None
        try:
            reps = requests.post(url=(self.UI_SERVER + self.FIND_DOCUMENT),
                                 headers = self.header,
                                 data = json.dumps(filter_request))
            # print("reps :",reps)
            if reps.status_code != 200:
                return page_num,doc_ids
            response = reps.json()
            if str(response.get("params").get("status")).lower() == "success":
                docs_list = response.get('result').get("documents")
                print("docs list :",len(docs_list))
                page_num = response.get("result").get("page")
                for itms in docs_list:
                    doc_ids.append({"documentId":itms.get("documentId"),
                                    "lastUpdatedOn": self.convert_epoc_time_into_date_format(itms.get("lastUpdatedOn"))})
            return page_num,doc_ids
        except:
            print("get_reviewed_perpage exception",traceback.print_exc())
            return page_num,doc_ids

    def get_all_reviewed_docs(self)->list:
        doc_ids = []
        try:
            urls = self.UI_SERVER + self.FIND_DOCUMENT
            for doc_filter in [self.get_doc_filter1(), self.get_doc_filter2(), self.get_doc_filter3()]:
                filter_request = doc_filter

                print("urls",urls,self.header,filter_request)
                reps = requests.post(url = urls,headers = self.header,data = json.dumps(filter_request))
                # print("reps2 :",reps)
                if reps.status_code != 200:
                    return doc_ids
                # pageCount = 1
                reps = reps.json()
                # print("reps keys:",reps)
                if (reps.get("result")) :
                    pageCount = reps.get("result").get("page")
                    count = reps.get("result").get("count")
                    perpage_records = reps.get("result").get("perPageRecords")
                    print("pageCount",pageCount,"count :",count,"\tperPageRecords :",perpage_records)
                    docs_ls = reps.get('result').get("documents")
                    # print("docs list :",docs_list)
                    for itms in docs_ls:
                        # print("stp status",itms.get("stp"))
                        doc_ids.append({"documentId":itms.get("documentId"),
                                        "lastUpdatedOn":self.convert_epoc_time_into_date_format(itms.get("lastUpdatedOn"))})
                    print("list of document id :",len(doc_ids))
                    try:    
                        while (int(count)>(pageCount * int(perpage_records))):
                            pageCount = pageCount + 1
                            filter_request["request"]["page"] = pageCount
                            print("Requesting Page No :",filter_request)
                            pageno, docids = self.get_filtered_docs_perpage(filter_request)
                            print("inside loop pageno",pageno,"\tdoc ids ",len(docids))
                            if (pageno is not None) and len(docids)>0:
                                doc_ids.extend(docids)
                            else:
                                print("Getting reviewed doc error")
                                break
                    except :
                        print("getting all pages docs exception")
                        pass
            return doc_ids
        except:
            print("get_all_reviewed_docs exception ",traceback.print_exc())
            return doc_ids

    def get_document_result(self, documentId : str) -> dict or None :

        try : 
            endPiont = self.UI_SERVER + self.GET_DOCUMENT_RESULT + documentId
            headers = {}
            headers["Content-Type"] = "application/json"
            doc_rs = requests.get(endPiont)
            # print("doc_rs :",doc_rs)
            if doc_rs.status_code != 200:
                return None
            #doc_rs = json.loads(doc_rs.text)
            # print("str",doc_rs.text)
            doc_rs = doc_rs.json()
            # documentInfo = doc_rs.get("result").get("document").get("documentInfo")
            # print("documentInfo :",documentInfo)
            return doc_rs
        except :
            print("get_document_result exception ",traceback.print_exc())
            return None

    # def capture_feild_reference_data(self,fields : list, metadata : dict, doc_result : dict) ->  dict:
    #     data = {} 
    #     try:
    #         for i in fields:
    #             data[i] ={}
    #         print("data :",data)
    #         filed_list = doc_result["result"]['document']['documentInfo']
    #         for i in data.keys():
    #             data[i]['submitted_on'] = metadata["lastUpdatedOn"]
    #             data[i]['doc_id'] = metadata["documentId"]
    #         print("data :",data)
    #         for item in filed_list:
    #             if item.get("filedId") == "vendorName":
    #                 for f in data.keys():
    #                     data[f]['vendor_name'] = item.get("fieldValue")

    #             if item.get("filedId") == "vendorGSTIN":
    #                 for f in data.keys():
    #                     data[f]['vendor_id'] = item.get("fieldValue")
    #             print("data :",data)
    #             if item.get("fieldId") in fields:
    #                 if 'correctedValue' in item.keys():

    #                     pred_file_path = dwonload_pred_files(str(i["documentId"]))
    #                     print("pred_file_path :",pred_file_path)
    #                     if not(pred_file_path):
    #                         continue
    #                     df1 = pd.read_csv(pred_file_path,encoding="utf-8")
    #                     df2=df1[df1['text']==item.get("correctedValue")]
    #                     if df2.shape[0]>0:
    #                         data[item.get("correctedValue")]['left'] = (df2['left'] * df2['image_widht']).values[0]
    #                         data[item.get("filedId")]['right'] = (df2['right'] * df2['image_widht']).values[0]
    #                         data[item.get("filedId")]['top'] = (df2['top'] * df2['image_height']).values[0]
    #                         data[item.get("filedId")]['bottom'] = (df2['bottom'] * df2['image_height']).values[0]
    #                         data[item.get("filedId")]["field_shape"] = wordshape(str(item.get("correctedValue")))
    #                         data[item.get("filedId")]['status'] = 1 if item.get("correctedValue") else 0
    #                     print("data :",data)
    #                 else :
    #                     data[item.get("filedId")][item.get("filedId")] =  item.get("fieldValue")
    #                     data[item.get("filedId")]['left'] = item['boundingBox']['left']
    #                     data[item.get("filedId")]['right'] = item['boundingBox']['right']
    #                     data[item.get("filedId")]['top'] = item['boundingBox']['top']
    #                     data[item.get("filedId")]['bottom'] = item['boundingBox']['bottom']
    #                     data[item.get("filedId")]["field_shape"] = wordshape(str(item.get("fieldValue")))
    #                     data[item.get("filedId")]['status'] = 0
    #                     print("data :",data)
    #         print("Data :",data)
    #         return data 
    #     except :
    #         print(" exception :",traceback.print_exc())
    #         return data
        
    def capture_feild_reference_data(self,field : str, metadata : dict, doc_result : dict) ->  dict:
        data = {} 
        try:
            filed_list = doc_result["result"]['document']['documentInfo']
            data['submitted_on'] = metadata["lastUpdatedOn"]
            data['doc_id'] = metadata["documentId"]
            print("data :",data)
            for item in filed_list:
                # print("item :",item)
                if item.get("fieldId") == "vendorName":
                    data['vendor_name'] = item.get("correctedValue") if (item.get("correctedValue")) else item.get("fieldValue")

                if item.get("fieldId") == "vendorGSTIN":
                    data['vendor_id'] = item.get("correctedValue") if (item.get("correctedValue")) else item.get("fieldValue")

                if item.get("fieldId") == field:
                    print("field name :",item.get("fieldId"))
                    if 'correctedValue' in item.keys():
                        data["field_name"] =  item.get("fieldId")
                        pred_file_path = dwonload_pred_files(str(metadata["documentId"]))
                        print("pred_file_path :",pred_file_path)
                        if not(pred_file_path):
                            return {}
                        df1 = pd.read_csv(pred_file_path,encoding="utf-8")
                        df2=df1[df1['text']==item.get("correctedValue")]
                        if df2.shape[0]==1:
                            data['left'] = (df2['left'] * df2['image_widht']).values[0]
                            data['right'] = (df2['right'] * df2['image_widht']).values[0]
                            data['top'] = (df2['top'] * df2['image_height']).values[0]
                            data['bottom'] = (df2['bottom'] * df2['image_height']).values[0]
                            data["field_shape"] = wordshape(str(item.get("correctedValue")))
                            data['status'] = 1 
                        else:
                            return {}
                    else :
                        data["field_name"] =  item.get("fieldId")
                        data['left'] = item['boundingBox']['left']
                        data['right'] = item['boundingBox']['right']
                        data['top'] = item['boundingBox']['top']
                        data['bottom'] = item['boundingBox']['bottom']
                        data["field_shape"] = wordshape(str(item.get("fieldValue")))
                        data['status'] = 1
                        print("data :",data)
            print("Data :",data)
            return data 
        except :
            print(" exception :",traceback.print_exc())
            return {}

    def get_data(self, df, fields : list, docs : list):
        new_data = pd.DataFrame(columns=df.columns) 
        try:
            for idx,doc_info in enumerate(docs):
                print("\nRefreshing Master data for :",idx," --> ",doc_info["documentId"])
                res =  self.get_document_result(doc_info["documentId"])
                # print("doc result",res)
                if not(res):
                    print("Failed while getiing result!")
                    continue
                field_list = res["result"]['document']['documentInfo']
                vendor_gstin = None
                vendor_gstin_count = 0
                for f in field_list:
                    if (f.get("fieldId") == "vendorGSTIN"):
                        vendor_gstin = f.get("fieldValue")
                if vendor_gstin and vendor_gstin != "N/A":
                    vendor_gstin_count = len(df[df['vendor_id'] == vendor_gstin])
                    if vendor_gstin_count >=30:
                        print("Vendor records are more than 10 records",vendor_gstin)
                        continue
                    print("Vendor reference data record is less than 10 records")
                    s1 = vendor_gstin_count < 20
                    print("require minimum records to meet 10 record per vendor:",(20-vendor_gstin_count),"new df gstin count :",len(df[df['vendor_id'] == vendor_gstin]),s1)

                    # if len(df[df['vendor_id'] == vendor_gstin])< (10-vendor_gstin_count):
                    if vendor_gstin_count < 30:
                        print("inside ")
                        for f in fields:
                            captured_data = self.capture_feild_reference_data(field=f,
                                                                metadata=doc_info,
                                                                doc_result=res)
                            print("captured_data :",captured_data,len(captured_data),type(captured_data))
                            df,captured_data=check_refdata_intersection(df,captured_data)
                            print("captured_data after  :",captured_data,len(captured_data),type(captured_data))
                            if len(captured_data)>0:
                                captured_data['layout'] = 0
                                captured_data['manual_updated'] = 0
                                captured_data['row_id'] = 0
                                captured_data['is_there'] = False
                                captured_data['review_status'] = 0
                                dftest = pd.DataFrame([captured_data])
                                print(df.columns,len(df.columns))
                                if len(captured_data)>0:
                                    # new_data = new_data.append(captured_data, ignore_index = True)
                                    new_data = pd.concat([new_data, dftest], ignore_index=True)
                    else :
                        print("15 records captured")
            return new_data,df
        except:
            print("get Data exception :",traceback.format_exc())
            return new_data,df

SubscriberId = cfg.getSubscriberId()
RESULT_DOWNLOAD_FOLDER = cfg.getUIDownloadFolder()

def dwonload_pred_files(doc_id:str):
    try:
        file = doc_id +"_pred.csv"
        blob_downloads = [os.path.join(SubscriberId,file)]
        # Define local download path and file name RESULT_DOWNLOAD_FOLDER
        #local_downloads = [os.path.join(ROOT_DIR,RESULT_DOWNLOAD_FOLDER,file)]
        local_downloads = [os.path.join(RESULT_DOWNLOAD_FOLDER,file)]
        print("download folders: ", blob_downloads, local_downloads)
        downloadUrls = zip(blob_downloads,local_downloads)
        FileDownloadStatus = putil.downloadFilesFromBlob(downloadUrls)
        if FileDownloadStatus:
            return local_downloads[0]
        return None
    except:
        print("file dwn exp ")
        return None
def intersection_ratio(rect1, rect2):
    try:
        # Calculate the top-left and bottom-right coordinates of the overlapping rectangle
        x1 = max(rect1[0], rect2[0])
        y1 = min(rect1[1], rect2[1])
        x2 = min(rect1[2], rect2[2])
        y2 = max(rect1[3], rect2[3])
        # Calculate the area of the overlapping rectangle
        intersection_area = max(0, x2 - x1) * max(0, y1 - y2)
        # print(intersection_area)
        # Calculate the areas of the individual rectangles
        area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
        area2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
        # Find the smaller area of the two rectangles
        smaller_area = min(area1, area2)
        # Calculate the ratio of intersection area over the smaller area
        ratio = abs(intersection_area / smaller_area)
        return ratio
    except:
        print("exception occured in getting intersection_ratio",traceback.print_exc())
        return 0
def check_refdata_intersection(df,row2):
    try:
        df1=df[df["vendor_id"] == row2["vendor_id"]]
        #print(df1)
        df1 = df1[df1['field_name']==row2["field_name"]]
        i=0
        print(df1.shape)
        if (df1.shape[0])>0:
            for index1,row1 in df1.iterrows():
                if ((row1.bottom<=float(row2['top'])) | (row1.left>=float(row2['right'])) | (row1.right<=float(row2['left'])) | (row1.top>=float(row2['bottom']))):
                    i=1+i
                    if (i>=df1.shape[0]):
                        print("new format referencedata")
                        return df,row2           
                elif intersection_ratio([row1.left,row1.bottom,row1.right,row1.top],[float(row2['left']),float(row2['bottom']),float(row2['right']),float(row2['top'])])>0.5:
                    print("intersection area is 0.5 so increasing boundary box for",row1)
                    if row2["field_shape"] in list(df1["field_shape"]):
                        df.at[index1, 'left'] = min(row1.left, float(row2['left']))
                        df.at[index1, 'right'] = max(row1.right, float(row2['right']))
                        df.at[index1, 'top'] = min(row1.top, float(row2['top']))
                        df.at[index1, 'bottom'] = max(row1.bottom, float(row2['bottom']))
                        return df,{}
                    else:
                        print("wordshape is not present")
                        return df,row2
                else:
                    print(intersection_ratio([row1.left,row1.bottom,row1.right,row1.top],[float(row2['left']),float(row2['bottom']),float(row2['right']),float(row2['top'])]))
                    i=i+1
                    if (i>=df1.shape[0]):
                        print("new format referencedata")
                        return df,row2
        else:
            print("first time vendor reference data")
            return df,row2
    except:
        print("exception in comparing with old ref data",traceback.print_exc())
        return df,row2

def main():
    try:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        # log_file_path = os.path.join(ROOT_DIR, "./files/Reference_Data_Auto_Update.log")
        log_file_path = os.path.join("./files/Reference_Data_Auto_Update.log")
        log_file_folder = os.path.join("./files")
        if not os.path.exists(log_file_folder):
            print("making dir")
            os.mkdir("files")
        log_file = open(log_file_path,"w")
        sys.stdout = log_file
        sys.stderr = log_file
        print("Strt Refreshing reference data :",datetime.datetime.now())
        REFERENCE_MASTER_DATA_PATH = cfg.getReferenceMasterData()  
        #REFERENCE_MASTER_DATA = os.path.join(ROOT_DIR,REFERENCE_MASTER_DATA_PATH)
        REFERENCE_MASTER_DATA = os.path.join(REFERENCE_MASTER_DATA_PATH)
        print("REFERENCE_MASTER_DATA :",REFERENCE_MASTER_DATA)
        capture = captureReferenceData()
        document_ids = capture.get_all_reviewed_docs()
        print("document_ids :",len(document_ids))
        ref_data = pd.read_csv(REFERENCE_MASTER_DATA, encoding='utf-8')
        print("ref_data :",ref_data.shape,"\n ref data",ref_data.columns)
        print("Manual updated count",ref_data["manual_updated"].value_counts())
        df,ref_data= capture.get_data(ref_data,docs = document_ids,fields=["invoiceDate","invoiceNumber"])
        print("New Ref data:",df.shape)
        # df.to_csv("New_Reference_Data.csv")
        if not(df.shape[0]):
            print("exit shape[0] not")
            return
        print("ref_data :",ref_data.shape,"\n ref data",ref_data.columns)
        print("Manual updated count",ref_data["manual_updated"].value_counts())

        ref_data["is_there"] = ref_data["doc_id"].isin(df["doc_id"].unique())
        print("is_there count ref_data",ref_data["is_there"].value_counts())
        ref_data1 = ref_data[ref_data["is_there"]== False]
        print("data shape after removing is_there",ref_data1.shape)
        ref_data1 = pd.concat([ref_data1,df], axis = 0)
        print("ref_data after appending :",ref_data1.shape)
        ref_data1.drop_duplicates(subset=['vendor_id','field_name','bottom','left','right','top'],inplace=True)
        columns_to_drop = [col for col in df.columns if 'unnamed' in col.lower()]
        ref_data1.drop(columns=columns_to_drop, inplace=True)
        print("ref_data after removing duplicates :", ref_data1.shape)
        # final_path = r"C:\Users\Admin\Desktop\GROFERS_VENDOR_EXTRACTION_HISTORY.csv"
        # ref_data1.to_csv(final_path,index = False,encoding="utf-8")
        print(REFERENCE_MASTER_DATA)
        ref_data1.to_csv(REFERENCE_MASTER_DATA,index = False,encoding="utf-8")
        return 
    except:
        print("main function exception ",traceback.print_exc())
        return 
    finally:
        if log_file is not None:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            log_file.close()
        pass


if __name__ == '__main__':

    main()
