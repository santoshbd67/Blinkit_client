# Final code
import time
import string
import re
import os
import pandas as pd
import traceback
import PyPDF2
import requests
import json
from typing import Dict, Union
import TAPPconfig as cfg
from preProcUtilities import downloadFilesFromBlob , uploadFilesToBlobStore
from preProcUtilities import join_path
import preProcUtilities as putil

UI_ROOT_FOLDER = cfg.getUIRootFolder()
UI_UPLOAD_FOLDER = cfg.getUI_UPLOAD_FOLDER()
SUBSCRIBER_ID = cfg.getSubscriberId()


class ProcessDsicrepancyNote:
    def __init__(self, title_regex=r"discrepancy note",
                 footer_regex = r"total payable|payable", 
                 field1_regex=r"purchase id|purchase num|purchase no|po no", 
                 field2_regex=r"invoice no|invoice id|invoice num"):
        self.title_regex = title_regex
        self.footer_regex = footer_regex
        self.field1_regex = field1_regex
        self.field2_regex = field2_regex
    
    def is_discrepancy_note(self, text:str):
        try:
            translator = str.maketrans('', '', string.punctuation)
            text = text.translate(translator)
            text = text.lower()

            t1 = re.search(self.title_regex, text)
            m1 = re.search(self.field1_regex, text)
            m2 = re.search(self.field2_regex, text)
            # print("t1 :",t1,"\nm1 :",m1,"\nm2 :",m2)
            if ((t1 is not None)):
                return True
            # else:
                # print("Mandatory labels not matched")
            return False
        except Exception as e:
            print("is_discrepancy_note identification exception", e)
            return False
    
    def save_customise_pdf(self, filePath:str,
                           pages:list,
                           new_file_name:str=None):

        try:
            pages.sort()
            pdf_reader = PyPDF2.PdfReader(open(filePath, 'rb'))
            pdf_writer = PyPDF2.PdfWriter()
            for page in pages:
                if page in range(len(pdf_reader.pages)):
                    pdf_writer.add_page(pdf_reader.pages[int(page)])
                else:
                    print("Page not present in the pdf")
                    return
            if not new_file_name:
                modified_name  = os.path.splitext(os.path.basename(filePath))[0] + "_DISCR.pdf"
                new_file_name = os.path.join(os.path.dirname(filePath),modified_name)
            with open(new_file_name, 'wb') as output_file:
                pdf_writer.write(output_file)
            return new_file_name
        except Exception as E:
            print(f"save_customise_pdf exception :{E}")
            return

    def is_footer_present(self,text:str)-> bool:
        try :
            translator = str.maketrans('', '', string.punctuation)
            text = text.translate(translator)
            text = text.lower()

            if re.search(self.footer_regex, text):
                return True
            else:
                return False
        except Exception as ex:
            print("is_footer_present exception :",ex)
            return False

    def get_discrepancy_pages_from_df(self,df:pd.DataFrame)->list:
        pages = []
        is_multipage = False
        try:
            total_pages = list(df["page_num"].unique())
            print("total inv paages :",total_pages)
            for pg in total_pages:
                print("DSCR is_multipage :",is_multipage)
                temp = df[(df["page_num"]==pg) & (df["line_num"]<15)]
                text = " ".join(temp["line_text"].astype('str'))
                footer_temp = df[(df["page_num"]==pg)]
                footer_text = " ".join(footer_temp["line_text"].astype('str'))
                # Added second-page discrepanncy note identificcation whose header not present 11 May
                # if self.is_discrepancy_note(text=text):
                #        pages.append(pg)
                if (self.is_discrepancy_note(text=text)) and (self.is_footer_present(text=footer_text)):
                    pages.append(pg)
                    print("Header and footer found")
                    is_multipage = False
                else:
                    if self.is_discrepancy_note(text=text):
                        pages.append(pg)
                        print("Header only found")
                        is_multipage = True
                    else:
                        if is_multipage:
                            print("Checking second DSCR page")
                            if self.is_footer_present(text=footer_text):
                                print("appending second DSCCR page",pg)
                                pages.append(pg)
                                print("seccond page maatch found")
                            else:
                                print("second page match not found")
                            is_multipage = False
                        else:
                            print("is_multipage not True")
                # Added second-page discrepanncy note identificcation whose header not present 11 May
            return pages
        except Exception as ex:
            print("get_discrepancy_pages_from_df exceptionn :",str(ex))
            return []
        
    def get_discrepancy_note(self,pdf_file_path:str,pages:list=[])->(str or None):
        if len(pages)>0:
            try:
                saved_file = self.save_customise_pdf(filePath=os.path.join(pdf_file_path),pages=pages)
                return saved_file
            except Exception as E:
                print("saving save_customise_pdf exception",E)
                return None
        else :
            return None
        
    def remove_discrepancy_pages_from_df(self,df:pd.DataFrame,pages:list)->pd.DataFrame:
        try:
            if len(pages) > 0:
                df = df[~df["page_num"].isin(pages)]
                # 24 August 2023 Added drop = True
                df.reset_index(drop = True, inplace=True)
            return df
        except:
            print("remove_discrepancy_pages_from_df exception:",traceback.print_exc())
            return df

class ingest_discreopancy_note():
    def __init__(self,discr_file:str,meta_data:dict=None):
        self.discr_file = discr_file
        self.UI_AGENT_SERVER =  cfg.GET_UI_AGENT_SERVER()
        self.RELATIVE_PATH = "/ui/upload/invoice"
        self.meta_data = meta_data
    
    def payload(self):
        data = {
                "api_name": "upload",
                "attachment_no": 1,
                "blobname": "",
                "documentId": "",
                "fileName": "",
                "message_id": "",
                "record_id": "",
                "size": 1234,
                "received_time": int(time.time()*1000),
                "received_from": "test.com",
                "subject": "test",
                "docType": "Discrepancy Note",
                "vendor_name": "",
                "upload_mode": "Email",
                "userId": "b5f301eff4794e9bbd8bcf0037d53fcf",
                "rpa_upload_time": int(time.time()*1000),
                "Invoice_Discr_pages":[],
                "linked_document":{}
                }
        try:
            if self.meta_data:
                m_data = self.meta_data.get("result").get("document")
                data["documentId"]= str(m_data.get("documentId"))+"_DISCR"
                data["Vendor_Name"]= m_data.get("Vendor_Name","")
                data["size"]= m_data.get("size",12345)
                data["fileName"]= m_data.get("fileName")
                data["blobname"]= m_data.get("blobname") 
                data["message_id"]= m_data.get("message_id","")
                data["record_id"] = m_data.get("record_id","") + "_DISCR"
                #change made for priority ranking 25th Oct
                data["priorityRanking"] = m_data.get("priorityRanking",2)
        except Exception as E:
            print("forming paload exception :",E)
        print("formed payload :",data)
        return data

    # without retry
    @putil.timing
    def send_request(self,doc_result:dict ,documentId:str ,doc_invNumber = None,stp = None,timeout: int = 30) -> Dict[str, Union[int, str]]:
        try:
            print("inside send request DISCR")
            agent = self.UI_AGENT_SERVER
            RP = self.RELATIVE_PATH
            URL = os.path.join(agent+RP)
            data = self.payload()
            if doc_invNumber!= None:
                data["invNumber"] = doc_invNumber
            else:
                data["invNumber"] = ""
            data["blobname"] = self.discr_file
            data["linked_document"]["vendorGSTIN"] = doc_result.get("vendorGSTIN").get("text")
            data["linked_document"]["shippingGSTIN"] = doc_result.get("shippingGSTIN").get("text")
            data["linked_document"]["billingGSTIN"] = doc_result.get("billingGSTIN").get("text")
            data["linked_document"]["invoiceNumber"] = doc_result.get("invoiceNumber").get("text")
            tax_slabs = {}
            # tax_slabs["subtotal_0"] = doc_result.get("subTotal_0%").get("text")
            tax_slabs["subTotal_5"] = doc_result.get("subTotal_5%").get("text")
            tax_slabs["subTotal_12"] = doc_result.get("subTotal_12%").get("text")
            tax_slabs["subTotal_18"] = doc_result.get("subTotal_18%").get("text")
            tax_slabs["subTotal_28"] = doc_result.get("subTotal_28%").get("text")
            data["linked_document"]["slabs"] = tax_slabs
            if (float(doc_result.get("CGSTAmount_2.5%").get("text")) > 0) or (float(doc_result.get("CGSTAmount_6%").get("text")) > 0) or (float(doc_result.get("CGSTAmount_9%").get("text")) > 0) or (float(doc_result.get("CGSTAmount_14%").get("text")) > 0):
                data["linked_document"]["cgstpresent"] = 1
            elif (float(doc_result.get("IGSTAmount_5%").get("text")) > 0) or (float(doc_result.get("IGSTAmount_12%").get("text")) > 0) or (float(doc_result.get("IGSTAmount_18%").get("text")) > 0) or (float(doc_result.get("IGSTAmount_28%").get("text")) > 0):
                data["linked_document"]["cgstpresent"] = 0
            else:
                data["linked_document"]["cgstpresent"] = -1
            data["linked_document"]["documentId"] = documentId
            data["linked_document"]["original_invoice_stp"] = stp
            print("Data post to agent",type(data),data)
            #print("Data post to agent2",type(json.dumps(data,default = "str")),json.dumps(data,default=str))            
            data = json.dumps(data)
            print("Data post to agent3",type(data),data)
            
            # response = requests.post(URL, data, timeout=timeout)
            # response.raise_for_status()  # Raise exception for non-successful response
            # print(f"Request to {URL} succeeded with response code {response.status_code}")
            # result = json.loads(response.text)
            
            max_retries = 3
            for count_retry in range(1, max_retries + 1):
                try:
                    print(f"Trying discr note for {count_retry} time")
                    response = requests.post(URL, data, timeout=timeout)
                    response.raise_for_status()  # Raise exception for non-successful response
                    
                    result = json.loads(response.text)
                    if result["status_code"] == 200:
                        print(f"Request to {URL} succeeded with response code {result['status_code']}")
                        break  # Break out of the loop if the request is successful
                    else:
                        print(f"Request to {URL} failed with response code {result['status_code']}")
                        print("Waiting for 10 secs before retrying discr note")
                        time.sleep(10)
                        
                except requests.exceptions.RequestException as e:
                    print(f"Request failed on attempt {count_retry} with error: {e}")
            
            return {"status_code": response.status_code,"message": "Request succeeded","result":result}
        except requests.exceptions.Timeout as e:
            print(f"Request to {URL} timed out after {timeout} seconds with error: {e}")
            return {"status_code": 408, "message": str(e)}
        except requests.exceptions.RequestException as e:
            print(f"Request to {URL} failed with error: {e}")
            return {"status_code": 500, "message": str(e)}

import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)
@putil.timing

#check blank discrepancy and reject it 11 Nov 23
def Check_for_blank_page_if_no_docType(df,pages):

    try:

        blank_discp=False
        blank=[]
        
        for page in pages:
            df_page = df[df["page_num"]==page]
            concatenated_text = df_page['text'].str.cat(sep=' ')
            num_amounts = df_page["is_amount"].sum()
            
            if ("NO DISCREPANCY" in concatenated_text) and num_amounts<=2:
                print("blank disc not present")
                blank.append(True)
            else:
                print("actual disc note present")
                blank.append(False)
            

        if any(blank)==False:
            blank_discp = False
            return blank_discp 
        else:
            blank_discp=True
            return blank_discp


                
        
        
    except:
        return False
            

def process_discr_note(df:pd.DataFrame,docMetaData:dict)->pd.DataFrame:
    try:
        discr_note = None
        if docMetaData:
            if docMetaData.get("result").get("document").get("docType") == "Discrepancy Note":
                print(f"DocType is already Discrepancy Note")
                
                return df,discr_note,docMetaData,False
            if not (docMetaData.get("result").get("document").get("blobname")):
                try:
                    file_name = (docMetaData.get("result").get("document").get("documentId"))+".pdf"
                    blob_downloads = [join_path(SUBSCRIBER_ID, file_name)]
                    local_downloads = [join_path(UI_ROOT_FOLDER, UI_UPLOAD_FOLDER,os.path.basename(file_name))]
                    print("download pdf file from blob : ", blob_downloads, local_downloads)
                    downloadUrls = zip(blob_downloads,local_downloads)
                    FileDownloadStatus = downloadFilesFromBlob(downloadUrls)
                    if FileDownloadStatus == True:
                        docMetaData["result"]["document"]["blobname"] = local_downloads[0]
                    else:
                        print("download pdf from blob failed !")
                except Exception as E:
                    print("download pdf from blob exception",E)
                    raise Exception
            else :
                if not (os.path.exists(docMetaData.get("result").get("document").get("blobname"))):
                    try:
                        file_name = (docMetaData.get("result").get("document").get("documentId"))+".pdf"
                        blob_downloads = [join_path(SUBSCRIBER_ID, file_name)]
                        #print("Folder's path:", UI_ROOT_FOLDER, UI_UPLOAD_FOLDER, file_name)
                        local_downloads = [join_path(UI_ROOT_FOLDER, UI_UPLOAD_FOLDER,os.path.basename(file_name))]
                        print("download pdf file from blob : ", blob_downloads, local_downloads)
                        downloadUrls = zip(blob_downloads,local_downloads)
                        FileDownloadStatus = downloadFilesFromBlob(downloadUrls)
                        if FileDownloadStatus == True:
                            docMetaData["result"]["document"]["blobname"] = local_downloads[0]
                        else:
                            print("download pdf from blob failed !")
                    except Exception as E:
                        print("download pdf from blob exception",E)   
                        raise Exception 
            discr_pdf = docMetaData.get("result").get("document").get("blobname")
            print("Local Discr_pdf path", discr_pdf)
        else:
            return df,discr_note,docMetaData,False
        discrepanancy_note = ProcessDsicrepancyNote()
        page_list = discrepanancy_note.get_discrepancy_pages_from_df(df)
        print("Discrepancy pages :",len(page_list))
        if len(page_list)>0:
            df1 = df[df["page_num"].isin(page_list)]
            docMetaData["result"]["document"]["Invoice_Discr_pages"] = page_list
            disc_pred_file_name = (docMetaData.get("result").get("document").get("documentId"))+"_DISCR_pred.csv"
            disc_pred_file_downloads = join_path(UI_ROOT_FOLDER, UI_UPLOAD_FOLDER,os.path.basename(disc_pred_file_name))
            blank_disc = Check_for_blank_page_if_no_docType(df1,page_list)
            

            print("df share befor removing discr pages :",df.shape)
            df = discrepanancy_note.remove_discrepancy_pages_from_df(df,page_list)
            print("df share after removing discr pages :",df.shape)
            if blank_disc:

            #add flag in metadata and link to proper rejected msg , the msg has to be system rejected
                return df, None, docMetaData,True
            df1.to_csv(disc_pred_file_downloads, index=False)
            success, blob_paths = uploadFilesToBlobStore([disc_pred_file_downloads])
            if success:
                print(f"Upload successful. Blob paths: {blob_paths}")
                # Delete the CSV file after upload
                os.remove(disc_pred_file_downloads)
                print(f"File {disc_pred_file_downloads} deleted successfully.")
            else:
                print("Upload failed.")
            discr_note = discrepanancy_note.get_discrepancy_note(pdf_file_path=discr_pdf,pages=page_list)
            print("discr_note file :",discr_note)
            # Need to be called after extraction fo invoice, so removed form here and added in tapp_client
            """if discr_note:
                upload = ingest_discreopancy_note(discr_file=discr_note,meta_data=docMetaData)
                upload_status = upload.send_request()
                print("upload_status :",upload_status)"""
        return df,discr_note,docMetaData,False
    except Exception as E:
        print("process_discr_note exception",E)
        print(traceback.print_exc())
        return df,discr_note,docMetaData,False
if __name__ == "__main__":
    dirPath = "/Volumes/Macintosh HD - Data/workspace/Swiggy_Data/pred_files/"
    f7 = "382ed378-c7f0-11ed-a139-8fff026529aa_pred.csv"
    filePath = os.path.join(dirPath,f7)
    df =  pd.read_csv(filePath)
    docMetaData = {'id':'api.document.get','ver':'1.0','ts':1679407517945,'params':{'resmsgid':'','msgid':'','status':'Success','err':'','errmsg':'','reason':''},'responseCode':'OK','result':{'document':{'uploadUrl':'/import/HR2214017594.pdf','fileName':'HR2214017594.pdf','mimeType':'application/pdf','documentId':'382ed378-c7f0-11ed-a139-8fff026529aa','documentType':'Invoice','status':'EXTRACTION_INPROGRESS','submittedBy':'system','api_name':'upload','attachment_no':1,'message_id':'183effb876121dc9','record_id':'183effb876121dc9-1','size':116538,'userId':'123','submittedOn':1679407009654,'ace':2,'bar_qr_data':{'0':[],'1':[],'2':[],'3':[],'4':[]},'lastProcessedOn':1679407009,'lastUpdatedBy':'system','lastUpdatedOn':1679407032634,'name':'382ed378-c7f0-11ed-a139-8fff026529aa','pageCount':5,'stage':'EXTRACTION','statusMsg':'Extraction process initiated'}}}
    df =process_discr_note(df,docMetaData)
    print("process_discr_note :",df.shape)
    