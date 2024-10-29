# -*- coding: utf-8 -*-

import requests
import json
from sys import argv
# import config as cfg
import TAPPconfig as cfg
import sys
import os
import shutil
import zipfile
import base64
import traceback

import pytz
import pandas as pd
import time
import datetime

from dateutil import tz

import smtplib
import tempfile
from email.mime.text import MIMEText
from email import encoders
from email.message import Message
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart  

try:
    from generate_summary_report_plots import generate_report
    from new_vendor_identification import identify_new_vendors
    from generate_client_report import generate_client_report
except Exception as e:
    print(e)
    pass

tz_ist = pytz.timezone('Asia/Kolkata')
tz_local = pytz.timezone('Asia/Kolkata')
tz_NY = pytz.timezone('America/New_York')

statusRevComp = cfg.getStatusReviewCompleted()
statusProcessed = cfg.getStatusProcessed()
statusReview = cfg.getStatusReview()
statusFailed = cfg.getStatusFailed()
statusNew = "NEW"
statusDeleted = "DELETED"

# non_sla_vendor_file_path = "/mnt/swginstaproddrive/prod_correction_report/Non_SLA_Vendors.xlsx"

# NON_SLA_VENDORS = pd.read_excel(non_sla_vendor_file_path, dtype=object)
# NON_SLA_VENDORS["SLA Start Date"] = pd.to_datetime(NON_SLA_VENDORS["SLA Start Date"], format='%d/%m/%Y')
# dict_non_sla_vendors = dict(zip(NON_SLA_VENDORS["Vendor GSTIN"], NON_SLA_VENDORS["SLA Start Date"]))
# print(dict_non_sla_vendors)


UI_SERVER = cfg.getUIServer()
UI_SERVER = "http://106.51.73.100:9595"
UI_SERVER = "http://20.163.100.254"
UI_SERVER = "http://106.51.73.100:9012"
UI_SERVER = "http://20.219.95.182"
# UI_SERVER = "http://20.235.114.214"
# UI_SERVER = "http://52.172.153.247:7777"
# # UI_SERVER = "https://www.scootsypaiges.com"

GET_DOCUMENT_RESULT = cfg.getDocoumentResult()
FIND_DOCUMENT = cfg.getDocumentFind()

# dict_GSTIN = {'37': ('Andhra Pradesh', 'AD'),
#  '12': ('Arunachal Pradesh', 'AR'),
#  '18': ('Assam', 'AS'),
#  '10': ('Bihar', 'BR'),
#  '22': ('Chattisgarh', 'CG'),
#  '07': ('Delhi', 'DL'),
#  '30': ('Goa', 'GA'),
#  '24': ('Gujarat', 'GJ'),
#  '06': ('Haryana', 'HR'),
#  '02': ('Himachal Pradesh', 'HP'),
#  '01': ('Jammu and Kashmir', 'JK'),
#  '20': ('Jharkhand', 'JH'),
#  '29': ('Karnataka', 'KA'),
#  '32': ('Kerala', 'KL'),
#  '31': ('Lakshadweep Islands', 'LD'),
#  '23': ('Madhya Pradesh', 'MP'),
#  '27': ('Maharashtra', 'MH'),
#  '14': ('Manipur', 'MN'),
#  '17': ('Meghalaya', 'ML'),
#  '15': ('Mizoram', 'MZ'),
#  '13': ('Nagaland', 'NL'),
#  '21': ('Odisha', 'OD'),
#  '34': ('Pondicherry', 'PY'),
#  '03': ('Punjab', 'PB'),
#  '08': ('Rajasthan', 'RJ'),
#  '11': ('Sikkim', 'SK'),
#  '33': ('Tamil Nadu', 'TN'),
#  '36': ('Telangana', 'TS'),
#  '16': ('Tripura', 'TR'),
#  '09': ('Uttar Pradesh', 'UP'),
#  '05': ('Uttarakhand', 'UK'),
#  '19': ('West Bengal', 'WB'),
#  '35': ('Andaman and Nicobar Islands', 'AN'),
#  '04': ('Chandigarh', 'CH'),
#  '26': ('Dadra & Nagar Haveli and Daman & Diu', 'DNHDD'),
#  '38': ('Ladakh', 'LA'),
#  '97': ('Other Territory', 'OT')}

# state_SLA = {"Maharashtra": datetime.datetime(2022, 9, 6),
# "Karnataka": datetime.datetime(2022, 9, 15)}

# fieldMapping = {}
# with open(r"fieldMapping.json","r") as f:
#     fieldMapping = json.load(f)

# list_fields = list(fieldMapping.keys())
count_header_fields = 20

def base64_encode(string :str):
    try:
        string = string.encode('ascii')
        string = base64.b64encode(string)
        string = string.decode('ascii')
        return string
    except:
        print("base64 encoding error",traceback.print_exc())
        return None

def base64_decode(string:str)->str:
    try:
        string = string.encode('ascii')
        string = base64.b64decode(string)
        string = string.decode('ascii')
        return string
    except:
        print("base64 decoding error",traceback.print_exc())
        return None

def initiate_payload_find(fetch_date):
    """
    """
    start_time = datetime.datetime(fetch_date.year, fetch_date.month, fetch_date.day)
    start_time = tz_local.localize(start_time)
    print("Start time:", start_time)
    end_time = start_time + datetime.timedelta(days=1)
    print("End Time:", end_time)

    start_time_epoch = (start_time - datetime.datetime(1970, 1, 1, tzinfo=pytz.utc)).total_seconds()*1000
    end_time_epoch = (end_time - datetime.datetime(1970, 1, 1, tzinfo=pytz.utc)).total_seconds()*1000

    print(start_time_epoch, end_time_epoch)
    
    payload = {
                "ver": "1.0",
                "params": {"msgid": ""},
                "request": {
                    "filter": {
                        "submittedOn": {
                            ">=": start_time_epoch,
                            "<": end_time_epoch
                        }
                    },
                "limit": 999999
                }
            }
    return payload


# def find_false_positives(headerItems, l_corrections):
#     """
#     Check if updated value equals any of the other fields in extracted value.
#     if yes we have a false positive. FP’ =Y
#     """
#     l_fp = []
#     # print("Corrections:", l_corrections)
#     # print(headerItems)
#     # For each extracted field, save fieldValue and correctdValue in the dictionary with fieldId as the key
#     dict_values = {}
#     for l in headerItems:
#         temp_dict = {}
#         if "fieldValue" in l:
#             temp_dict["fieldValue"] = l["fieldValue"]
#         if "correctedValue" in l:
#             temp_dict["correctedValue"] = l["correctedValue"]
#         dict_values[l["fieldId"]] = temp_dict

#     # print(dict_values)

#     # Check whether for the corrections, the correctedValue is equal to fieldValue for any other field
#     for c in l_corrections:
#         if c in dict_values:
#             if "fieldValue" in dict_values[c]:
#                 extracted_value = dict_values[c]["fieldValue"]
#                 for key, value in dict_values.items():
#                     if (key != c) & (str(value["fieldValue"]).strip() == str(extracted_value).strip()):
#                         # False Positive found
#                         # print(key + "extracted as" + c)
#                         if str(extracted_value).strip() != "0":
#                             l_fp.append(c)

#     # print(l_fp)
#     return l_fp


def convert_epoch_ist(epoch_time):
    """
    """
    epoch_time = str(epoch_time)
    if len(epoch_time) > 10:
        epoch_time = float(epoch_time)/1000

    epoch_time = float(epoch_time)
    time_ist = datetime.datetime.fromtimestamp(epoch_time, tz_ist).strftime("%d/%m/%Y %H:%M:%S")

    return time_ist


def remove_millisecond_time(epoch_time):
    """
    """
    epoch_time = str(epoch_time)
    if len(epoch_time) > 10:
        epoch_time = float(epoch_time)/1000

    epoch_time = float(epoch_time)

    return epoch_time


def convert_epoch_local_time(epoch_time):
    """
    """
    epoch_time = str(epoch_time)
    if len(epoch_time) > 10:
        epoch_time = float(epoch_time)/1000

    epoch_time = float(epoch_time)
    time_ist = datetime.datetime.fromtimestamp(epoch_time, tz_local).strftime("%d/%m/%Y %H:%M:%S")

    return time_ist


def fetch_corrections(fetch_date):
    """
    """
    find_url = UI_SERVER + FIND_DOCUMENT
    doc_result_get_url = UI_SERVER + GET_DOCUMENT_RESULT

    headers = {}    
    headers["Content-Type"] = "application/json"

    payload = initiate_payload_find(fetch_date)
    data = json.dumps(payload)

    print("Calling Find API:",find_url)
    print("Payload:", data)

    try:
        response =  requests.post(find_url, headers=headers, data = data)
    except Exception as e:
        print("Error in document_get API",traceback.print_exc())
        return pd.DataFrame()


    response = response.json()

    list_df = []

    if response["responseCode"] == "OK":
        # print("Response Success")
        results = response["result"]
        total_res_count = results["count"]
        documents = results["documents"]
        print("TOTAL DOCUMENT COUNT:", total_res_count)

        for doc in documents:
            print("Processing:", doc["documentId"])            
            # doc_id = doc
            # status = statusRevComp
            # print(doc)
            file_name = ""
            if "fileName" in doc:
                file_name = doc["fileName"]

            doc_id = doc["documentId"]
            status = doc["status"]
            
            dict_row = {}
            dict_row["File Name"] = file_name
            dict_row["Document ID"] = doc_id
            dict_row["Status"] = status
            # print(dict_row)
            dict_row["Submitted On"] = ""
            dict_row["Extraction Completed On"] = ""
            dict_row["Extraction Time (secs)"] = ""
            dict_row["Pages"] = ""
            dict_row["Quality Score"] = ""
            dict_row["Document Type"] = ""
            dict_row["VENDOR NAME"] = ""
            dict_row["VENDOR GSTIN"] = ""
            # dict_row["VENDOR State"] = ""
            dict_row["Billing GSTIN"] = ""
            dict_row["Billing Name"] = ""
            dict_row["Shipping GSTIN"] = ""
            dict_row["Shipping Name"] = ""
            dict_row["User"] = ""
            dict_row["Error Message"] = ""
            dict_row["STP System"] = ""
            dict_row["ACE"] = ""
            dict_row["User Comment"] = ""
            dict_row["Total Review Time"] = ""
            dict_row["Review Completion/Deletion Time"] = ""
            dict_row["Delete Reason"] = ""
            dict_row["Manually Reviewed"] = "Not Applicable"
            dict_row["Reassign Reason"] = ""
            dict_row["Status Msg"] = ""
            dict_row["Disc Note AutoDeleted"] = ""
            # dict_row["Incorrect FP"] = 0


            if "comment" in doc:
                dict_row["User Comment"] = doc["comment"]

            if "submittedOn" in doc:
                dict_row["Submitted On"] = convert_epoch_local_time(doc["submittedOn"])

            if "extractionCompletedOn" in doc:
                dict_row["Extraction Completed On"] = convert_epoch_local_time(doc["extractionCompletedOn"])

            if ("submittedOn" in doc) & ("extractionCompletedOn" in doc):
                extraction_time = (remove_millisecond_time(doc["extractionCompletedOn"]) - remove_millisecond_time(doc["submittedOn"]))
                dict_row["Extraction Time (secs)"] = extraction_time

            if "stp" in doc:
                dict_row["STP System"] = str(doc["stp"])

            if "pageCount" in doc:
                dict_row["Pages"] = doc["pageCount"]

            if "qualityScore" in doc:
                dict_row["Quality Score"] = doc["qualityScore"]

            if "docType" in doc:
                dict_row["Document Type"] = doc["docType"]

            if "ace" in doc:
                if doc["ace"] == 0:
                    dict_row["ACE"] = "NO"
                if doc["ace"] == 1:
                    dict_row["ACE"] = "YES"   
                if doc["ace"] == 2:
                    dict_row["ACE"] = "Not Applicable"             

            if "reassignReason" in doc:
                dict_row["Reassign Reason"] = doc["reassignReason"]

            if "statusMsg" in doc:
                dict_row["Status Msg"] = doc["statusMsg"]

            if "discNoteAutoDeleted" in doc:
                dict_row["Disc Note AutoDeleted"] = 1
            # Get document Results
            url = UI_SERVER + GET_DOCUMENT_RESULT + str(doc_id)
            print("Fetching doc result for:", url)
            resp = requests.get(url, headers= headers)

            
            try:
                resp = resp.json()
            except Exception as e:
                print(e)
                resp = {}
                resp["responseCode"] = "NOT OK"
                pass
            
            # print(resp)
            if resp["responseCode"] == "OK":
                if ((status == statusRevComp) | ("RPA" in status)) & ("documentInfo" in resp["result"]["document"]):
                    print("Inside:", status)
                    # Review Completed Documents
                    # Change Code for BCP
                    headerItems = resp["result"]["document"]["documentInfo"]
                    dict_row["Total Review Time"] = ""
                    if "reviewedBy" in doc:
                        dict_row["User"] = doc["reviewedBy"]
                        dict_row["Manually Reviewed"] = "Yes"
                    else:
                        dict_row["User"] = "DUMMY_USER"
                        dict_row["Manually Reviewed"] = "No"

#                    if "totalReviewedTime" in doc:
                    if ("totalReviewedTime" in doc) and (doc["totalReviewedTime"] is not None):                                                
                        dict_row["Total Review Time"] = doc["totalReviewedTime"]/60

                    if "reviewedAt" in doc:
                        dict_row["Review Completion/Deletion Time"] = convert_epoch_local_time(doc["reviewedAt"])

                    # dict_row = {**dict_row, **dict.fromkeys(list_fields, "")}
                    # print(headerItems)
                    # Call method to extract False Positives
                    # print(headerItems)
                    dict_count = {"Missed": [], "Incorrect": []}
                    for l in headerItems:
                        # Code to add vendorName
                        if l["fieldId"] == "vendorName":
                            if "correctedValue" in l:
                                dict_row["VENDOR NAME"] = l["correctedValue"]
                            else:
                                dict_row["VENDOR NAME"] = l["fieldValue"]
                        if l["fieldId"] == "vendorGSTIN":
                            if "correctedValue" in l:
                                dict_row["VENDOR GSTIN"] = l["correctedValue"]
                            else:
                                dict_row["VENDOR GSTIN"] = l["fieldValue"]
                        if l["fieldId"] == "billingGSTIN":
                            if "correctedValue" in l:
                                dict_row["Billing GSTIN"] = l["correctedValue"]
                            else:
                                dict_row["Billing GSTIN"] = l["fieldValue"]
                        if l["fieldId"] == "billingName":
                            if "correctedValue" in l:
                                dict_row["Billing Name"] = l["correctedValue"]
                            else:
                                dict_row["Billing Name"] = l["fieldValue"]
                        if l["fieldId"] == "shippingGSTIN":
                            if "correctedValue" in l:
                                dict_row["Shipping GSTIN"] = l["correctedValue"]
                            else:
                                dict_row["Shipping GSTIN"] = l["fieldValue"]
                        if l["fieldId"] == "shippingName":
                            if "correctedValue" in l:
                                dict_row["Shipping Name"] = l["correctedValue"]
                            else:
                                dict_row["Shipping Name"] = l["fieldValue"]
                        if "correctedValue" in l:
                            print(l)
                            # if (float(l["confidence"]) == 0) & (str(l["correctedValue"]).strip() != ""):
                            if (str(l["fieldValue"]).strip() == "") & (str(l["correctedValue"]).strip() != ""):
                                # Inside No Extraction Case: M
                                # print("Inside No Extraction Case")
                                dict_row[l["fieldId"]] = "Missed"
                                # Code to add list of Missed and Incorrect Extracted Values
                                dict_count["Missed"].append(l["fieldId"])
                            elif (str(l["fieldValue"]).strip() != "") & (str(l["correctedValue"]).strip() != ""):
                                # Inside Wrong Extraction Case: X
                                # print("Inside Wrong Extraction Case")
                                # Code added fotr 0 removal in total post decimal points
                                # For total fields, compare the extracted and corrected value, if they are same, no correction
                                # Example: 3337.00 corrected as 3337 means no correction
                                if (str(l["fieldValue"]).strip() != str(l["correctedValue"]).strip()):
                                    dict_row[l["fieldId"]] = "Incorrect"
                                    # Code to add list of Missed and Incorrect Extracted Values
                                    dict_count["Incorrect"].append(l["fieldId"])
                                else:
                                    dict_row[l["fieldId"]] = "OK"
                            elif (str(l["fieldValue"]).strip() != "") & (str(l["correctedValue"]).strip() == ""):
                                dict_row[l["fieldId"]] = "Incorrect"
                                # Code to add list of Missed and Incorrect Extracted Values
                                dict_count["Incorrect"].append(l["fieldId"])
                        else:
                            # Inside No Correction Case: OK
                            # Inside Field not present Case: ""
                            dict_row[l["fieldId"]] = "OK"

                        # Code added to include extraction confidence in Correction Report
                        if "confidence" in l:
                            dict_row[l["fieldId"] + "_confidence"] = l["confidence"]

                    # Code to add list of Missed and Incorrect Extracted Values
                    dict_row = {**dict_row, **dict_count}

                    # l_corrections.append("CGSTAmount")
                    # l_fp = list(set(find_false_positives(headerItems, l_corrections)))
                    # dict_row["False Positives"] = l_fp
                    # dict_row["Incorrect FP"] = len(l_fp)
                    # dict_row["SLA_flag"], dict_row["Non_SLA_reason"] = populate_SLA_flag_GSTIN(headerItems)

                    # Code to calculate LineItem Accuracy
                    if "documentLineItems" in resp["result"]["document"]:
                        # LineItem exists
                        line_items = resp["result"]["document"]["documentLineItems"]
                        dict_row = calculate_table_accuracy(dict_row, line_items)
                elif status == statusDeleted:
                    if "deleteReason" in doc:
                        dict_row["Delete Reason"] = doc["deleteReason"]
                    if "deletedBy" in doc:
                        dict_row["User"] = doc["deletedBy"]
                        dict_row["Manually Reviewed"] = "Yes"
                    else:
                        dict_row["User"] = "DUMMY_USER"
                        dict_row["Manually Reviewed"] = "No"
                    if "deleteTime" in doc:
                        dict_row["Review Completion/Deletion Time"] = convert_epoch_local_time(doc["deleteTime"])

#                    if "totalReviewedTime" in doc:
                    if ("totalReviewedTime" in doc) and (doc["totalReviewedTime"] is not None):                        
                        dict_row["Total Review Time"] = doc["totalReviewedTime"]/60

                    if "result" in resp:
                        if "document" in resp["result"]:
                            if "documentInfo" in resp["result"]["document"]:
                                headerItems = resp["result"]["document"]["documentInfo"]
                                for l in headerItems:
                                    if l["fieldId"] == "vendorName":
                                        if "correctedValue" in l:
                                            dict_row["VENDOR NAME"] = l["correctedValue"]
                                        else:
                                            dict_row["VENDOR NAME"] = l["fieldValue"]
                                    # if l["fieldId"] == "billingGSTIN":
                                    #     if "correctedValue" in l:
                                    #         dict_row["Billing GSTIN"] = l["correctedValue"]
                                    #     else:
                                    #         dict_row["Billing GSTIN"] = l["fieldValue"]
                    # dict_row["SLA_flag"], dict_row["Non_SLA_reason"] = 0, ["Document Rejected"]
                elif status == statusReview:
                    headerItems = resp["result"]["document"]["documentInfo"]
                    for l in headerItems:
                        # Code to add vendorName
                        if l["fieldId"] == "vendorName":
                            if "correctedValue" in l:
                                dict_row["VENDOR NAME"] = l["correctedValue"]
                            else:
                                dict_row["VENDOR NAME"] = l["fieldValue"]
                        # if l["fieldId"] == "vendorGSTIN":
                        #     if "correctedValue" in l:
                        #         dict_row["VENDOR GSTIN"] = l["correctedValue"]
                        #     else:
                        #         dict_row["VENDOR GSTIN"] = l["fieldValue"]
                        # if l["fieldId"] == "billingGSTIN":
                        #     if "correctedValue" in l:
                        #         dict_row["Billing GSTIN"] = l["correctedValue"]
                        #     else:
                        #         dict_row["Billing GSTIN"] = l["fieldValue"]
                        # if l["fieldId"] == "shippingGSTIN":
                        #     if "correctedValue" in l:
                        #         dict_row["Shipping GSTIN"] = l["correctedValue"]
                        #     else:
                        #         dict_row["Shipping GSTIN"] = l["fieldValue"]
                    # dict_row["SLA_flag"], dict_row["Non_SLA_reason"] = populate_SLA_flag_GSTIN(headerItems)
                # else:
                #     # Other Status
                #     dict_row["SLA_flag"], dict_row["Non_SLA_reason"] = 1, []
            else:
                print("Inside FAILED")
                dict_row["Error Message"] = "Error in fetching Document Results"
                # print(dict_row)
                # dict_row["SLA_flag"], dict_row["Non_SLA_reason"] = 1, []

            # # Code to extract Vendor State
            # if "VENDOR GSTIN" in dict_row:
            #     if dict_row["VENDOR GSTIN"] != "":
            #         if len(str(dict_row["VENDOR GSTIN"])) >= 2:
            #             gstin_v = dict_row["VENDOR GSTIN"]
            #             if gstin_v[0:2] in dict_GSTIN:
            #                 dict_row["VENDOR State"] = dict_GSTIN[gstin_v[0:2]][0]

            # # Code to extract Billing State
            # if "Billing GSTIN" in dict_row:
            #     if dict_row["Billing GSTIN"] != "":
            #         if len(str(dict_row["Billing GSTIN"])) >= 2:
            #             gstin_b = dict_row["Billing GSTIN"]
            #             if gstin_b[0:2] in dict_GSTIN:
            #                 dict_row["Billing State"] = dict_GSTIN[gstin_b[0:2]][0]

            # state_based_SLA_Flag(dict_row)

            list_df.append(dict_row)
    else:
        print("Error in find API!!!")

    # print(list_df)
    DF = pd.DataFrame(list_df)
    return DF


def calculate_table_accuracy(dict_row, table_items):
    """
    """
    # doc_1675421511754_98baababbaa
    total_extracted_items = len(table_items)
    count_deleted_items = 0
    count_extracted_fields = {}
    count_corrected_fields = {}
    for item in table_items:
        if "isDeleted" in item:
            # Don't count deleted item to calculate overall accuracy
            count_deleted_items += 1
            continue
        if "fieldset" in item:
            field_set = item["fieldset"]
            for field in field_set:
                f = field["fieldId"]
                if f in count_extracted_fields:
                    count_extracted_fields[f] = count_extracted_fields[f] + 1
                else:
                    count_extracted_fields[f] = 1
                if "correctedValue" in field:
                    if f in count_corrected_fields:
                        count_corrected_fields[f] = count_corrected_fields[f] + 1
                    else:
                        count_corrected_fields[f] = 1

    print(total_extracted_items, count_deleted_items, count_extracted_fields, count_corrected_fields)

    table_accuracy = None
    if len(count_extracted_fields) > 0:
        extracted_fields = sum(count_extracted_fields.values())
        corrected_fields = sum(count_corrected_fields.values()) 
        table_accuracy = ((extracted_fields - corrected_fields)/extracted_fields)*100

    count_extracted_fields = {"LI_" + str(k) + '_extracted': v for k, v in count_extracted_fields.items()}
    count_corrected_fields = {"LI_" + str(k) + '_corrected': v for k, v in count_corrected_fields.items()}

    dict_row = {**dict_row, **count_extracted_fields}
    dict_row = {**dict_row, **count_corrected_fields}
    dict_row["LI_Accuracy"] = table_accuracy

    dict_row["LI_extracted_count"] = total_extracted_items
    dict_row["LI_deleted_count"] = count_deleted_items
    
    return dict_row


def rearrange_columns(DF):
    """
    """
    # Rearrange Columns
    cols_at_end = []

    cols = list(DF.columns)
    for col in cols:
        if ("LI_" in col) & ("Accuracy" not in col):
            cols_at_end.append(col)

    cols_at_end.extend(["Correct Extraction", "Not Extracted", "Incorrect Extraction"])

    if "LI_deleted_count" in cols_at_end:
        cols_at_end.remove("LI_deleted_count")
        cols_at_end.append("LI_deleted_count")

    if "LI_extracted_count" in cols_at_end:
        cols_at_end.remove("LI_extracted_count")
        cols_at_end.append("LI_extracted_count")

    for col in cols:
        if "Accuracy" in col:
            cols_at_end.append(col)

    print(cols_at_end)

    DF = DF[[c for c in DF if c not in cols_at_end] + [c for c in cols_at_end if c in DF]]

    print(DF.columns)

    return DF


def calculate_accuracy(DF):
    """
    """
    DF["Correct Extraction"] = (DF == 'OK').T.sum()
    DF["Not Extracted"] = (DF == 'Missed').T.sum()
    DF["Incorrect Extraction"] = (DF == 'Incorrect').T.sum()

    # DF["Accuracy SWIGGY"] = ((DF["Correct Extraction"] + DF["Not Present"])/count_header_fields)*100
    DF["Accuracy"] = (DF["Correct Extraction"]/(DF["Correct Extraction"] + DF["Not Extracted"] + DF["Incorrect Extraction"]))*100
    # DF["Probable STP"] = 0
    # DF.loc[(DF["Accuracy SWIGGY"] == 100), "Probable STP"] = 1

    # DF["Incorrect Extraction"] = DF["Incorrect Extraction"] - DF["Incorrect FP"]
    DF.loc[~((DF["Status"] == statusRevComp) | (DF["Status"].str.contains("RPA"))), 
    ["Correct Extraction", "Not Extracted", "Incorrect Extraction", "Accuracy"]] = ""

    DF = rearrange_columns(DF)

    return DF


# def state_based_SLA_Flag(dict_row):
#     """
#     """
#     # print(dict_row)
#     # State based SLA is valid only for REVIEW and REVIEW_COMPLETED documents
#     # DELETED, PURGED and FAILED should not be considered 
#     if dict_row["Status"] not in [statusReview, statusRevComp]:
#         return

#     if "Billing State" in dict_row:
#         state = dict_row["Billing State"]
#         if state in state_SLA:
#             submit_date = datetime.datetime.strptime(dict_row["Submitted On"], "%d/%m/%Y %H:%M:%S")
#             sla_start_date = state_SLA[state]
#             if submit_date < sla_start_date:
#                 dict_row["SLA_flag"] = 0
#                 dict_row["Non_SLA_reason"].append("State SLA Date not started")
#         else:
#             dict_row["SLA_flag"] = 0
#             dict_row["Non_SLA_reason"].append("Non-SLA State")
#     else:
#         dict_row["SLA_flag"] = 0
#         dict_row["Non_SLA_reason"].append("Non-SLA State")

#     # Vendor Specific SLA Flag 
#     if "VENDOR GSTIN" in dict_row:
#         vendor_gstin = dict_row["VENDOR GSTIN"]
#         if vendor_gstin in dict_non_sla_vendors:
#             sla_start_date_vendor = dict_non_sla_vendors[vendor_gstin]
#             submit_date = datetime.datetime.strptime(dict_row["Submitted On"], "%d/%m/%Y %H:%M:%S")
#             if submit_date < sla_start_date_vendor:
#                 dict_row["SLA_flag"] = 0
#                 dict_row["Non_SLA_reason"].append("non-SLA Vendor")

#     # Quality Score based SLA Flag
#     try:
#         if "Quality Score" in dict_row:
#             quality_score = dict_row["Quality Score"]
#             if quality_score < 0.85:
#                 dict_row["SLA_flag"] = 0
#                 dict_row["Non_SLA_reason"].append("Poor Quality Document")
#     except Exception as e:
#         pass



# def populate_SLA_flag_GSTIN(headerItems):
#     """
#     """
#     # print(headerItems)
#     sla_flag = 1
#     non_sla_reason = []
#     # Populate SLA Flag for vendorGSTIN, billingGSTIN, shippingGSTIN
#     for l in headerItems:
#         if l["fieldId"] in ["vendorGSTIN", "billingGSTIN", "shippingGSTIN"]:
#             if "entityMasterdata" in l:
#                 if l["entityMasterdata"] == 0:
#                     # Non-SLA case as entityMasterdata not present
#                     sla_flag = 0
#                     non_sla_reason.append(str(l["fieldId"]) + " not present in MasterData/Document")
#         elif l["fieldId"] in ["invoiceDate", "invoiceNumber"]:
#             if "isReferenceDataPresent" in l:
#                 if l["isReferenceDataPresent"] == 0:
#                     # Non-SLA case as RefernceData not present
#                     sla_flag = 0
#                     non_sla_reason.append(str("Refernce Data for " + l["fieldId"]) + " not present")

#     return (sla_flag, non_sla_reason)


def send_email(offset):
    """
    """
    print("Sending Email!!!!")
    zf = 'CORRECTION_REPORTS.zip'
    
    recipients = ["sahil.aggarwal@taoautomation.com", "narayana.n@taoautomation.com", "swaroop.samavedam@taoautomation.com",  "harika.reddy@taoautomation.com","rupesh.alluri@taoautomation.com", "amit.rajan@taoautomation.com", "hariharamoorthy.theriappan@taoautomation.com", "pradeepkumar.hk@taoautomation.com", "chaitra.shivanagoudar@taoautomation.com", "narendra.venkata@taoautomation.com", "divya.maggu@taoautomation.com","uma.maheshwari@taoautomation.com","pragati.sarode@taoautomation.com"]

    #recipients = ["sahil.aggarwal@taoautomation.com","chaitra.shivanagoudar@taoautomation.com","narendra.venkata@taoautomation.com"]
    #recipients = ["sahil.aggarwal@taoautomation.com"]

    sender = "cGFpZ2VzLmFkbWluQHRhb2F1dG9tYXRpb24uY29t"
    #sender = "cnVwZXNoLmFsbHVyaUB0YW9hdXRvbWF0aW9uLmNvbQ=="

    email_subject = 'Correction Report | Date: ' 
    today = datetime.datetime.now().date()
    from_date = str(today - datetime.timedelta(days=offset[0]))
    email_subject = email_subject + from_date

    to_date = ""
    if len(offset) > 1:
        to_date = str(today - datetime.timedelta(days=offset[-1]))
        email_subject = email_subject + " to " + to_date

    email_subject = email_subject + " | ENV: " + str(UI_SERVER)

    themsg = MIMEMultipart()
    themsg['Subject'] = email_subject
    themsg['To'] = ', '.join(recipients)
    themsg['From'] = base64_decode(sender)

    email_body = "Hello,\n\nPlease find the attached Correction Report for:\nDate: " + from_date
    if to_date != "":
        email_body = email_body + " to " + to_date
    email_body = email_body + "\nENV: " + str(UI_SERVER)
    email_body = email_body + "\n\nThanks"
    body = MIMEText(email_body)
    
    themsg.attach(body)

    msg = MIMEBase('application', 'zip')
    msg.set_payload(open(zf, "rb").read())
    encoders.encode_base64(msg)
    msg.add_header('Content-Disposition', 'attachment', 
                   filename='CORRECTION_REPORTS.zip')
    themsg.attach(msg)
    themsg = themsg.as_string()

    # send the message
    SERVER = "smtp-mail.outlook.com"
    port=587
    #FROM = "cnVwZXNoLmFsbHVyaUB0YW9hdXRvbWF0aW9uLmNvbQ=="
    FROM = "cGFpZ2VzLmFkbWluQHRhb2F1dG9tYXRpb24uY29t"
    pwd = "WGRGZ0hqVXZfMDk="
    #pwd = "VGFvI0AxMjM="

    server = smtplib.SMTP(SERVER, port)
    server.connect(SERVER, port)
    server.ehlo()
    server.starttls()
    server.ehlo()

    server.login(base64_decode(FROM), base64_decode(pwd))
    server.sendmail(base64_decode(FROM), recipients, themsg)
    server.quit()

    # smtp = smtplib.SMTP()
    # smtp.connect()
    # smtp.sendmail(sender, recipients, themsg)
    # smtp.close()


def generate_sla_report_for_day(DF, fetch_date):
    """
    """
    print("Generating SLA report for current day:", fetch_date)
    cols_to_retain = ["File Name", "Document ID", "Status", "Submitted On", "VENDOR NAME", "VENDOR GSTIN",
    "VENDOR State", "Billing GSTIN", "Billing State", "Shipping GSTIN", "STP System", "Delete Reason",
    "SLA_flag", "Non_SLA_reason"]
    file_name = "Reports/SLA Report For Documents Uploaded on " + str(fetch_date) + ".csv"
    DF[cols_to_retain].to_csv(file_name, index=False)


def main(offset):
    """
    """
    print("Inside main method!!!!")
    print("Offset:", offset)
    # Clear folder and recreate it
    try:
        shutil.rmtree('./Reports')
        os.remove('CORRECTION_REPORTS.zip')
    except Exception as e:
        pass

    os.mkdir('./Reports')

    today = datetime.datetime.now().date()
    print("Current Date:", today)
    DF_COMBINED = pd.DataFrame()
    for off in offset:
        fetch_date = today - datetime.timedelta(days=off)
        print("Fetch Date:", fetch_date)
        DF = fetch_corrections(fetch_date)
        if DF.shape[0] > 0:
            DF = calculate_accuracy(DF)

        file_name = "Reports/Correction Report For Documents Uploaded on " + str(fetch_date) + ".csv"
        DF_COMBINED = pd.concat([DF_COMBINED, DF], ignore_index=True)
        DF_COMBINED = rearrange_columns(DF_COMBINED)
        DF.to_csv(file_name, index=False)
        # if off == 1:
        #     generate_sla_report_for_day(DF, fetch_date)
        #     try:
        #         identify_new_vendors(DF, fetch_date)
        #     except Exception as e:
        #         print(e)
        #         pass

    file_name_combined = "Reports/Combined Correction Report" + ".csv"
    DF_COMBINED.to_csv(file_name_combined, index=False)

    # Call code to generate summary report
    try:
        DF = pd.read_csv(file_name_combined)
        identify_new_vendors(DF, "COMBINED")
        generate_report(DF)
        strat_date_current_month = datetime.datetime(2023, 1, 6).date()
        generate_report(DF, start_date = strat_date_current_month)
        generate_client_report(DF, None)
    except Exception as e:
        print(e)
        print("Error in generate_report", traceback.print_exc())
        pass

    # Zip File
    zipped_file_name = "CORRECTION_REPORTS.zip"
    try:
        os.remove(zipped_file_name)
    except Exception as e:
        pass
    files = os.listdir('./Reports')
    ZipFile = zipfile.ZipFile(zipped_file_name, "a" )

    for a in files:
        ZipFile.write(os.path.join('./Reports', a), compress_type=zipfile.ZIP_DEFLATED)
    ZipFile.close()
    print("***************** Reports generated and files ready to be sent *****************")
    send_email(offset)


if __name__ == "__main__":
    offset = [0]
    if len(argv) > 1:
        if str(argv[1]) == "START":
            strat_date = datetime.datetime(2024, 2, 1).date()
            #strat_date = datetime.datetime(2023, 10, 1).date()
            today = datetime.datetime.now().date()
            difference = today - strat_date
            offset = list(range(0, difference.days+1))
            offset.reverse()
            print("Generating report from START")
            print("Offset:", offset)
        elif str(argv[1]) == "WEEK":
            offset = [6,5,4,3,2,1,0]
            offset = [11,10,9,8,7,6,5,4,3,2,1,0]

        else:
            offset = [int(argv[1])]

    if len(argv) > 2:
        UI_SERVER = str(argv[2])

    main(offset=offset)
