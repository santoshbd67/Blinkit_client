import pandas as pd
import os
from csv import DictWriter
import re
import json
import traceback
import operator
import requests
from TAPPconfig import getUIServer
import datetime
import pytz

from klein import Klein
app = Klein()

statusRevComp = "REVIEW_COMPLETED"
statusProcessed = "PROCESSED"
statusReview = "REVIEW"
statusFailed = "FAILED"
statusNew = "NEW"
statusDeleted = "DELETED"
statusPurged = "PURGED"

count_fields_client = 18

tz_ist = pytz.timezone('Asia/Kolkata')

dict_GSTIN = {'37': ('Andhra Pradesh', 'AD'),
 '12': ('Arunachal Pradesh', 'AR'),
 '18': ('Assam', 'AS'),
 '10': ('Bihar', 'BR'),
 '22': ('Chattisgarh', 'CG'),
 '07': ('Delhi', 'DL'),
 '30': ('Goa', 'GA'),
 '24': ('Gujarat', 'GJ'),
 '06': ('Haryana', 'HR'),
 '02': ('Himachal Pradesh', 'HP'),
 '01': ('Jammu and Kashmir', 'JK'),
 '20': ('Jharkhand', 'JH'),
 '29': ('Karnataka', 'KA'),
 '32': ('Kerala', 'KL'),
 '31': ('Lakshadweep Islands', 'LD'),
 '23': ('Madhya Pradesh', 'MP'),
 '27': ('Maharashtra', 'MH'),
 '14': ('Manipur', 'MN'),
 '17': ('Meghalaya', 'ML'),
 '15': ('Mizoram', 'MZ'),
 '13': ('Nagaland', 'NL'),
 '21': ('Odisha', 'OD'),
 '34': ('Pondicherry', 'PY'),
 '03': ('Punjab', 'PB'),
 '08': ('Rajasthan', 'RJ'),
 '11': ('Sikkim', 'SK'),
 '33': ('Tamil Nadu', 'TN'),
 '36': ('Telangana', 'TS'),
 '16': ('Tripura', 'TR'),
 '09': ('Uttar Pradesh', 'UP'),
 '05': ('Uttarakhand', 'UK'),
 '19': ('West Bengal', 'WB'),
 '35': ('Andaman and Nicobar Islands', 'AN'),
 '04': ('Chandigarh', 'CH'),
 '26': ('Dadra & Nagar Haveli and Daman & Diu', 'DNHDD'),
 '38': ('Ladakh', 'LA'),
 '97': ('Other Territory', 'OT')}

state_SLA = {"Maharashtra": datetime.datetime(2022, 9, 6),
"Karnataka": datetime.datetime(2022, 9, 15)}

call_back_url = getUIServer()
# call_back_url = "http://106.51.73.100:8888"
# call_back_url = "http://20.235.114.214"
# call_back_url = "http://52.172.153.247:9191"
# call_back_url = "http://13.71.23.200:9999"
# call_back_url = "https://taotapp-bcp.com"

def form_call_back_url_doc_result_get():
    """
    :return:
    """
    endpoint = "document/result/get"
    url = call_back_url + "/" + endpoint + "/"
    return url


def form_call_back_url_doc_get():
    """
    :return:
    """
    endpoint = "document/get"
    url = call_back_url + "/" + endpoint + "/"
    return url


def populate_SLA_flag_GSTIN(headerItems):
    """
    Vendor Specific Flag
    """
    # print(headerItems)
    sla_flag = 1
    non_sla_reason = []
    # Populate SLA Flag for vendorGSTIN, billingGSTIN, shippingGSTIN
    for l in headerItems:
        if l["fieldId"] in ["vendorGSTIN", "billingGSTIN", "shippingGSTIN"]:
            if "entityMasterdata" in l:
                if l["entityMasterdata"] == 0:
                    # Non-SLA case as entityMasterdata not present
                    sla_flag = 0
                    non_sla_reason.append(str(l["fieldId"]) + " not present in MasterData/Document")
        elif l["fieldId"] in ["invoiceDate", "invoiceNumber"]:
            if "isReferenceDataPresent" in l:
                if l["isReferenceDataPresent"] == 0:
                    # Non-SLA case as RefernceData not present
                    sla_flag = 0
                    non_sla_reason.append(str("Refernce Data for " + l["fieldId"]) + " not present")

    return (sla_flag, non_sla_reason)


def convert_epoch_ist(epoch_time):
    """
    """
    epoch_time = str(epoch_time)
    if len(epoch_time) > 10:
        epoch_time = float(epoch_time)/1000

    epoch_time = float(epoch_time)
    time_ist = datetime.datetime.fromtimestamp(epoch_time, tz_ist).strftime("%d/%m/%Y %H:%M:%S")

    return time_ist


def populate_SLA_flag(dict_row):
    """
    """
    # print(dict_row)
    # State based SLA is valid only for REVIEW and REVIEW_COMPLETED documents
    # DELETED, PURGED and FAILED should not be considered 
    if dict_row["Status"] not in [statusReview, statusRevComp]:
        return

    if "Billing State" in dict_row:
        state = dict_row["Billing State"]
        if state in state_SLA:
            submit_date = datetime.datetime.strptime(convert_epoch_ist(dict_row["Submitted On"]), "%d/%m/%Y %H:%M:%S")
            sla_start_date = state_SLA[state]
            if submit_date < sla_start_date:
                dict_row["SLA_flag"] = 0
                dict_row["Non_SLA_reason"].append("SLA Date not started")
        else:
            dict_row["SLA_flag"] = 0
            dict_row["Non_SLA_reason"].append("Non-SLA State")
    else:
        dict_row["SLA_flag"] = 0
        dict_row["Non_SLA_reason"].append("Non-SLA State")

    # SLA Flag for Vendors
    if "nonSLAVendorFlag" in dict_row:
        if dict_row["nonSLAVendorFlag"] == 1:
            dict_row["SLA_flag"] = 0
            dict_row["Non_SLA_reason"].append("Non-SLA Vendor")

    # Quality Score based SLA Flag
    try:
        if "Quality Score" in dict_row:
            quality_score = dict_row["Quality Score"]
            if quality_score < 0.85:
                dict_row["SLA_flag"] = 0
                dict_row["Non_SLA_reason"].append("Poor Quality Document")
    except Exception as e:
        pass


def get_doc_metadata(document_id):
    """
    :param call_back_url:
    :param document_id:
    :param endpoint:
    :return:
    """
    url = form_call_back_url_doc_get()
    url = url + document_id

    doc_metadata = None
    try:
        print("Getting Document Metadat:", url)
        result = requests.get(url, verify=False).json().get('result')
        doc_metadata = result.get('document')
    except Exception as e:
        print(e)
        pass

    return doc_metadata


def get_doc_result(document_id):
    """
    :param call_back_url:
    :param document_id:
    :param endpoint:
    :return:
    """
    url = form_call_back_url_doc_result_get()
    url = url + document_id

    doc_res = None
    try:
        print("Getting Document Result:", url)
        result = requests.get(url, verify=False).json().get('result')
        doc_res = result.get('document')
    except Exception as e:
        print(e)
        pass

    return doc_res


def form_doc_metadata(doc_metadata):
    """
    """
    file_name = ""
    if "fileName" in doc_metadata:
        file_name = doc_metadata["fileName"]

    doc_id = doc_metadata["documentId"]
    status = doc_metadata["status"]
    
    dict_row = {}
    dict_row["Document ID"] = doc_id
    dict_row["Status"] = status

    if status == statusPurged:
        return dict_row

    dict_row["File Name"] = file_name
    # dict_row["Submitted On"] = ""
    # dict_row["Pages"] = ""
    # dict_row["Upload User"] = ""
    # dict_row["VENDOR NAME"] = ""
    # dict_row["VENDOR ID"] = ""
    # dict_row["VENDOR GSTIN"] = ""
    # dict_row["VENDOR State"] = ""
    # dict_row["Billing GSTIN"] = ""
    # dict_row["Billing State"] = ""
    # dict_row["Shipping GSTIN"] = ""
    # dict_row["Vendor Master"] = 0
    # dict_row["Billing Master"] = 0
    # dict_row["Shipping Master"] = 0
    # dict_row["User"] = ""
    # dict_row["Error Message"] = ""
    # dict_row["STP System"] = ""
    # dict_row["ACE"] = ""
    # dict_row["pAIges Confidence"] = ""
    # dict_row["pAIges Accuracy"] = ""
    # dict_row["File Size"] = ""
    # dict_row["User Comment"] = ""
    # dict_row["Total Review Time"] = ""
    # dict_row["Review Completion/Deletion Time"] = ""
    # dict_row["Delete Reason"] = ""
    # dict_row["Manually Reviewed"] = "Not Applicable"
    # # New field added for TAPP - BCP
    # dict_row["doc_type"] = None
    # dict_row["status_msg"] = None
    # dict_row["stage"] = None
    # dict_row["approval_status"] = None
    # dict_row["approver_email"] = None
    # dict_row["sent_to_approval_on"] = None
    # dict_row["approved_on"] = None
    # dict_row["approver_comment"] = None
    # dict_row["approver_designation"] = None
    # dict_row["re_opened"] = None
    # dict_row["posting_status"] = None


    if "comment" in doc_metadata:
        dict_row["User Comment"] = doc_metadata["comment"]

    if "submittedOn" in doc_metadata:
        dict_row["Submitted On"] = doc_metadata["submittedOn"]

    if "stp" in doc_metadata:
        dict_row["STP System"] = str(doc_metadata["stp"])

    if "pageCount" in doc_metadata:
        dict_row["Pages"] = doc_metadata["pageCount"]

    if "overall_score" in doc_metadata:
        dict_row["pAIges Confidence"] = doc_metadata["overall_score"]

    if "accuracy" in doc_metadata:
        dict_row["pAIges Accuracy"] = doc_metadata["accuracy"]

    if "size" in doc_metadata:
        dict_row["File Size"] = doc_metadata["size"]

    if "qualityScore" in doc_metadata:
        dict_row["Quality Score"] = doc_metadata["qualityScore"]

    if "vendorId" in doc_metadata:
        dict_row["VENDOR ID"] = doc_metadata["vendorId"]

    if "userId" in doc_metadata:
        dict_row["Upload User"] = doc_metadata["userId"]

    if "extractionCompletedOn" in doc_metadata:
        # Convert to millisecond
        epoch_time = str(doc_metadata["extractionCompletedOn"])
        if len(epoch_time) <= 10:
            epoch_time = int(epoch_time)*1000

        epoch_time = int(epoch_time)
        dict_row["Extraction Completion Time"] = epoch_time

    if "ace" in doc_metadata:
        if doc_metadata["ace"] == 0:
            dict_row["ACE"] = "NO"
        if doc_metadata["ace"] == 1:
            dict_row["ACE"] = "YES"   
        if doc_metadata["ace"] == 2:
            dict_row["ACE"] = "Not Applicable"

    # New field added for TAPP - BCP
    if "docType" in doc_metadata:
        dict_row["doc_type"] = doc_metadata["docType"]

    if "statusMsg" in doc_metadata:
        dict_row["status_msg"] = doc_metadata["statusMsg"]

    if "stage" in doc_metadata:
        dict_row["stage"] = doc_metadata["stage"]

    if "approvalStatus" in doc_metadata:
        dict_row["approval_status"] = doc_metadata["approvalStatus"]

    if "approverEmail" in doc_metadata:
        dict_row["approver_email"] = doc_metadata["approverEmail"]

    if "sentToApprovalOn" in doc_metadata:
        dict_row["sent_to_approval_on"] = doc_metadata["sentToApprovalOn"]

    if "approvedOn" in doc_metadata:
        dict_row["approved_on"] = doc_metadata["approvedOn"]

    if "approverComment" in doc_metadata:
        dict_row["approver_comment"] = doc_metadata["approverComment"]

    if "approverDesignation" in doc_metadata:
        dict_row["approver_designation"] = doc_metadata["approverDesignation"]

    if "reOpened" in doc_metadata:
        dict_row["re_opened"] = doc_metadata["reOpened"]

    if "postingStatus" in doc_metadata:
        dict_row["posting_status"] = doc_metadata["postingStatus"]

    # Code added for Total Review Time
    if "totalReviewedTime" in doc_metadata:
        dict_row["Total Review Time"] = doc_metadata["totalReviewedTime"]/60

    if "reviewedBy" in doc_metadata:
        dict_row["User"] = doc_metadata["reviewedBy"]
        dict_row["Manually Reviewed"] = "Yes"
    else:
        dict_row["User"] = "DUMMY_USER"
        dict_row["Manually Reviewed"] = "No"

    if "reviewedAt" in doc_metadata:
        dict_row["Review Completion/Deletion Time"] = doc_metadata["reviewedAt"]

    #Code added to mark Non-SLA Vendors
    if "nonSLAVendorFlag" in doc_metadata:
        dict_row["nonSLAVendorFlag"] = doc_metadata["nonSLAVendorFlag"]

    return dict_row


def form_doc_result_update_doc_metadata(doc_id, doc_metadata, dict_doc_metadata):
    """
    """
    doc_res = get_doc_result(document_id=doc_id)
    headerItems = {}
    if (doc_res is not None) and ("documentInfo" in doc_res):
        headerItems = doc_res["documentInfo"]
    status = doc_metadata["status"]
    list_doc_res = []

    if status == statusRevComp:
        # dict_doc_metadata["Total Review Time"] = ""
        # if "reviewedBy" in doc_metadata:
        #     dict_doc_metadata["User"] = doc_metadata["reviewedBy"]
        #     dict_doc_metadata["Manually Reviewed"] = "Yes"
        # else:
        #     dict_doc_metadata["User"] = "DUMMY_USER"
        #     dict_doc_metadata["Manually Reviewed"] = "No"

        # if "totalReviewedTime" in doc_metadata:
        #     dict_doc_metadata["Total Review Time"] = doc_metadata["totalReviewedTime"]/60

        # if "reviewedAt" in doc_metadata:
        #     dict_doc_metadata["Review Completion/Deletion Time"] = doc_metadata["reviewedAt"]

        count_correct = 0
        count_incorrect = 0
        count_total = 0
        for l in headerItems:
            # Code to add vendorName/GSTINs
            if l["fieldId"] == "vendorName":
                if "correctedValue" in l:
                    dict_doc_metadata["VENDOR NAME"] = l["correctedValue"]
                else:
                    dict_doc_metadata["VENDOR NAME"] = l["fieldValue"]
            if l["fieldId"] == "vendorGSTIN":
                if "correctedValue" in l:
                    dict_doc_metadata["VENDOR GSTIN"] = l["correctedValue"]
                else:
                    dict_doc_metadata["VENDOR GSTIN"] = l["fieldValue"]
                if "entityMasterdata" in l:
                    if l["entityMasterdata"] == 1:
                        dict_doc_metadata["Vendor Master"] = 1
            if l["fieldId"] == "billingGSTIN":
                if "correctedValue" in l:
                    dict_doc_metadata["Billing GSTIN"] = l["correctedValue"]
                else:
                    dict_doc_metadata["Billing GSTIN"] = l["fieldValue"]
                if "entityMasterdata" in l:
                    if l["entityMasterdata"] == 1:
                        dict_doc_metadata["Billing Master"] = 1
            if l["fieldId"] == "shippingGSTIN":
                if "correctedValue" in l:
                    dict_doc_metadata["Shipping GSTIN"] = l["correctedValue"]
                else:
                    dict_doc_metadata["Shipping GSTIN"] = l["fieldValue"]
                if "entityMasterdata" in l:
                    if l["entityMasterdata"] == 1:
                        dict_doc_metadata["Shipping Master"] = 1

            dict_row = {}
            dict_row["Document ID"] = doc_id
            dict_row["Field"] = l["fieldId"]
            if "correctedValue" in l:
                if (float(l["confidence"]) == 0) & (str(l["correctedValue"]).strip() != ""):
                    # Inside No Extraction Case: M
                    # print("Inside No Extraction Case")
                    dict_row["Status"] = "Missed"
                    count_incorrect += 1
                    count_total += 1
                elif (str(l["fieldValue"]).strip() != "") & (str(l["correctedValue"]).strip() != ""):
                    # Inside Wrong Extraction Case: X
                    # print("Inside Wrong Extraction Case")
                    # Code added fotr 0 removal in total post decimal points
                    # For total fields, compare the extracted and corrected value, if they are same, no correction
                    # Example: 3337.00 corrected as 3337 means no correction
                    if ("AMOUNT" in str(l["fieldId"]).upper()) | ("TOTAL" in str(l["fieldId"]).upper()):
                        try:
                            if (float(str(l["fieldValue"]).strip()) == float(str(l["correctedValue"]).strip())):
                                dict_row["Status"] = "OK"
                            else:
                                dict_row["Status"] = "Incorrect"
                                count_incorrect += 1
                        except Exception as e:
                            dict_row["Status"] = "Incorrect"
                            count_incorrect += 1
                            pass
                    else:
                        dict_row["Status"] = "Incorrect"
                        count_incorrect += 1
                    count_total += 1
                else:
                    dict_row["Status"] = "Incorrect"
                    count_incorrect += 1
                    count_total += 1
            else:
                # Inside No Correction Case: OK
                # Inside Field not present Case: ""
                if ("AMOUNT" in str(l["fieldId"]).upper()) | ("TOTAL" in str(l["fieldId"]).upper()):
                    try:
                        if (float(str(l["fieldValue"]).strip()) == 0) & (float(l["confidence"]) == 0):
                            dict_row["Status"] = ""
                        else:
                            dict_row["Status"] = "OK"
                            count_correct += 1
                            count_total += 1
                    except Exception as e:
                        dict_row["Status"] = "OK"
                        count_correct += 1
                        count_total += 1
                        pass
                else:
                    dict_row["Status"] = "OK"
                    count_correct += 1
                    count_total += 1

            if "boundingBox" in l:
                # Save original bounding box
                bb = l["boundingBox"]
                if "left" in bb:
                    dict_row["Left"] = bb["left"]
                if "right" in bb:
                    dict_row["Right"] = bb["right"]
                if "top" in bb:
                    dict_row["Top"] = bb["top"]
                if "bottom" in bb:
                    dict_row["Bottom"] = bb["bottom"]

            if "correctedBoundingBox" in l:
                # Save original bounding box
                    bb = l["correctedBoundingBox"]
                    if "left" in bb:
                        dict_row["Corr Left"] = bb["left"]
                    if "right" in bb:
                        dict_row["Corr Right"] = bb["right"]
                    if "top" in bb:
                        dict_row["Corr Top"] = bb["top"]
                    if "bottom" in bb:
                        dict_row["Corr Bottom"] = bb["bottom"]
            list_doc_res.append(dict_row)
        
        dict_doc_metadata["Calculated Accuracy pAIges"] = (count_correct/count_total)*100
        dict_doc_metadata["Calculated Accuracy Client"] = ((count_fields_client - count_incorrect)/count_fields_client)*100
        # Document SLA
        dict_doc_metadata["SLA_flag"], dict_doc_metadata["Non_SLA_reason"] = populate_SLA_flag_GSTIN(headerItems)
    elif status == statusDeleted:
        if "deleteReason" in doc_metadata:
            dict_doc_metadata["Delete Reason"] = doc_metadata["deleteReason"]
        if "deletedBy" in doc_metadata:
            dict_doc_metadata["User"] = doc_metadata["deletedBy"]
            dict_doc_metadata["Manually Reviewed"] = "Yes"
        else:
            dict_doc_metadata["User"] = "DUMMY_USER"
            dict_doc_metadata["Manually Reviewed"] = "No"
        if "deleteTime" in doc_metadata:
            dict_doc_metadata["Review Completion/Deletion Time"] = doc_metadata["deleteTime"]

        if headerItems is not None:
            for l in headerItems:
                if l["fieldId"] == "vendorName":
                    if "correctedValue" in l:
                        dict_doc_metadata["VENDOR NAME"] = l["correctedValue"]
                    else:
                        dict_doc_metadata["VENDOR NAME"] = l["fieldValue"]
                    if "entityMasterdata" in l:
                        if l["entityMasterdata"] == 1:
                            dict_doc_metadata["Vendor Master"] = 1
                if l["fieldId"] == "billingGSTIN":
                    if "correctedValue" in l:
                        dict_doc_metadata["Billing GSTIN"] = l["correctedValue"]
                    else:
                        dict_doc_metadata["Billing GSTIN"] = l["fieldValue"]
                    if "entityMasterdata" in l:
                        if l["entityMasterdata"] == 1:
                            dict_doc_metadata["Billing Master"] = 1
                if l["fieldId"] == "shippingGSTIN":
                    if "correctedValue" in l:
                        dict_doc_metadata["Shipping GSTIN"] = l["correctedValue"]
                    else:
                        dict_doc_metadata["Shipping GSTIN"] = l["fieldValue"]
                    if "entityMasterdata" in l:
                        if l["entityMasterdata"] == 1:
                            dict_doc_metadata["Shipping Master"] = 1
        # Document SLA
        dict_doc_metadata["SLA_flag"], dict_doc_metadata["Non_SLA_reason"] = 0, ["Document Rejected"]
    elif status == statusReview:
        for l in headerItems:
            # Code to add vendorName
            if l["fieldId"] == "vendorName":
                if "correctedValue" in l:
                    dict_doc_metadata["VENDOR NAME"] = l["correctedValue"]
                else:
                    dict_doc_metadata["VENDOR NAME"] = l["fieldValue"]
            if l["fieldId"] == "vendorGSTIN":
                if "correctedValue" in l:
                    dict_doc_metadata["VENDOR GSTIN"] = l["correctedValue"]
                else:
                    dict_doc_metadata["VENDOR GSTIN"] = l["fieldValue"]
                if "entityMasterdata" in l:
                    if l["entityMasterdata"] == 1:
                        dict_doc_metadata["Vendor Master"] = 1
            if l["fieldId"] == "billingGSTIN":
                if "correctedValue" in l:
                    dict_doc_metadata["Billing GSTIN"] = l["correctedValue"]
                else:
                    dict_doc_metadata["Billing GSTIN"] = l["fieldValue"]
                if "entityMasterdata" in l:
                    if l["entityMasterdata"] == 1:
                        dict_doc_metadata["Billing Master"] = 1
            if l["fieldId"] == "shippingGSTIN":
                if "correctedValue" in l:
                    dict_doc_metadata["Shipping GSTIN"] = l["correctedValue"]
                else:
                    dict_doc_metadata["Shipping GSTIN"] = l["fieldValue"]
                if "entityMasterdata" in l:
                    if l["entityMasterdata"] == 1:
                        dict_doc_metadata["Shipping Master"] = 1
        # Document SLA
        dict_doc_metadata["SLA_flag"], dict_doc_metadata["Non_SLA_reason"] = populate_SLA_flag_GSTIN(headerItems)
    elif status == statusPurged:
        print("Document Purged")
    elif "RPA" in status:
        print("RPA State Document!!")
    else:
        # Other Status
        # Document SLA
        dict_doc_metadata["SLA_flag"], dict_doc_metadata["Non_SLA_reason"] = 1, []
            
    # Code to extract Vendor State
    if "VENDOR GSTIN" in dict_doc_metadata:
        if dict_doc_metadata["VENDOR GSTIN"] != "":
            if len(str(dict_doc_metadata["VENDOR GSTIN"])) >= 2:
                gstin_v = dict_doc_metadata["VENDOR GSTIN"]
                if gstin_v[0:2] in dict_GSTIN:
                    dict_doc_metadata["VENDOR State"] = dict_GSTIN[gstin_v[0:2]][0]

    # Code to extract Billing State
    if "Billing GSTIN" in dict_doc_metadata:
        if dict_doc_metadata["Billing GSTIN"] != "":
            if len(str(dict_doc_metadata["Billing GSTIN"])) >= 2:
                gstin_b = dict_doc_metadata["Billing GSTIN"]
                if gstin_b[0:2] in dict_GSTIN:
                    dict_doc_metadata["Billing State"] = dict_GSTIN[gstin_b[0:2]][0]
    
    populate_SLA_flag(dict_doc_metadata)
    res = {"document_id":doc_id, "doc_metadata": dict_doc_metadata}
    if len(list_doc_res) > 0:
        res["doc_res"] = list_doc_res
    return res


def populate_rpa_metadata(doc_id, doc_metadata, res):
    """
    """
    status = doc_metadata["status"]
    dict_rpa_metadata = {}
    if "RPA" in status:
        if "documentId" in doc_metadata:
            dict_rpa_metadata["document_id"] = doc_metadata["documentId"]
        if "postingStatus" in doc_metadata:
            dict_rpa_metadata["rpa_posting_status"] = doc_metadata["postingStatus"]
        if "statusMsg" in doc_metadata:
            dict_rpa_metadata["rpa_posting_comment"] = doc_metadata["statusMsg"]
        if "rpa_posting_time" in doc_metadata:
            dict_rpa_metadata["rpa_posting_time"] = doc_metadata["rpa_posting_time"]


    if len(dict_rpa_metadata) > 0:
        res["rpa_metadata"] = dict_rpa_metadata

    return res


def form_metadata(doc_id):
    """
    """
    print("Forming metadata for:", doc_id)
    doc_metadata = (get_doc_metadata(doc_id))
    dict_doc_metadata = form_doc_metadata(doc_metadata)
    res = form_doc_result_update_doc_metadata(doc_id, doc_metadata, dict_doc_metadata)
    # Code added to populate RPA Metadata for RPA Status documents
    res = populate_rpa_metadata(doc_id, doc_metadata, res)
    return res

    
# @app.route('/collect_metadata', methods=['POST'])
def collect_metadata_wrapper(request):
    """

    :return:
    """
    response_object = {}
    try:
        print("Request received!!!")
        # rawContent = request.content.read()
        # encodedContent = rawContent.decode("utf-8")
        # content = json.loads(encodedContent)
        content = request
        print(content)
        document_id = content["document_id"]

        response_object = form_metadata(document_id)


        print("collect_metadata_wrapper:", response_object)
        response_object['status'] = "Success"
        response_object['responseCode'] = 200
    except Exception as e:
        print(e)
        traceback.print_exc()
        response_object['status'] = "Failure"
        response_object['responseCode'] = 500
        response_object['message'] = "Failure"
        pass

    response = json.dumps(response_object, indent=4, sort_keys=False, default=str)
    return response


if __name__ == "__main__":
    #main()
    app.run("0.0.0.0", 2222)