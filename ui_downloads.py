import pandas as pd
import os
import TAPPconfig as cfg
import json
import traceback
import requests
import time
import pytz
import datetime
import numpy as np


from klein import Klein
app = Klein()


UI_SERVER = cfg.getUIServer()
# UI_SERVER = "http://106.51.73.100:8888"
# UI_SERVER = "https://taotapp-bcp.com"
FIND_DOCUMENT = cfg.getDocumentFind()
download_path = os.path.join(cfg.getUIRootFolder(),cfg.getUIDownloadFolder())

time_zone_code = "Asia/Kolkata"
time_zone = pytz.timezone(time_zone_code)
time_format = "%d/%m/%Y %H:%M:%S"
date_format = "%d/%m/%Y"

def convert_int(s):
    val = None
    try:
        val = int(s)
    except Exception as e:
        pass
    return val


def form_call_back_url_doc_result_get():
	"""
	:return:
	"""
	endpoint = "document/result/get"
	url = UI_SERVER + "/" + endpoint + "/"
	return url


def form_call_back_url_doc_get():
	"""
	:return:
	"""
	endpoint = "document/get"
	url = UI_SERVER + "/" + endpoint + "/"
	return url

def form_call_back_url_doc_find(end_point):
	"""
	"""
	find_url = UI_SERVER + end_point
	return find_url


def calculate_extraction_time(submittedOn, extractionCompletedOn):
	"""
	"""
	submittedOn = str(submittedOn)
	extractionCompletedOn = str(extractionCompletedOn)

	if len(submittedOn) > 10:
		submittedOn = float(submittedOn)/1000

	if len(extractionCompletedOn) > 10:
		extractionCompletedOn = float(extractionCompletedOn)/1000

	extraction_time = float(extractionCompletedOn) - float(submittedOn)

	return round(extraction_time, 2)



def convert_epoch_time_zone(epoch_time):
    """
    Convert 13/10 digit epoch time to date-time
    """
    # print(epoch_time)
    try:
    	epoch_time = str(int(epoch_time))
    	if len(epoch_time) > 10:
    		epoch_time = int(epoch_time)/1000
    	else:
    		epoch_time = int(epoch_time)

    	time_local = datetime.datetime.fromtimestamp(epoch_time, time_zone).strftime(time_format)
    	return pd.to_datetime(time_local, format=time_format)
    except Exception as e:
    	return ''


def download_data(payload):
	"""
	"""
	print("download_data payload:", payload)
	end_point = FIND_DOCUMENT
	if "request" in payload:
		if "endPoint" in payload["request"]:
			print("Received endPoint in payload:", payload["request"]["endPoint"])
			end_point = payload["request"]["endPoint"]

	find_url = form_call_back_url_doc_find(end_point)

	headers = {}
	headers["Content-Type"] = "application/json"
	data = json.dumps(payload)
	

	print("Calling Find API:",find_url)
	print("Payload:", data)
	
	current_time_stamp = round(time.time() * 1000)
	download_file_name = "DocumentList_" + str(current_time_stamp) + ".xlsx"
	local_path = os.path.join(download_path , download_file_name)
	print(local_path)

	try:
		response =  requests.post(find_url, headers=headers, data = data)
	except Exception as e:
		print("Error in document_get API",traceback.print_exc())
		return {"downloaded": False, "download_path": download_file_name}

	response = response.json()

	list_df = []

	if response["responseCode"] == "OK":
		results = response["result"]
		total_res_count = results["count"]
		documents = results["documents"]
		print("TOTAL DOCUMENT COUNT:", total_res_count)

		for doc in documents:
			dict_row = {}

			dict_row["File Name"] = doc["fileName"]
			dict_row["Document ID"] = doc["documentId"]
			dict_row["Status"] = doc["status"]
			dict_row["Submitted Date"] = doc["submittedOn"]
			
			if "lastUpdatedOn" in doc:
				dict_row["Last Updated"] = doc["lastUpdatedOn"]

			if "stp" in doc:
				dict_row["STP"] = doc["stp"]

			if "ace" in doc:
				if doc["ace"] == 0:
					dict_row["ACE"] = "NO"
				if doc["ace"] == 1:
					dict_row["ACE"] = "YES"   
				if doc["ace"] == 2:
					dict_row["ACE"] = "Not Applicable"

			if "overall_score" in doc:
				dict_row["Confidence"] = doc["overall_score"]

			if "docType" in doc:
				dict_row["Document Type"] = doc["docType"]

			if "totalReviewedTime" in doc:
				dict_row["Review Time"] = doc["totalReviewedTime"]

			if ("extractionCompletedOn" in doc) and ("submittedOn" in doc):
				dict_row["Extraction Time"] = calculate_extraction_time(doc["submittedOn"], doc["extractionCompletedOn"])

			if "invoiceDate" in doc:
				dict_row["Invoice Date"] = doc["invoiceDate"]

			if "invoiceNumber" in doc:
				dict_row["Invoice Number"] = doc["invoiceNumber"]

			if "totalAmount" in doc:
				dict_row["Total Amount"] = doc["totalAmount"]

			if "vendorName" in doc:
				dict_row["Vendor Name"] = doc["vendorName"]

			if "stage" in doc:
				dict_row["Stage"] = doc["stage"]

			if "statusMsg" in doc:
				dict_row["Status Message"] = doc["statusMsg"]

			if "postingStatus" in doc:
				dict_row["Posting Status"] = doc["postingStatus"]

			if "approverDesignation" in doc:
				dict_row["Approver Designation"] = doc["approverDesignation"]

			if "approverEmail" in doc:
				dict_row["Approver Email"] = doc["approverEmail"]

			if "sentToApprovalOn" in doc:
				dict_row["Sent to Approval"] = doc["sentToApprovalOn"]

			if "approvalStatus" in doc:
				dict_row["Approval Status"] = doc["approvalStatus"]

			if "approverComment" in doc:
				dict_row["Approver Comment"] = doc["approverComment"]

			if "approvedOn" in doc:
				dict_row["Approved On"] = doc["approvedOn"]

			if "comment" in doc:
				dict_row["Comment"] = doc["comment"]

			list_df.append(dict_row)
	else:
		return {"downloaded": False, "download_path": download_file_name}

	try:
		DF = pd.DataFrame(list_df)
		cols = list(DF.columns)

		if "Approved On" in cols:
			DF["Approved On"] = DF["Approved On"].replace('', np.nan)
			# print(list(DF["Approved On"]))
			DF["Approved On"] = DF["Approved On"].apply(convert_int)
			DF["Approved On"] = DF["Approved On"].astype('Int64')
			# print(list(DF["Approved On"]))
			# DF["Approved On"] = DF["Approved On"].apply(remove_millisecond_epoch)
			# print(list(DF["Approved On"]))
			DF["Approved On"] = DF["Approved On"].apply(convert_epoch_time_zone)

		if "Sent to Approval" in cols:
			# print(list(DF["Sent to Approval"]))
			DF["Sent to Approval"] = DF["Sent to Approval"].replace('', np.nan)
			# print(list(DF["Sent to Approval"]))
			DF["Sent to Approval"] = DF["Sent to Approval"].apply(convert_int)
			DF["Sent to Approval"] = DF["Sent to Approval"].astype('Int64')
			# DF["Sent to Approval"] = DF["Sent to Approval"].apply(remove_millisecond_epoch)
			# print(list(DF["Sent to Approval"]))
			DF["Sent to Approval"] = DF["Sent to Approval"].apply(convert_epoch_time_zone)

		if "Submitted Date" in cols:
			DF["Submitted Date"] = DF["Submitted Date"].astype('Int64')
			# DF["Submitted Date"] = DF["Submitted Date"].apply(remove_millisecond_epoch)
			# print(list(DF["Submitted Date"]))
			DF["Submitted Date"] = DF["Submitted Date"].apply(convert_epoch_time_zone)

		if "Last Updated" in cols:
			DF["Last Updated"] = DF["Last Updated"].astype('Int64')
			# DF["Last Updated"] = DF["Last Updated"].apply(remove_millisecond_epoch)
			# print(list(DF["Last Updated"]))
			DF["Last Updated"] = DF["Last Updated"].apply(convert_epoch_time_zone)

		# DF.to_excel("../../../bbbb.xlsx", index=False)
		DF.to_excel(local_path, index=False)
	except Exception as e:
		print("Error in Writing File",traceback.print_exc())
		return {"downloaded": False, "download_path": download_file_name}

	return {"downloaded": True, "download_path": download_file_name}


@app.route('/ui_downloads/list_view', methods=['POST'])
def download_list_view_data(request):
	"""
	"""
	response_object = {}
	try:
		print("Request received!!!")
		# rawContent = request.content.read()
		# encodedContent = rawContent.decode("utf-8")
		# payload = json.loads(encodedContent)
		payload = request
		print(payload)

		if "time_zone" in payload:
			global time_zone_code, time_zone
			time_zone_code = payload["time_zone"]
			time_zone = pytz.timezone(time_zone_code)

		response_object = download_data(payload)

		response_object['status'] = "Success"
		response_object['responseCode'] = 200
	except Exception as e:
		print(e)
		traceback.print_exc()
		response_object['status'] = "Failure"
		response_object['responseCode'] = 500
		response_object['message'] = "Failure"
		pass

	# request.responseHeaders.addRawHeader(b"content-type", b"application/json")
	response = json.dumps(response_object, indent=4, sort_keys=False, default=str)
	return response


def main():
	"""
	"""
	global time_zone
	time_zone = pytz.timezone('Asia/Kolkata')

	download_list_view_data()


if __name__ == "__main__":
	# main()
	app.run("0.0.0.0", 2222)
