import pandas as pd
import random
import dashboard as dsh
import math
import time

tapp_implementation = False

time_format = "%d/%m/%Y %H:%M:%S"
date_format = "%d/%m/%Y"

colors = ["rgb(231, 76, 60, 0.6)", "rgb(155, 89, 182, 0.6)", "rgb(41, 128, 185, 0.6)", "rgb(26, 188, 156, 0.6)",
"rgb(39, 174, 96, 0.6)", "rgb(241, 196, 15, 0.6)", "rgb(230, 126, 34, 0.6)", "rgb(189, 195, 199, 0.6)", "rgb(52, 73, 94, 0.6)",
"rgb(74, 35, 90, 0.6)", "rgb(14, 98, 81, 0.6)", "rgb(120, 66, 18, 0.6)"]

status_mapping = {"DELETED": "DELETED",
"EXTRACTION_INPROGRESS": "OTHERS",
"FAILED": "FAILED",
"PURGED": "PURGED",
"REVIEW": "REVIEW",
"REVIEW_COMPLETED": "REVIEW COMPLETED"}


def select_colors(count_):
	"""
	"""
	selected_colors = colors.copy()
	if count_ <= len(colors):
		selected_colors = random.sample(colors, count_)
	else:
		while (len(selected_colors) != count_):
			selected_colors.append(random.choice(colors))

	return selected_colors


def status_wise_document_count(DF):
	"""
	Pie/Donught/Polar Area Chart: Status-wise Count
	"""
	# Form Data
	TEMP = DF.groupby(["status"])[["document_id"]].count().reset_index()
	TEMP["status"] = TEMP["status"].map(status_mapping).fillna(TEMP["status"])
	TEMP = TEMP.groupby(["status"])[["document_id"]].sum().reset_index()

	TEMP = TEMP.loc[~TEMP["document_id"].isna()]

	# Select Colors
	selected_colors = select_colors(TEMP.shape[0])

	labels = list(TEMP["status"])
	values = list(TEMP["document_id"])

	res = {"labels": labels,
	"datasets": [
    {"label": 'Document Count by Status',
      "data": values,
      "backgroundColor": selected_colors}
    ]}

	return res


def cum_ace_percentage_by_date(DF):
	"""
	Bar/Line Chart: Displays Cumulative Ace Percentage by Date
	"""
	TEMP = DF.groupby(["submit_date"])[["document_id"]].count().reset_index()
	TEMP["total_count"] = TEMP["document_id"].astype(int)

	ACE_COUNT = DF.loc[DF["ace"] == "YES"].groupby(["submit_date"])[["document_id"]].count().reset_index()
	ACE_COUNT["ace_count"] = ACE_COUNT["document_id"].astype(int)

	TEMP = pd.merge(TEMP, ACE_COUNT, on="submit_date", how="outer")

	TEMP['total_count_cum_sum'] = TEMP['total_count'].cumsum()
	TEMP['ace_count_cum_sum'] = TEMP['ace_count'].cumsum()

	TEMP["ace_percentage"] = 100*(TEMP["ace_count_cum_sum"]/TEMP["total_count_cum_sum"])
	TEMP["ace_percentage"] = TEMP["ace_percentage"].round(2)

	TEMP = TEMP.loc[~TEMP["ace_percentage"].isna()]

	labels = list(TEMP["submit_date"])
	values = list(TEMP["ace_percentage"])
	

	labels = [l.strftime(date_format)for l in labels]

	res = {"labels": labels,
	"datasets": [
    {"label": 'Cumulative ACE %',
      "data": values,
      "borderWidth": 1,
      "backgroundColor": 'rgba(169, 223, 191, 0.8)',
      "borderColor": '#283747'}
    ]}

	return res


def document_ace_count_by_date(DF):
	"""
	Bar/Line Chart: Displays Total Document Uploaded per Date
	"""
	TEMP = DF.groupby(["submit_date"])[["document_id"]].count().reset_index()
	TEMP["total_count"] = TEMP["document_id"].astype(int)

	ACE_COUNT = DF.loc[DF["ace"] == "YES"].groupby(["submit_date"])[["document_id"]].count().reset_index()
	ACE_COUNT["ace_count"] = ACE_COUNT["document_id"].astype(int)

	TEMP = pd.merge(TEMP, ACE_COUNT, on="submit_date", how="outer")

	TEMP.fillna({'total_count':0, 'ace_count':0}, inplace=True)

	labels = list(TEMP["submit_date"])
	values1 = list(TEMP["total_count"])
	values2 = list(TEMP["ace_count"])

	labels = [l.strftime(date_format)for l in labels]

	res = {"labels": labels,
	"datasets": [
	{"label": 'ACE Count',
      "data": values2,
      "borderWidth": 1,
      "backgroundColor": 'rgba(174, 214, 241, 0.5)',
      "borderColor": '#283747'},
    {"label": 'Documents Uploaded',
      "data": values1,
      "borderWidth": 1,
      "backgroundColor": 'rgba(246, 221, 204, 0.5)',
      "borderColor": '#283747'}
    ]}

	return res


def document_count_by_hour(DF):
	"""
	Line/Bar Chart: Displays Average Document Uploaded per Hour as per Hour of the Day
	"""
	TEMP = DF.groupby(["submit_hour"])[["document_id"]].count().reset_index()
	total_days = len(list(DF["submit_date"].unique()))
	TEMP["average_count"] = (TEMP["document_id"]/total_days).astype(int)

	TEMP = TEMP.loc[~TEMP["average_count"].isna()]

	labels = list(TEMP["submit_hour"])
	values = list(TEMP["average_count"])

	res = {"labels": labels,
	"datasets": [
    {"label": 'Average Documents Uploaded',
      "data": values,
      "borderWidth": 1,
      "backgroundColor": 'rgba(232, 218, 239, 0.8)',
      "borderColor": '#7D3C98'}
    ]}

	return res


def vendor_summary(DF):
	"""
	Tabular Format Data
	Code added to display Vendor Summary in Tabular Format
	"""
	DF["vendor_name"] = DF["vendor_name"].replace("", "UNKNOWN")
	DF["total_review_time"] = DF["total_review_time"].fillna(0)
	DF["status"] = DF["status"].map(status_mapping).fillna(DF["status"])


	TEMP = DF.groupby(["vendor_name"]).agg({'document_id':'count', 
                         'calculated_accuracy_paiges':'mean',
                         'total_review_time': 'mean'}).reset_index()
	TEMP["calculated_accuracy_paiges"] = TEMP["calculated_accuracy_paiges"].round(2)
	TEMP["total_review_time"] = (TEMP["total_review_time"]*60).round(2)

	TEMP.rename(columns = {'document_id': 'Document Count',
		'calculated_accuracy_paiges': 'Accuracy (%)',
		'total_review_time': 'Average Review Time (in Secs)'}, inplace = True)

	unique_status = list(DF["status"].unique())
	for status in unique_status:
		A = DF.loc[DF["status"] == status]
		B = A.groupby(["vendor_name", "status"]).agg({'document_id':'count'}).reset_index()
		B["document_id"] = B["document_id"].astype(int)
		B["document_id"] = B["document_id"].fillna(0)
		B.rename(columns = {'document_id':status}, inplace = True)
		TEMP = TEMP.merge(B[["vendor_name", status]], on=['vendor_name'], how='left')

	TEMP = TEMP.loc[~TEMP["Document Count"].isna()]
	TEMP.fillna(0, inplace=True)
	TEMP = TEMP.sort_values(["Document Count"], ascending=False)

	for status in unique_status:
		TEMP[status] = TEMP[status].astype(int)
	
	TEMP.rename(columns = {'vendor_name': 'Vendor Name'}, inplace = True)
	print(TEMP)
	
	res = TEMP.to_dict('records')

	return res


def document_count_by_billing_unit(DF):
	"""
	Polar Area Chart: 
	"""
	DF["billing_state"] = DF["billing_state"].replace("", "UNKNOWN")
	TEMP = DF.groupby(["billing_state"])[["document_id"]].count().reset_index()

	TEMP = TEMP.loc[~TEMP["document_id"].isna()]
	# Select Colors
	selected_colors = select_colors(TEMP.shape[0])

	labels = list(TEMP["billing_state"])
	values = list(TEMP["document_id"])

	res = {"labels": labels,
	"datasets": [
    {"label": 'Document Count by Billing Unit',
      "data": values,
      "backgroundColor": selected_colors}
    ]}

	return res


def accuracy_vs_confidence(DF):
	"""
	Scatter Plot: Accuracy vs Confidence
	"""
	TEMP = DF.loc[~DF["paiges_confidence"].isna()]
	TEMP = DF.loc[~DF["calculated_accuracy_paiges"].isna()]

	confidence_ = list(TEMP["paiges_confidence"])
	accuracy_ = list(TEMP["calculated_accuracy_paiges"])

	data = [{"x":confidence_[i], "y": accuracy_[i]} for i in range(len(confidence_))]

	res = {
	"datasets": [
    {"label": 'Accuracy vs Confidence',
      "data": {"x":confidence_, "y":accuracy_},
      "borderWidth": 1,
      "backgroundColor": 'rgba(232, 218, 239, 0.8)',
      "borderColor": '#7D3C98'}
    ]}

	return res


def document_count_by_date_posting_status(DF):
	"""
	"""
	TEMP = DF.loc[~DF["rpa_posting_status"].isna()]

	A = TEMP.groupby(["rpa_receive_date", "rpa_posting_status"])[["rpa_record_id"]].count().reset_index()


	unique_psoting_status = list(TEMP["rpa_posting_status"].unique())

	B = None
	for l in unique_psoting_status:
		A = TEMP.loc[TEMP["rpa_posting_status"] == l].groupby(["rpa_receive_date"])[["rpa_record_id"]].count().reset_index()
		A.rename(columns = {'rpa_record_id':l}, inplace = True)
		if B is None:
			B = A.copy()
		else:
			B = pd.merge(B,A, how='outer', on='rpa_receive_date')

	if B is None:
		raise Exception("No data found")
		return

	B.fillna(0, inplace=True)

	B = B.sort_values(["rpa_receive_date"], ascending=True)
	labels = list(B["rpa_receive_date"])
	labels = [l.strftime(date_format)for l in labels]

	datasets = []
	# Choose Color
	selected_colors = select_colors(len(unique_psoting_status))
	for idx, l in enumerate(unique_psoting_status):
		datasets.append({"label": "Posting Status -" + str(l),
			"data": list(B[l]),
			"borderWidth": 1,
			"backgroundColor": selected_colors[idx],
			"borderColor": '#283747'})

	res = {"labels": labels,
	"datasets": datasets}

	return res


def approval_aging_by_approver(DF):
	"""
	Show Plot of Approval Age of documents for Approvers
	"""
	TEMP = DF.loc[~DF["sent_to_approval_on"].isna()]
	
	# TEMP = TEMP.loc[TEMP["approved_on"].isna()]

	current_time = round(time.time() * 1000)
	TEMP['approved_on'] = TEMP['approved_on'].fillna(value=current_time)
	
	TEMP["approval_age"] = TEMP["approved_on"] - TEMP["sent_to_approval_on"]
	# Convert time to minutes
	TEMP["approval_age"] = TEMP["approval_age"]/60000
	# Convert time to days
	TEMP["approval_age"] = TEMP["approval_age"]/1440

	TEMP["approval_age_category"] = "0-1 DAY"
	TEMP.loc[(TEMP["approval_age"] > 1) & (TEMP["approval_age"] <= 2), 'approval_age_category'] = "1-2 DAY"
	TEMP.loc[(TEMP["approval_age"] > 2) & (TEMP["approval_age"] <= 3), 'approval_age_category'] = "2-3 DAY"
	TEMP.loc[(TEMP["approval_age"] > 3) & (TEMP["approval_age"] <= 7), 'approval_age_category'] = "3-7 DAY"
	TEMP.loc[(TEMP["approval_age"] > 7) , 'approval_age_category'] = "MORE THAN 7 DAYS"

	A = TEMP.groupby(["approval_age_category", "approver_email"])[["rpa_record_id"]].count().reset_index()


	unique_approval_age = list(TEMP["approval_age_category"].unique())

	B = None
	for l in unique_approval_age:
		A = TEMP.loc[TEMP["approval_age_category"] == l].groupby(["approver_email"])[["rpa_record_id"]].count().reset_index()
		A.rename(columns = {'rpa_record_id':l}, inplace = True)
		if B is None:
			B = A.copy()
		else:
			B = pd.merge(B,A, how='outer', on='approver_email')

	if B is None:
		raise Exception("No data found")
		return

	B.fillna(0, inplace=True)

	labels = list(B["approver_email"])

	datasets = []
	selected_colors = select_colors(len(unique_approval_age))
	for idx, l in enumerate(unique_approval_age):
		datasets.append({"label": "Approval Age -" + str(l),
			"data": list(B[l]),
			"borderWidth": 1,
			"backgroundColor": selected_colors[idx],
			"borderColor": '#283747'})

	res = {"labels": labels,
	"datasets": datasets}

	return res


def approval_aging(DF):
	"""
	Show Plot of Approval Age of documents
	"""
	TEMP = DF.loc[~DF["sent_to_approval_on"].isna()]
	TEMP = TEMP.loc[TEMP["approved_on"].isna()]

	current_time = round(time.time() * 1000)
	TEMP["current_time"] = current_time
	TEMP["approval_age"] = TEMP["current_time"] - TEMP["sent_to_approval_on"]
	# Convert time to minutes
	TEMP["approval_age"] = TEMP["approval_age"]/60000
	# Convert time to days
	TEMP["approval_age"] = TEMP["approval_age"]/1440

	TEMP["approval_age_category"] = "0-1 DAY"
	TEMP.loc[(TEMP["approval_age"] > 1) & (TEMP["approval_age"] <= 2), 'approval_age_category'] = "1-2 DAY"
	TEMP.loc[(TEMP["approval_age"] > 2) & (TEMP["approval_age"] <= 3), 'approval_age_category'] = "2-3 DAY"
	TEMP.loc[(TEMP["approval_age"] > 3) & (TEMP["approval_age"] <= 7), 'approval_age_category'] = "3-7 DAY"
	TEMP.loc[(TEMP["approval_age"] > 7) , 'approval_age_category'] = "MORE THAN 7 DAYS"

	A = TEMP.groupby(["approval_age_category"])[["rpa_record_id"]].count().reset_index()

	selected_colors = select_colors(A.shape[0])
	labels = list(A["approval_age_category"])
	values = list(A["rpa_record_id"])

	res = {"labels": labels,
	"datasets": [
    {"label": 'Document Approval Age',
      "data": values,
      "borderWidth": 1,
      "backgroundColor": selected_colors,
      "borderColor": '#283747'}
    ]}

	return res



def posting_approver_summary(DF):
	"""
	Bar Chart: X-axis Approver Email
	Y-Axis Stacked Bar with different Status Documents
	"""
	TEMP = DF.loc[~DF["approval_status"].isna()]
	TEMP = TEMP.loc[~DF["approver_email"].isna()]
	A = TEMP.groupby(["approver_email", "approval_status"])[["rpa_record_id"]].count().reset_index()


	unique_approval_status = list(TEMP["approval_status"].unique())

	B = None
	for l in unique_approval_status:
		A = TEMP.loc[TEMP["approval_status"] == l].groupby(["approver_email"])[["rpa_record_id"]].count().reset_index()
		A.rename(columns = {'rpa_record_id':l}, inplace = True)
		if B is None:
			B = A.copy()
		else:
			B = pd.merge(B,A, how='outer', on='approver_email')

	if B is None:
		raise Exception("No data found")
		return

	B.fillna(0, inplace=True)

	labels = list(B["approver_email"])

	datasets = []
	selected_colors = select_colors(len(unique_approval_status))
	for idx, l in enumerate(unique_approval_status):
		datasets.append({"label": "Approval Status -" + str(l),
			"data": list(B[l]),
			"borderWidth": 1,
			"backgroundColor": selected_colors[idx],
			"borderColor": '#283747'})

	res = {"labels": labels,
	"datasets": datasets}

	return res

def document_count_by_exception(DF):
	"""
	"""
	DF = DF.loc[DF["rpa_posting_status"] != "Success"]
	TEMP = DF.groupby(["rpa_posting_comment"])[["rpa_record_id"]].count().reset_index()
	print(TEMP)
	# TEMP = TEMP.loc[TEMP["rpa_record_id"] >= 10]

	if (TEMP is None) | (TEMP.shape[0] == 0):
		raise Exception("No data found")
		return

	# Select Colors
	selected_colors = select_colors(TEMP.shape[0])

	labels = list(TEMP["rpa_posting_comment"])
	values = list(TEMP["rpa_record_id"])

	# Truncate Exception
	labels_truncated = []
	for l in labels:
		if len(l) > 69:
			truncated_label = l[0:69]
			truncated_label = truncated_label + " ..."
			labels_truncated.append(truncated_label)
		else:
			labels_truncated.append(l)

	res = {"labels": labels_truncated,
	"datasets": [
    {"label": 'Document Count by Exception',
      "data": values,
      "backgroundColor": selected_colors}
    ]}

	return res


def field_level_accuracy(DF):
	"""
	"""
	list_document_ids = list(DF["document_id"])
	tuple_document_ids = tuple(list_document_ids)
	TEMP = dsh.field_level_accuracy(tuple_document_ids)

	labels = list(TEMP["field_id"])
	values = list(TEMP["accuracy"])

	res = {"labels": labels,
	"datasets": [
    {"label": 'Field Level Accuracy',
      "data": values,
      "borderWidth": 1,
      "backgroundColor": 'rgba(34, 153, 84 , 0.8)',
      "borderColor": '#283747'}
    ]}

	return res


def posting_forever_summary(DF):
	"""
	"""
	res = {}
	TEMP = dsh.posting_forever_summary()
	if TEMP is None or TEMP.shape[0] == 0:
		return res

	dict_res = dict(TEMP.iloc[0])
	
	res["Documents Received"] = {"Value": int(dict_res["Documents Received"])}
	res["Documents Submitted"] = {"Value": int(dict_res["Documents Submitted"])}
	res["Documents Posted"] = {"Value": int(dict_res["Documents Posted"])}
	res["Documents Exception"] = {"Value": int(dict_res["Documents Exception"])}
	res["Extraction TAT"] = {"Value": round(dict_res["Extraction TAT"], 2), "Unit": "Minutes"}
	
	return res


def posting_current_summary(DF):
	"""
	"""
	doc_count = DF.shape[0]

	res = {}
	res["Documents Received"] = {"Value": doc_count}
	res["Documents Submitted"] = {"Value": int(DF.loc[DF["rpa_upload_status"] == "Success"].shape[0])}
	res["Documents Posted"] = {"Value": int(DF.loc[DF["rpa_posting_status"] == "Success"].shape[0])}
	res["Documents Exception"] = {"Value": int(DF.loc[(~DF["rpa_posting_status"].isna()) & (DF["rpa_posting_status"] != "Success")].shape[0])}

	TEMP = DF.loc[~DF["rpa_receive_time"].isna()]
	TEMP = TEMP.loc[~TEMP["extraction_completion_time"].isna()]

	print(TEMP[["rpa_receive_time", "extraction_completion_time"]].head())
	extraction_tat = ((TEMP["extraction_completion_time"] - TEMP["rpa_receive_time"])/60000).mean()

	res["Extraction TAT"] = {"Value": round(extraction_tat, 2), "Unit": "Minutes"}

	return res


def extraction_current_summary(DF):
	"""
	"""
	print("")
	doc_count = DF.shape[0]

	res = {}
	res["Documents Extracted"] = {"Value": doc_count}
	res["Pages Extracted"] = {"Value": int(DF["pages"].sum())}
	if not tapp_implementation:
		res["Average Confidence"] = {"Value": str(round(DF["paiges_confidence"].mean(), 2)) + " %"}

	if math.isnan(DF["calculated_accuracy_client"].mean()):
		res["Average Accuracy"] = {"Value": str("-") + " %", "Unit": "(Headers)"}
	else:
		res["Average Accuracy"] = {"Value": str(round(DF["calculated_accuracy_client"].mean(), 2)) + " %", "Unit": "(Headers)"}

	if not tapp_implementation:
		res["ACE Percentage"] = {"Value": str(round(((DF.loc[DF["ace"] == "YES"].shape[0])/doc_count)*100, 2)) + " %"}
	res["Average Review Time"] = {"Value": round(float(((DF["total_review_time"].sum())/doc_count)*60), 2), "Unit": "Seconds"}

	return res


def extraction_forever_summary(DF):
	"""
	"""
	res = {}
	TEMP = dsh.extraction_forever_summary()
	if TEMP is None or TEMP.shape[0] == 0:
		return res

	dict_res = dict(TEMP.iloc[0])
	
	res["Documents Extracted"] = {"Value": int(dict_res["Documents Extracted"])}
	res["Pages Extracted"] = {"Value": int(dict_res["Pages Extracted"])}

	if not tapp_implementation:
		res["Average Confidence"] = {"Value": str(float(dict_res["Average Confidence"])) + " %"}

	if dict_res["Average Accuracy"] is not None:
		res["Average Accuracy"] = {"Value": str(float(dict_res["Average Accuracy"])) + " %", "Unit": "(Headers)"}
	else:
		res["Average Accuracy"] = {"Value": str("-") + " %", "Unit": "(Headers)"}

	if not tapp_implementation:
		if dict_res["ACE Percentage"] is not None:
			res["ACE Percentage"] = {"Value": str(float(dict_res["ACE Percentage"])) + " %"}
		else:
			res["ACE Percentage"] = {"Value": str("-") + " %"}

	if dict_res["Average Review Time"] is not None:
		res["Average Review Time"] = {"Value": float(dict_res["Average Review Time"]), "Unit": "Seconds"}
	else:
		res["Average Review Time"] = {"Value": 0.0, "Unit": "Seconds"}
	
	return res
