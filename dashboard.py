# -*- coding: utf-8 -*-

import psycopg2
# import psycopg2.extras
import pandas as pd
import pytz
import datetime
import traceback
import random
import json
import graph_functions


# from sqlalchemy import create_engine
# engine = create_engine('postgresql+psycopg2://swginsta_admin@swginstapaiges:jQ0y]QcA0Qv.p]D@swginstapaiges.postgres.database.azure.com/swginsta_pAIges')

from klein import Klein
app = Klein()

# Time Zone is needed to form data after filtering
# After filetring, UTC time will be converted to local time zone for all the other operations
# such as, aggregation based on Month, Week, Day of Hour etc.
# time_zone will be passed in the API call, default value is 'Asia/Kolkata'
time_zone_code = "Asia/Kolkata"
time_zone = pytz.timezone(time_zone_code)
time_format = "%d/%m/%Y %H:%M:%S"
date_format = "%d/%m/%Y"

# Should be True for SWIGGY
improper_data_type_sql = False
# Code to connect to SQL Database
conn = psycopg2.connect(
   database="blinkit-tapp", user='blinkit_sql@blinkittappsql', password='9+px|,rJ&Jt&FZJ',
   host='blinkittappsql.postgres.database.azure.com',
   port= '5432'
)
# BCP PROD Server Details
# conn = psycopg2.connect(
#    database="bcp-tapp", user='bcppaiges@bcppaigessql', password="jm|-b'T3*9,dj4%",
#    host='bcppaigessql.postgres.database.azure.com',
#    port= '5432'
# )
# print(conn.encoding)
# conn.set_client_encoding('UNICODE')
cursor = conn.cursor()
cursor.execute("select version()")

# # cursor = conn.cursor('cursor_unique_name', cursor_factory=psycopg2.extras.DictCursor)
# # cursor.execute("select version()")
# data = cursor.fetchone()
# print("Connection established to: ",data)


table_rpa_metadata = "blinkit_rpa_metadata"
table_document_metadata = "blinkit_document_metadata"
table_document_result = "blinkit_document_result"

# table_rpa_metadata = "bcp_prod_posting_data"
# table_document_metadata = "bcp_prod_document_metadata"
# table_document_result = "bcp_prod_document_result"

# table_rpa_metadata = "client_demo_posting_data"
# table_document_metadata = "client_demo_document_metadata"
# table_document_result = "client_demo_document_result"

def posting_forever_summary():
	"""
	"""
	print("Posting Forever Summary:")
	postgreSQL_select_statement = "select " 
	postgreSQL_select_statement += "count(rpa_record_id) as \"Documents Received\","
	postgreSQL_select_statement += "(select count(*) from "
	postgreSQL_select_statement +=  table_rpa_metadata
	postgreSQL_select_statement += " where rpa_upload_status = 'Success') as \"Documents Submitted\","
	postgreSQL_select_statement += "(select count(*) from "
	postgreSQL_select_statement += table_rpa_metadata 
	postgreSQL_select_statement += " where rpa_posting_status = 'Success') as \"Documents Posted\","
	postgreSQL_select_statement += "(select count(*) from "
	postgreSQL_select_statement += table_rpa_metadata 
	postgreSQL_select_statement += " where rpa_posting_status != 'Success') as \"Documents Exception\","
	postgreSQL_select_statement += "(select avg((cast(extraction_completion_time as double precision) - rpa_receive_time)/60000) from "
	postgreSQL_select_statement += table_rpa_metadata 
	postgreSQL_select_statement += " where (rpa_receive_time is not null) and (extraction_completion_time is not null)) as \"Extraction TAT\""
	postgreSQL_select_statement += " from "
	postgreSQL_select_statement += table_rpa_metadata

	print(postgreSQL_select_statement)

	cursor.execute(postgreSQL_select_statement)
	column_names = [i[0] for i in cursor.description]
	
	# records = cursor.fetchall()
	list_records = []
	for row in cursor:
		list_records.append(row)

	DF = pd.DataFrame(list_records, columns=column_names)
	return DF


def extraction_forever_summary():
	"""
	"""
	print("Extraction Forever Summary:")
	postgreSQL_select_statement = "select " 
	postgreSQL_select_statement += "count(document_id) as \"Documents Extracted\","
	postgreSQL_select_statement += "sum(cast(coalesce(nullif(pages,''),'0') as float)) as \"Pages Extracted\","
	if improper_data_type_sql:
		postgreSQL_select_statement += "round(avg(cast(coalesce(nullif(paiges_confidence,'')) as float))::numeric, 2) as \"Average Confidence\","
	else:
		postgreSQL_select_statement += "round(avg(paiges_confidence)::numeric, 2) as \"Average Confidence\","
	if improper_data_type_sql:
		postgreSQL_select_statement += "round(avg(cast(coalesce(nullif(calculated_accuracy_client,'')) as float))::numeric, 2) as \"Average Accuracy\","
	else:
		postgreSQL_select_statement += "round(avg(calculated_accuracy_client)::numeric, 2) as \"Average Accuracy\","
	postgreSQL_select_statement += "round((cast((select count(*) from "
	postgreSQL_select_statement += table_document_metadata 
	postgreSQL_select_statement += " where ace = 'YES') as float)*100/cast((select count(*) from "
	postgreSQL_select_statement += table_document_metadata
	postgreSQL_select_statement += ") as float))::numeric, 2) as \"ACE Percentage\","
	if improper_data_type_sql:
		postgreSQL_select_statement += "round((sum(cast(coalesce(nullif(total_review_time,'')) as float))*60/count(document_id))::numeric, 2) as \"Average Review Time\""
	else:
		postgreSQL_select_statement += "round((sum(total_review_time)*60/count(document_id))::numeric, 2) as \"Average Review Time\""
	postgreSQL_select_statement += " from "
	postgreSQL_select_statement += table_document_metadata

	print(postgreSQL_select_statement)

	cursor.execute(postgreSQL_select_statement)
	column_names = [i[0] for i in cursor.description]
	
	# records = cursor.fetchall()
	list_records = []
	for row in cursor:
		list_records.append(row)

	DF = pd.DataFrame(list_records, columns=column_names)
	return DF


def field_level_accuracy(document_ids):
	"""
	"""
	print("field_level_accuracy:")
	postgreSQL_select_statement = "select " 
	postgreSQL_select_statement += "field_id, status, count(document_id) from "
	postgreSQL_select_statement += table_document_result
	postgreSQL_select_statement += " where document_id in "
	postgreSQL_select_statement += str(document_ids)
	postgreSQL_select_statement += " group by field_id, status"

	cursor.execute(postgreSQL_select_statement)
	column_names = [i[0] for i in cursor.description]
	
	# records = cursor.fetchall()
	list_records = []
	for row in cursor:
		list_records.append(row)

	DF = pd.DataFrame(list_records, columns=column_names)
	DF = DF.loc[~DF["status"].isna()]
	DF = DF.loc[DF["status"] != ""]

	A = DF.groupby(["field_id"])[["count"]].sum().reset_index()
	B = DF.loc[DF["status"] == "OK"].groupby(["field_id"])[["count"]].sum().reset_index()
	A.rename(columns = {'count':"total_count"}, inplace = True)
	B.rename(columns = {'count':"correct_count"}, inplace = True)


	TEMP = pd.merge(A, B, on="field_id", how="outer")
	TEMP.fillna({'correct_count':0}, inplace=True)
	TEMP["accuracy"] = (TEMP["correct_count"]/TEMP["total_count"])*100
	TEMP["accuracy"] = TEMP["accuracy"].round(2)

	return TEMP


def build_execute_sql_query(table_name, where_clause):
	"""
	table_name: Name of the table
	where_clause: Where Clause in string format

	"""
	print("build_execute_sql_query:")
	# where_clause = "status = 'PURGED'"
	postgreSQL_select_statement = "select * from " + table_name + " where " + where_clause
	print(postgreSQL_select_statement)
	
	cursor.execute(postgreSQL_select_statement)
	column_names = [i[0] for i in cursor.description]
	
	# records = cursor.fetchall()
	list_records = []
	for row in cursor:
		list_records.append(row)

	return pd.DataFrame(list_records, columns=column_names)


def add_millisecond_epoch(epoch_time):
    """
    Convert 13/10 digit epoch time to date-time
    """
    # print(epoch_time)
    try:
    	epoch_time = str(int(epoch_time))
    	if len(epoch_time) > 10:
    		epoch_time = int(epoch_time)
    	else:
    		epoch_time = int(epoch_time) * 1000

    	return epoch_time
    except Exception as e:
    	return None


def convert_epoch_time_zone(epoch_time):
    """
    Convert 13/10 digit epoch time to date-time
    """
    # print(epoch_time)
    try:
    	epoch_time = str(epoch_time)
    	if len(epoch_time) > 10:
    		epoch_time = int(epoch_time)/1000
    	else:
    		epoch_time = int(epoch_time)

    	time_local = datetime.datetime.fromtimestamp(epoch_time, time_zone).strftime(time_format)
    	return pd.to_datetime(time_local, format=time_format)
    except Exception as e:
    	print(epoch_time)
    	return ''


def fetch_rpa_metadata():
	"""
	"""
	print("fetch_rpa_metadata:")


def build_where_clause(filters):
	"""
	"""
	where_clause = ""
	for clause in filters:
		col_name = clause["col_name"]
		operator = clause["operator"]
		value = clause["value"]
		c = "("
		c += str(col_name)
		c += " "
		c += str(operator)
		c += " "
		if isinstance(value, str):
			if (str(operator) == "IN") or (str(operator) == "NOT IN"):
				c += str(value)
			else:
				c += "'"
				c += str(value)
				c += "'"
		else:
			c += str(value)
		c += ") "
		c += "and "
		where_clause += c

		print(where_clause)

	where_clause += "True"

	return where_clause


def get_billing_units():
	"""
	"""
	print("Getting Billing Units")
	# where_clause = "status = 'PURGED'"
	postgreSQL_select_statement = "select distinct billing_state from " + table_document_metadata
	print(postgreSQL_select_statement)
	
	cursor.execute(postgreSQL_select_statement)

	column_names = [i[0] for i in cursor.description]
	
	# records = cursor.fetchall()
	list_records = []
	for row in cursor:
		list_records.append(row)

	DF = pd.DataFrame(list_records, columns=column_names)

	response = {}

	if (DF is not None) or (DF.shape[0] != 0):
		l = list(DF["billing_state"])
		l = [x for x in l if x]
		l.sort()
		response["billing_units"] = l

	return response


def get_vendors():
	"""
	"""
	print("Getting Vendors")
	# where_clause = "status = 'PURGED'"
	postgreSQL_select_statement = "select distinct vendor_name from " + table_document_metadata
	print(postgreSQL_select_statement)
	
	cursor.execute(postgreSQL_select_statement)

	column_names = [i[0] for i in cursor.description]
	
	# records = cursor.fetchall()
	list_records = []
	for row in cursor:
		list_records.append(row)

	DF = pd.DataFrame(list_records, columns=column_names)

	response = {}

	if (DF is not None) or (DF.shape[0] != 0):
		l = list(DF["vendor_name"])
		l = [x for x in l if x]
		# Sort in Alphabetical Order
		l.sort()
		response["vendors"] = l

	return response


def fetch_and_format_document_metadata(filters):
	"""
	"""
	print("fetch_document_metadata:")
	where_clause = build_where_clause(filters)
	print("Built where_clause:", where_clause)
	DF = build_execute_sql_query(table_document_metadata, where_clause)
	print("Records fetched:", DF.shape)

	if DF.shape[0] == 0:
		return DF
	# DF = build_execute_sql_query(table_document_metadata, "status = 'REVIEW'")
	try:
		# DF["review_completion_or_deletion_time_timezone"] = DF["review_completion_or_deletion_time"].apply(convert_epoch_time_zone)
		DF["submitted_on_timezone"] = DF["submitted_on"].apply(convert_epoch_time_zone)
		# DF["extraction_completion_time_timezone"] = DF["extraction_completion_time"].apply(convert_epoch_time_zone)

		DF["submit_date"] = DF["submitted_on_timezone"].dt.date
		DF["submit_time"] = DF["submitted_on_timezone"].dt.time
		DF["submit_day"] = DF["submitted_on_timezone"].dt.dayofweek
		DF["submit_hour"] = DF["submitted_on_timezone"].dt.hour


		DF["review_completion_or_deletion_time"] = pd.to_numeric(DF["review_completion_or_deletion_time"])
		DF["submitted_on"] = pd.to_numeric(DF["submitted_on"])
		DF["extraction_completion_time"] = pd.to_numeric(DF["extraction_completion_time"])
		DF["pages"] = pd.to_numeric(DF["pages"])
		DF["total_review_time"] = pd.to_numeric(DF["total_review_time"])
		DF["paiges_confidence"] = pd.to_numeric(DF["paiges_confidence"])
		DF["paiges_accuracy"] = pd.to_numeric(DF["paiges_accuracy"])
		DF["calculated_accuracy_paiges"] = pd.to_numeric(DF["calculated_accuracy_paiges"])
		DF["calculated_accuracy_client"] = pd.to_numeric(DF["calculated_accuracy_client"])
		DF["quality_score"] = pd.to_numeric(DF["quality_score"])
		DF["sla_flag"] = pd.to_numeric(DF["sla_flag"])
	except Exception as e:
		print(e)
		traceback.print_exc()
	
	return DF


def fetch_and_format_rpa_metadata(filters):
	"""
	"""
	DF = None
	print("Fetching Posting Data:")
	where_clause = build_where_clause(filters)
	print("Built where_clause:", where_clause)
	DF = build_execute_sql_query(table_rpa_metadata, where_clause)
	print("Records fetched:", DF.shape)

	if DF.shape[0] == 0:
		return DF

	try:
		print("Inside formatting columns fetch_and_format_rpa_metadata")
		DF["rpa_receive_time"] = DF["rpa_receive_time"].astype(int)
		DF["extraction_completion_time"] = DF["extraction_completion_time"].astype(float)

		DF["sent_to_approval_on"] = DF["sent_to_approval_on"].astype('Int64')
		DF["sent_to_approval_on"] = DF["sent_to_approval_on"].apply(add_millisecond_epoch)
		DF["approved_on"] = DF["approved_on"].astype('Int64')
		DF["approved_on"] = DF["approved_on"].apply(add_millisecond_epoch)
		# DF["review_completion_or_deletion_time_timezone"] = DF["review_completion_or_deletion_time"].apply(convert_epoch_time_zone)
		DF["rpa_receive_time_timezone"] = DF["rpa_receive_time"].apply(convert_epoch_time_zone)
		# DF["extraction_completion_time_timezone"] = DF["extraction_completion_time"].apply(convert_epoch_time_zone)

		DF["rpa_receive_date"] = DF["rpa_receive_time_timezone"].dt.date
		DF["rpa_receive_tm"] = DF["rpa_receive_time_timezone"].dt.time
		DF["rpa_receive_day"] = DF["rpa_receive_time_timezone"].dt.dayofweek
		DF["rpa_receive_hour"] = DF["rpa_receive_time_timezone"].dt.hour

		# Clean rpa_posting_comment
		DF['rpa_posting_comment'] = DF['rpa_posting_comment'].str.upper()
		DF['rpa_posting_comment'] = DF['rpa_posting_comment'].str.replace("'","")
		DF.loc[DF['rpa_posting_comment'].str.contains("CODE", na=False), 'rpa_posting_comment'] = "OTHERS"
		DF.loc[DF['rpa_posting_comment'].str.contains("OTHERS", na=False), 'rpa_posting_comment'] = "OTHERS"
	except Exception as e:
		print(e)
		traceback.print_exc()

	return DF


def fetch_document_result():
	"""
	"""
	print("fetch_document_result:")


def populate_dashboard(list_charts, filters, chart_tab):
	"""
	"""
	# Get Data
	response = {}
	response["chart_data"] = {}
	# extraction_forever_summary()
	# return response
	DF = None
	try:
		if chart_tab == "EXTRACTION":
			DF = fetch_and_format_document_metadata(filters)
		elif chart_tab == "POSTING":
			DF = fetch_and_format_rpa_metadata(filters)
	except Exception as e:
		print(e)
		traceback.print_exc()
		conn.rollback()
		return response

	if (DF is None) or (DF.shape[0] == 0):
		return response

	chart_data = {}
	for chart in list_charts:
		try:
			function_name = getattr(graph_functions, chart)
			res = function_name(DF)
			chart_data[chart] = res
		except Exception as e:
			print(e)
			traceback.print_exc()
			conn.rollback()
			pass
		

	response["chart_data"] = chart_data

	return response


# @app.route('/populate_dashboard/get_vendors', methods=['POST'])
def get_vendors_wrapper(request):
	"""
	"""
	response_object = {}
	try:
		print("Request received!!!")
		# rawContent = request.content.read()
		# encodedContent = rawContent.decode("utf-8")
		# content = json.loads(encodedContent)
		content = request
		print(content)

		response_object = get_vendors()

		response_object['status'] = "Success"
		response_object['responseCode'] = 200
	except Exception as e:
		print(e)
		traceback.print_exc()
		conn.rollback()
		response_object['status'] = "Failure"
		response_object['responseCode'] = 500
		response_object['message'] = "Failure"
		pass

	# request.responseHeaders.addRawHeader(b"content-type", b"application/json")
	response = json.dumps(response_object, indent=4, sort_keys=False, default=str)
	return response


# @app.route('/populate_dashboard/get_billing_units', methods=['POST'])
def get_billing_units_wrapper(request):
	"""
	"""
	response_object = {}
	try:
		print("Request received!!!")
		# rawContent = request.content.read()
		# encodedContent = rawContent.decode("utf-8")
		# content = json.loads(encodedContent)
		content = request
		print(content)

		response_object = get_billing_units()

		response_object['status'] = "Success"
		response_object['responseCode'] = 200
	except Exception as e:
		print(e)
		traceback.print_exc()
		conn.rollback()
		response_object['status'] = "Failure"
		response_object['responseCode'] = 500
		response_object['message'] = "Failure"
		pass

	# request.responseHeaders.addRawHeader(b"content-type", b"application/json")
	response = json.dumps(response_object, indent=4, sort_keys=False, default=str)
	return response


# @app.route('/populate_dashboard', methods=['POST'])
def populate_dashboard_wrapper(request):
	"""
	"""
	response_object = {}
	try:
		print("Request received!!!")
		# rawContent = request.content.read()
		# encodedContent = rawContent.decode("utf-8")
		# content = json.loads(encodedContent)
		content = request
		print(content)

		if "time_zone" in content:
			global time_zone_code, time_zone
			time_zone_code = content["time_zone"]
			time_zone = pytz.timezone(time_zone_code)

		list_charts = content["list_charts"]
		filters = content["filters"]
		chart_tab = content["chart_tab"]

		response_object = populate_dashboard(list_charts, filters, chart_tab)

		response_object['status'] = "Success"
		response_object['responseCode'] = 200
	except Exception as e:
		print(e)
		traceback.print_exc()
		conn.rollback()
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

	populate_dashboard_wrapper()


if __name__ == "__main__":
    # main()
    app.run("0.0.0.0", 2222)


