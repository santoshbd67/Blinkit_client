import pandas as pd
import TAPPconfig as cfg
pd.options.mode.chained_assignment = None



def read_refernce_data(refrence_file_path):
	"""
	"""
	print("Reading:", rpa_stats_path)
	xls = pd.ExcelFile(refrence_file_path)
	GRN_STATS = pd.read_excel(xls, 'GRN_Stats')
	PROCESSING_STATS = pd.read_excel(xls, 'Processing_Stats')
	
	return GRN_STATS, PROCESSING_STATS


# def generate_rpa_stats_table():
# 	"""
# 	"""
# 	print("Generating RPA Stats:")
# 	GRN_STATS, PROCESSING_STATS  = read_refernce_data(rpa_stats_path)
# 	GRN_STATS["GRN_Date"] = pd.to_datetime(GRN_STATS["GRN_Date"])
# 	PROCESSING_STATS["GRN_Date"] = pd.to_datetime(PROCESSING_STATS["GRN_Date"])
# 	PROCESSING_STATS["Processing_Date"] = pd.to_datetime(PROCESSING_STATS["Processing_Date"])
# 	print(GRN_STATS)
# 	print(PROCESSING_STATS)
# 	GRN_STATS.fillna(0, inplace = True)
# 	# PROCESSING_STATS.fillna(0, inplace = True)

# 	GRN_STATS["Status_Reason"] = GRN_STATS["Status_Reason"].str.replace(". To Be Processed Manually.", "")
# 	GRN_STATS["Status_Reason"] = GRN_STATS["Status_Reason"].str.replace(".", "")
# 	GRN_STATS["Status_Reason"] = GRN_STATS["Status_Reason"].str.strip()
# 	GRN_STATS.sort_values(by='GRN_Date', inplace = True)
# 	GRN_STATS["GRN_Date"] = GRN_STATS["GRN_Date"].dt.date

# 	DF_1 = GRN_STATS.groupby(["GRN_Date", "Status_Reason"])[["Document_Count"]].sum().reset_index()

# 	TEMP = DF_1.pivot(index='Status_Reason', columns='GRN_Date', values='Document_Count').reset_index()

# 	all_index = TEMP.index.tolist()
# 	total_index = TEMP.index[TEMP["Status_Reason"] == "Total"].tolist()

# 	reordered_index = [x for x in all_index if x not in total_index]
# 	reordered_index = total_index + reordered_index

# 	TEMP = TEMP.reindex(reordered_index)
# 	TEMP.rename(columns={'Status_Reason': 'Category/GRN Date'}, inplace=True)


# 	PROCESSING_STATS.sort_values(by = ["Processing_Date", 'GRN_Date'], inplace = True)
# 	DF_2 = PROCESSING_STATS[["Processing_Date", "S3_Document_Count"]].drop_duplicates()
# 	DF_2["Processing_Date"] = DF_2["Processing_Date"].dt.date
# 	# DF_2.fillna(0, inplace = True)
# 	DF_2.rename(columns={'S3_Document_Count': 'Documents Downloaded', 'Processing_Date': 'Processing Date'}, inplace=True)
	
# 	TEMP_1 = DF_2.set_index('Processing Date').T
# 	TEMP_1 = TEMP_1.reset_index()
# 	TEMP_1.rename(columns={'index': 'Processing Date'}, inplace=True)
# 	# PROCESSING_STATS["GRN_Date"] = PROCESSING_STATS["GRN_Date"].dt.date
# 	# PROCESSING_STATS["Processing_Date"] = PROCESSING_STATS["Processing_Date"].dt.date


# 	# PROCESSING_STATS['Date_Diff'] = (PROCESSING_STATS['GRN_Date'] - PROCESSING_STATS['Processing_Date']).dt.days

# 	# PROCESSING_STATS["Processing_Date"] = PROCESSING_STATS["Processing_Date"].dt.date

# 	# DF_2 = PROCESSING_STATS.groupby(["Processing_Date", "Date_Diff"])[["S3_Document_Count", "Document_Processed"]].sum().reset_index()

# 	# TEMP_1 = DF_2.pivot(index='Date_Diff', columns='Processing_Date', values='Document_Processed').reset_index()

# 	print(TEMP_1)
# 	return TEMP, TEMP_1


def derive_columns(DF):
    try:
        DF["Submitted On"] = pd.to_datetime(DF["Submitted On"], format='%d/%m/%Y %H:%M:%S')
    except:
        DF["Submitted On"] = pd.to_datetime(DF["Submitted On"], format='%d/%m/%y %H:%M')
        pass
    try:
        DF["Review Completion/Deletion Time"] = pd.to_datetime(
            DF["Review Completion/Deletion Time"], format='%d/%m/%Y %H:%M:%S')
    except:
        DF["Review Completion/Deletion Time"] = pd.to_datetime(
            DF["Review Completion/Deletion Time"], format='%d/%m/%y %H:%M')
        pass
    DF["Submit Date"] = DF["Submitted On"].dt.date
    DF["Submit Time"] = DF["Submitted On"].dt.time
    DF["Submit Day"] = DF["Submitted On"].dt.dayofweek
    DF["Submit Hour"] = DF["Submitted On"].dt.hour
    DF["STP System"].fillna(False, inplace=True)

    DF["Review Completion Date"] = DF["Review Completion/Deletion Time"].dt.date
    DF["Review Completion Time"] = DF["Review Completion/Deletion Time"].dt.hour
    
    DF["Status_"] = "OTHER"
    DF.loc[DF['Status'].isin(["REVIEW", "REVIEW_COMPLETED"]),
         'Status_'] = DF["Status"]
    
    return DF


def document_review_status(DF):
	"""
	"""

	DF = DF.loc[DF["Document Type"] == "Invoice"]

	TEMP = DF.groupby(["Submit Date"])[["Document ID"]].count().reset_index()
	TEMP.rename(columns={'Document ID': 'Total Documents Successfully Uploaded'}, inplace=True)

	B = DF.loc[DF["Status"] != "FAILED"].groupby(["Submit Date"])[["Document ID"]].count().reset_index()
	B.rename(columns={'Document ID': 'Total Documents Extracted'}, inplace=True)
	TEMP = pd.merge(TEMP, B, on=["Submit Date"], how="outer")

	C = DF.loc[DF["Status"] == "FAILED"].groupby(["Submit Date"])[["Document ID"]].count().reset_index()
	C.rename(columns={'Document ID': 'Extraction Failures'}, inplace=True)
	TEMP = pd.merge(TEMP, C, on=["Submit Date"], how="outer")

	D = DF.loc[(DF["Status"] != "FAILED") & (DF["Document Type"] == "Invoice")].groupby(["Submit Date"])[["Document ID"]].count().reset_index()
	D.rename(columns={'Document ID': 'Total Invoices Extracted'}, inplace=True)
	TEMP = pd.merge(TEMP, D, on=["Submit Date"], how="outer")

	E = DF.loc[(DF["Status"] != "FAILED") & (DF["Document Type"] == "Discrepancy Note")].groupby(["Submit Date"])[["Document ID"]].count().reset_index()
	E.rename(columns={'Document ID': 'Total Discrepancy Note Extracted'}, inplace=True)
	TEMP = pd.merge(TEMP, E, on=["Submit Date"], how="outer")

	F = DF.loc[(DF["Status"] == "REVIEW_COMPLETED") | (DF["Status"].str.contains("RPA"))].groupby(["Submit Date"])[["Document ID"]].count().reset_index()
	F.rename(columns={'Document ID': 'Documents Review Completed'}, inplace=True)
	TEMP = pd.merge(TEMP, F, on=["Submit Date"], how="outer")

	G = DF.loc[(DF["Status"] == "DELETED")].groupby(["Submit Date"])[["Document ID"]].count().reset_index()
	G.rename(columns={'Document ID': 'Documents Rejected'}, inplace=True)
	TEMP = pd.merge(TEMP, G, on=["Submit Date"], how="outer")

	H = DF.loc[(~DF["Reassign Reason"].isna())].groupby(["Submit Date"])[["Document ID"]].count().reset_index()
	H.rename(columns={'Document ID': 'Documents Reassigned to Pierian'}, inplace=True)
	TEMP = pd.merge(TEMP, H, on=["Submit Date"], how="outer")

	I = DF.loc[(DF["Status"] == "REVIEW")].groupby(["Submit Date"])[["Document ID"]].count().reset_index()
	I.rename(columns={'Document ID': 'Documents Review Pending'}, inplace=True)
	TEMP = pd.merge(TEMP, I, on=["Submit Date"], how="outer")

	TEMP.rename(columns={'Submit Date': 'Processing Date'}, inplace=True)
	
	TEMP = TEMP.set_index('Processing Date').T
	TEMP = TEMP.reset_index()
	TEMP.rename(columns={'index': 'Processing Date'}, inplace=True)
	TEMP.fillna(0, inplace = True)
	return TEMP


def document_posting_status(DF):
	"""
	"""
	DF = DF.loc[DF["Document Type"] == "Invoice"]

	TEMP = DF.loc[(DF["Status"] == "RPA_FAILED") & (DF['Status Msg'].str.contains("Po didnt match"))].groupby(["Submit Date"])[["Document ID"]].count().reset_index()
	TEMP.rename(columns={'Document ID': '2-Way Match Failure: PO Mismatch'}, inplace=True)

	A = DF.loc[(DF["Status"] == "RPA_FAILED") & (DF['Status Msg'].str.contains("Amounts Mismatch"))].groupby(["Submit Date"])[["Document ID"]].count().reset_index()
	A.rename(columns={'Document ID': '2-Way Match Failure: Amount Mismatch'}, inplace=True)
	TEMP = pd.merge(TEMP, A, on=["Submit Date"], how="outer")


	B = DF.loc[(DF["Status"].str.contains("RPA")) & (~DF['Status Msg'].str.contains("Po didnt match")) 
	& (~DF['Status Msg'].str.contains("Amounts Mismatch"))].groupby(["Submit Date"])[["Document ID"]].count().reset_index()
	B.rename(columns={'Document ID': '2-Way Match Success'}, inplace=True)
	TEMP = pd.merge(TEMP, B, on=["Submit Date"], how="outer")

	C = DF.loc[(DF["Status"] == "RPA_PROCESSED")].groupby(["Submit Date"])[["Document ID"]].count().reset_index()
	C.rename(columns={'Document ID': 'SAP Posting Success'}, inplace=True)
	TEMP = pd.merge(TEMP, C, on=["Submit Date"], how="outer")

	D = DF.loc[(DF["Status"] == "RPA_FAILED") & (~DF['Status Msg'].str.contains("Po didnt match")) 
	& (~DF['Status Msg'].str.contains("Amounts Mismatch"))].groupby(["Submit Date"])[["Document ID"]].count().reset_index()
	D.rename(columns={'Document ID': 'SAP Posting Failed'}, inplace=True)
	TEMP = pd.merge(TEMP, D, on=["Submit Date"], how="outer")


	submit_date = TEMP['Submit Date']
	TEMP.drop(labels=['Submit Date'], axis=1,inplace = True)
	TEMP.insert(0, 'Submit Date', submit_date)

	TEMP.rename(columns={'Submit Date': 'Processing Date'}, inplace=True)
	
	TEMP = TEMP.set_index('Processing Date').T
	TEMP = TEMP.reset_index()
	TEMP.rename(columns={'index': 'Processing Date'}, inplace=True)
	TEMP.fillna(0, inplace = True)


	print(TEMP)
	return TEMP


def posting_failed_status(DF):
	"""
	"""
	DF = DF.loc[DF["Document Type"] == "Invoice"]

	S = DF.loc[(DF["Status"] == "RPA_FAILED") & (~DF['Status Msg'].str.contains("Po didnt match")) 
	& (~DF['Status Msg'].str.contains("Amounts Mismatch"))]

	# SAP Error Categorization
	S["SAP Posting Failure Reason"] = "Generic"
	S.loc[S['Status Msg'].str.lower().str.contains("vendor master"), "SAP Posting Failure Reason"] = "Vendor Master Data Missing"
	S.loc[S['Status Msg'].str.lower().str.contains("location master"), "SAP Posting Failure Reason"] = "Location Master Data Missing"
	S.loc[S['Status Msg'].str.lower().str.contains("withholding tax code 48i"), "SAP Posting Failure Reason"] = "Withholding tax code 48I"
	S.loc[S['Status Msg'].str.lower().str.contains("invoice date format"), "SAP Posting Failure Reason"] = "Invalid Invoice Date format"
	S.loc[S['Status Msg'].str.lower().str.contains("is inactive"), "SAP Posting Failure Reason"] = "Vendor Inactive"
	S.loc[S['Status Msg'].str.lower().str.contains("date deviates"), "SAP Posting Failure Reason"] = "Date deviates from permissible range"
	
	
	TEMP = S.groupby(["Submit Date", "SAP Posting Failure Reason"])[["Document ID"]].count().reset_index()

	TEMP.rename(columns={'Submit Date': 'Processing Date'}, inplace=True)
	
	TEMP = TEMP.pivot(index='SAP Posting Failure Reason', columns='Processing Date', values='Document ID').reset_index()

	TEMP.fillna(0, inplace = True)


	print(TEMP)
	return TEMP


def generate_client_report(DF, name_extender):
	"""
	"""
	summary_report_file_name = 'Reports/Client_Report.xlsx'

	if name_extender is not None:
		summary_report_file_name = 'Reports/Client_Report_' + str(name_extender) + '.xlsx'

	DF = derive_columns(DF)

	writer = pd.ExcelWriter(summary_report_file_name, engine='xlsxwriter')
	workbook=writer.book
	section_header_format = workbook.add_format()
	section_header_format.set_bold()
	section_header_format.set_font_size(20)
	section_header_format.set_font_color('#27AE60')

	table_header_format = workbook.add_format()
	table_header_format.set_bold()
	section_header_format.set_font_size(15)
	table_header_format.set_font_color('#A93226')
	worksheet=workbook.add_worksheet('Result')
	writer.sheets['Result'] = worksheet

	start_row = 1
	start_col = 0
	# worksheet.write_string(start_row, start_col, "GRN/S3 STATS", section_header_format)
	# start_row = start_row + 1
    

	# TEMP, TEMP_1 = generate_rpa_stats_table()

	# TEMP.name = "GRN Status"
	# TEMP_1.name = "S3 Status"

	# worksheet.write_string(start_row, start_col, TEMP.name, table_header_format)
	# start_row = start_row + 1
	# TEMP.to_excel(writer,sheet_name='Result', startrow=start_row , startcol=start_col, index=False)
	# start_row = start_row + TEMP.shape[0] + 2

	# worksheet.write_string(start_row, start_col, TEMP_1.name, table_header_format)
	# start_row = start_row + 1
	# TEMP_1.to_excel(writer,sheet_name='Result', startrow=start_row , startcol=start_col, index=False)
	# start_row = start_row + TEMP_1.shape[0] + 2

	worksheet.write_string(start_row, start_col, "Documents Extraction/Review STATS", section_header_format)
	start_row = start_row + 1

	TEMP = document_review_status(DF)
	TEMP.name = "Document Status: Extraction/Review"

	worksheet.write_string(start_row, start_col, TEMP.name, table_header_format)
	start_row = start_row + 1
	TEMP.to_excel(writer,sheet_name='Result',
                            startrow=start_row , startcol=start_col, index=False)
	start_row = start_row + TEMP.shape[0] + 2

	worksheet.write_string(start_row, start_col, "Documents Posting STATS", section_header_format)
	start_row = start_row + 1

	TEMP = document_posting_status(DF)

	TEMP.name = "Document Status: Posting"

	worksheet.write_string(start_row, start_col, TEMP.name, table_header_format)
	start_row = start_row + 1
	TEMP.to_excel(writer,sheet_name='Result',
                            startrow=start_row , startcol=start_col, index=False)
	start_row = start_row + TEMP.shape[0] + 2

	TEMP = posting_failed_status(DF)

	TEMP.name = "SAP Posting Failures"

	worksheet.write_string(start_row, start_col, TEMP.name, table_header_format)
	start_row = start_row + 1
	TEMP.to_excel(writer,sheet_name='Result',
                            startrow=start_row , startcol=start_col, index=False)
	start_row = start_row + TEMP.shape[0] + 2

	writer.close()