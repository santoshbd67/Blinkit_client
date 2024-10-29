# import requests
# from io import StringIO
# from io import BytesIO
import pandas as pd
import os
# import time
from datetime import date as dt
import copy
import sys
old_stdout = sys.stdout
old_stderr = sys.stderr
log_file_path = os.path.join("files","update_masterdata_log.txt")
log_file = open(log_file_path,"w")
sys.stdout = log_file
sys.stderr = log_file


import smtplib
import traceback
import base64
from email.message import EmailMessage

key = "cGFybWVzaHdhci5iaGFud2FsZUB0YW9hdXRvbWF0aW9uLmNvbQ=="
val = "X19QYXJtZXNoQFRBTw=="

import smtplib
import traceback
import base64
from email.message import EmailMessage

key = "cGFybWVzaHdhci5iaGFud2FsZUB0YW9hdXRvbWF0aW9uLmNvbQ=="
val = "X19QYXJtZXNoQFRBTw=="

def send_email(eFrom: str = None,
               eTo: str = None,
               subject: str = None,
               files: list = [],
              content: str = None,
              attachment: bool = False) -> str or None:
    message = EmailMessage()
    message['Subject'] = subject if subject else "Important: Data Mismatch Report"
    message['From'] = eFrom if eFrom else 'parmeshwar.bhanwale@taoautomation.com'
    message['Reply-to'] = eFrom if eFrom else 'parmeshwar.bhanwale@taoautomation.com'
    send_list = ['swaroop.samavedam@taoautomation.com', 'harika.reddy@taoautomation.com', 'sahil.aggarwal@taoautomation.com']
    # send_list = ['sahil.aggarwal@taoautomation.com']
    message['To'] = send_list
    # message['To'] = ['sahil.aggarwal@taoautomation.com']
    if content == None:
        content = 'Dear Swaroop,\n\nPlease find attached the data mismatch report.\n\nBest regards,\nSahil'
    message.set_content(content)
    if attachment == True:
        for file in files:
            with open(file, 'rb') as fileObj:
                file_data = fileObj.read()
                message.add_attachment(file_data, maintype="text/csv", subtype='csv', filename=file)
                print("attached file to message:", file)
    try:
        with smtplib.SMTP('smtp-mail.outlook.com', 587) as server:
            server.ehlo()
            server.starttls()
            server.login(base64.b64decode(key).decode(), base64.b64decode(val).decode())
            for person in send_list: 
                server.sendmail(message['From'], person, str(message.as_string()))
            return True

    except Exception as e:
        print("first exception:", traceback.format_exc())
        try:
            with smtplib.SMTP_SSL('smtp-mail.outlook.com', 465) as server:
                server.ehlo()
                server.starttls()
                server.login(base64.b64decode(key).decode(), base64.b64decode(val).decode())
                server.sendmail(message['From'], message['To'], message.as_string())
                return True
        except Exception as e:
            print("Send email failed:", traceback.format_exc())
            return False

    return


# Example usage


def check_vendor_code(DROP_VENDOR_MASTER):
    DROP_VENDOR_MASTER['duplicated'] = 0
    DROP_VENDOR_MASTER_COPY = DROP_VENDOR_MASTER.copy(deep = True)
    try:
        duplicates = DROP_VENDOR_MASTER[(DROP_VENDOR_MASTER['VENDOR_GSTIN'].duplicated(keep=False)) & (~DROP_VENDOR_MASTER['VENDOR_GSTIN'].isnull())]
        DROP_VENDOR_MASTER.loc[DROP_VENDOR_MASTER['VENDOR_GSTIN'].isin(duplicates['VENDOR_GSTIN']), 'duplicated'] = 1
        return DROP_VENDOR_MASTER
    except Exception as e:
        print("Exception occured in checking unique id:", e)
        return DROP_VENDOR_MASTER_COPY
    


def check_unique_id(appended_data_df,previous_data_df, data_modified, unique_column):
    DROP_VENDOR_MASTER_COPY = DROP_VENDOR_MASTER.copy(deep = True)
    deep_copy_data_modified = copy.deepcopy(data_modified)
    try:
        duplicate_ids = appended_data_df[unique_column].duplicated(keep=False)
        print(f"Found Duplicates in {unique_column}: {list(duplicate_ids).count(True)}")
        data_to_append = []
        for index, row in appended_data_df.iterrows():
            if not (pd.isna(row[unique_column])) and duplicate_ids[index]:
            # if duplicate_ids[index]:
                ## only contains data dor duplicated rows
                id_value = row[unique_column]
                # print(id_value)
                vendor_id = row["VENDOR_ID"]
                vendor_gstin = row["VENDOR_GSTIN"]
                previous_row = previous_data_df[previous_data_df[unique_column] == id_value]
                if not any(data[unique_column] == id_value for data in data_to_append):
                    ## Only duplicate entry if data is not already updated 
                    if not previous_row.empty:
                        # Use the row from the previous data DataFrame
                        # print("Duplicated row, using previous masterdata's row")
                        data_modified["Vendor_id"].append(vendor_id)
                        data_modified["Vendor_name"].append(row["VENDOR_NAME"])
                        if unique_column == "VENDOR_ID":
                            error_msg = f"There are same Vendor ID ({vendor_id}) with different GSTIN, since data is present in current masterdata we are not updating this entry"
                        else:
                            error_msg = f"There are same GSTIN ({vendor_gstin}) with different Vendor ID's, since data is present in current masterdata we are not updating this entry"
                        data_modified["Comments"].append(error_msg)
                        data_to_append.append(previous_row.loc[previous_row.index[0], ['S No', 'VENDOR_NAME', 'VENDOR_ID', 'VENDOR_GSTIN', 'Wtax Code', 'State Code']])

                    else:
                        # If data is not present in previous Dataframe, use it from new dataframe(appended)
                        # print("Duplicated row, using new masterdata's row since data for that id is not present in old masterdata")
                        data_modified["Vendor_id"].append(vendor_id)
                        data_modified["Vendor_name"].append(row["VENDOR_NAME"])
                        if unique_column == "VENDOR_ID":
                            error_msg = f"There are same Vendor ID ({vendor_id}) with different GSTIN, since data is not present in current masterdata we are updating this entry with first entry in new masterdata"
                        else:
                            error_msg = f"There are same GSTIN ({vendor_gstin}) with different Vendor ID's, since data is not present in current masterdata we are updating this entry with first entry in new masterdata"
                        data_modified["Comments"].append(error_msg)
                        data_to_append.append(row.loc[['S No', 'VENDOR_NAME', 'VENDOR_ID', 'VENDOR_GSTIN', 'Wtax Code', 'State Code']])

            else:
                # If the ID is unique or no matching ID found in the previous data, use the row from the appended data DataFrame
                data_to_append.append(row.loc[['S No', 'VENDOR_NAME', 'VENDOR_ID', 'VENDOR_GSTIN', 'Wtax Code', 'State Code']])

        # Step 3: Create a new DataFrame with the data to append
        data_to_append_df = pd.DataFrame(data_to_append)
        #data_to_append_df.to_csv(r"C:\Users\Admin\Desktop\test1.csv",index = False)
        #print(data_to_append_df.shape)
        return data_to_append_df, data_modified
    except Exception as e:
        print("Exception occured in checking unique id:", e)
        return DROP_VENDOR_MASTER_COPY, deep_copy_data_modified

def check_gstin(DROP_VENDOR_MASTER, CURRENT_VENDOR_MASTER, data_modified):
    DROP_VENDOR_MASTER_COPY = DROP_VENDOR_MASTER.copy(deep = True)
    deep_copy_data_modified = copy.deepcopy(data_modified)
    try:
        for idx, row in DROP_VENDOR_MASTER.iterrows():
            drop_vendor_id = row["VENDOR_ID"]
            ab = CURRENT_VENDOR_MASTER.loc[CURRENT_VENDOR_MASTER["VENDOR_ID"] == drop_vendor_id, "VENDOR_GSTIN"]
            # Convert ab to a string to remove index and get the actual value if it exists
            if ab.shape[0] > 0:
                current_vendor_gstin = str(ab.values[0]).upper() if not ab.empty else "Not Found"
                drop_vendor_gstin = str(row["VENDOR_GSTIN"]).upper()
                if current_vendor_gstin != "Not Found":
                    # print(current_vendor_gstin, drop_vendor_gstin)
                    if current_vendor_gstin != drop_vendor_gstin:
                        print("GSTIN not matching, keeping the previous data")
                        data_modified["Vendor_id"].append(drop_vendor_id)
                        data_modified["Vendor_name"].append(row["VENDOR_NAME"])
                        error_msg = f"Vendor GSTIN is updated for Vendor ID '{drop_vendor_id}', so we are using data from current masterdata"
                        data_modified["Comments"].append(error_msg)
                        DROP_VENDOR_MASTER.at[idx, "VENDOR_GSTIN"] = current_vendor_gstin
                    else:
                        # print("GSTIN matched")
                        pass
        return DROP_VENDOR_MASTER, data_modified
    except Exception as e:
        print("Exception occured in checking gstin", e)
        return DROP_VENDOR_MASTER_COPY, deep_copy_data_modified

def remove_duplicates(DROP_VENDOR_MASTER):
    DROP_VENDOR_MASTER_COPY = DROP_VENDOR_MASTER.copy(deep = True)
    try:
        # DROP_VENDOR_MASTER = pd.read_csv(r"C:\Users\Admin\Desktop\test_drop.csv")
        # print("Initial shape of masterdata", DROP_VENDOR_MASTER.shape)
        print("before", DROP_VENDOR_MASTER.shape)
        df_with_gstin = DROP_VENDOR_MASTER[DROP_VENDOR_MASTER['VENDOR_GSTIN'].notna()]
        df_without_gstin = DROP_VENDOR_MASTER[DROP_VENDOR_MASTER['VENDOR_GSTIN'].isna()]
        print("different gstin shape",df_with_gstin.shape, df_without_gstin.shape)
        subset_columns_with_gstin = ['VENDOR_ID', 'VENDOR_GSTIN']
        subset_columns_without_gstin = ['VENDOR_ID', 'VENDOR_NAME']
        
        duplicates_mask_with_gstin = df_with_gstin.duplicated(subset=subset_columns_with_gstin)
        duplicates_mask_without_gstin = df_without_gstin.duplicated(subset=subset_columns_without_gstin)
        
        filtered_df_with_gstin = df_with_gstin[~duplicates_mask_with_gstin]
        filtered_df_without_gstin = df_without_gstin[~duplicates_mask_without_gstin]
        print("after filtering shape is:", filtered_df_with_gstin.shape, filtered_df_without_gstin.shape)
        filtered_df = pd.concat([filtered_df_with_gstin, filtered_df_without_gstin], ignore_index=True)
        # duplicates_mask = DROP_VENDOR_MASTER.duplicated(subset=subset_columns)
        print("removed gstin's row is:",df_with_gstin[duplicates_mask_with_gstin])
        print("removed gstin's row is",df_without_gstin[duplicates_mask_without_gstin])
        # filtered_df = DROP_VENDOR_MASTER[~duplicates_mask]
        print("Final shape of masterdata after removing duplicate is:", filtered_df.shape)
        return filtered_df
    except Exception as e:
        print("Exception occured in remove duplicates:", e)
        return DROP_VENDOR_MASTER_COPY

def validate_masterdata(DROP_VENDOR_MASTER,CURRENT_VENDOR_MASTER,data_modified):
    DROP_VENDOR_MASTER_COPY = DROP_VENDOR_MASTER.copy(deep = True)
    try:
        # Condition 1 (1.	If duplicate vendor codes are present, an alert will be raised, and the affected record will not be updated.)
        print("At start dropped vendor masterdata shape is:", DROP_VENDOR_MASTER.shape)
        # Condition 5 Removing duplicates from the dataframe
        DROP_VENDOR_MASTER = remove_duplicates(DROP_VENDOR_MASTER)
        print("After removing duplicates drop masterdata shape is:", DROP_VENDOR_MASTER.shape)

        # To be commented
        # DROP_VENDOR_MASTER, data_modified = check_unique_id(DROP_VENDOR_MASTER, CURRENT_VENDOR_MASTER, data_modified, unique_column = "VENDOR_ID")
        # print("After checking duplicate vendor code condition drop masterdata shape is:", DROP_VENDOR_MASTER.shape)
        # Condition 2 (2.	In case different vendor codes are present for the same vendor GSTIN, an alert will be raised, and the record will not be updated.)
        # DROP_VENDOR_MASTER,data_modified = check_unique_id(DROP_VENDOR_MASTER,CURRENT_VENDOR_MASTER, data_modified, unique_column = "VENDOR_GSTIN")
        
        # Condition 2 Modified (2.	In case different vendor codes are present for the same vendor GSTIN, an alert will be raised, and the flag will be updated.)
        DROP_VENDOR_MASTER = check_vendor_code(DROP_VENDOR_MASTER)
        
        print("After checking same vendor GSTIN condition drop masterdata shape is:", DROP_VENDOR_MASTER.shape)
        # Condition 4 (If Vendor GSTIN is updated for any vendor code, or vice versa, an alert will be raised, and the record will not be updated.)
        # To be commented
        # DROP_VENDOR_MASTER, data_modified = check_gstin(DROP_VENDOR_MASTER,CURRENT_VENDOR_MASTER, data_modified)
        # print("After checking if vendor GSTIN is updated for vendor code, drop masterdata shape is:", DROP_VENDOR_MASTER.shape)
        # df_data_modified = pd.DataFrame(data_modified)
        # if df_data_modified.shape[0]>0:
        #     df_data_modified.to_csv(r"data_mismatch_report.csv", index= False)
        #     files = [r"data_mismatch_report.csv"]
        #     send_email(files= files, attachment= True)
        #     print("Mail sent") 
        # else:
        #     print("Data is correct in new masterdata file")
        #     send_email(content= "Dear Swaroop,\n\nValidations for Masterdata was correct.\n\nBest regards,\nSahil")
        #     print("Mail sent") 
        # # DROP_VENDOR_MASTER.to_csv(r"C:\Users\Admin\Desktop\test_updated.csv", index= False)
        return DROP_VENDOR_MASTER, True
    except Exception as e:
        print("Exception occured in validating masterdata :", e)
        return DROP_VENDOR_MASTER_COPY, False


def check_column_presence(list1, list2):
    set_list2 = set(list2)
    result = all(element in set_list2 for element in list1)
    return result

# drop_masterdata_file = r"C:\Users\Admin\Desktop\VENDOR_ADDRESS_MASTERDATA_DROP.xlsx"
# current_masterdata_file = r"C:\Users\Admin\Desktop\VENDOR_ADDRESS_MASTERDATA_CURRENT.csv"
# drop_masterdata_full_path,current_masterdata_full_path,backup_masterdata_original_full_path,backup_masterdata_transformd_full_path
def check_file_presence(file_path1,file_path2):
    if os.path.exists(file_path1) and os.path.exists(file_path2):
        print("The file is present in the folder.")
        return True
    else:
        print("The file is not found in the folder.")
        return False

def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print("File removed successfully")   
        # input("Enter something to proceed")
# print(DROP_VENDOR_MASTER.columns)
if __name__ == '__main__':
    data_modified = {"Vendor_id": [], "Vendor_name" : [], "Comments":[]}
    today = str(dt.today())
    # print(today)
    drop_masterdata_path = "/afs/blinkitnfs/Drop"
    current_masterdata_path = "/afs/blinkitnfs/MasterData"
    backup_mastardata_path  = "/afs/blinkitnfs/Backup"

    drop_masterdata_file = "VENDOR_ADDRESS_MASTERDATA.xlsx"
    current_masterdata_file = "VENDOR_ADDRESS_MASTERDATA.csv"
    backup_masterdata_file_original = today + "_" + "VENDOR_ADDRESS_MASTERDATA_ORIGINAL.csv"
    backup_masterdata_file_transformed = today + "_" + "VENDOR_ADDRESS_MASTERDATA_TRANSFORMED.csv"

    drop_masterdata_full_path = os.path.join(drop_masterdata_path,drop_masterdata_file)
    current_masterdata_full_path = os.path.join(current_masterdata_path,current_masterdata_file)
    backup_masterdata_original_full_path = os.path.join(backup_mastardata_path,backup_masterdata_file_original)
    backup_masterdata_transformd_full_path = os.path.join(backup_mastardata_path,backup_masterdata_file_transformed)

    # input("Enter something to proceed")
    ## While testing in local Use this
    ## Comment to be added in production
    # drop_masterdata_full_path = r"C:\Users\sahil.aggarwal\Desktop\Vendor Master Internal - Oct 12.xlsx"
    # current_masterdata_full_path = r"C:\Users\sahil.aggarwal\Desktop\VENDOR_ADDRESS_MASTERDATA_CURRENT.csv"
    # backup_masterdata_original_full_path = r"C:\Users\sahil.aggarwal\Desktop\test\VENDOR_ADDRESS_MASTERDATA_DROP.xlsx"
    # backup_masterdata_transformd_full_path  = r"C:\Users\sahil.aggarwal\Desktop\test\VENDOR_ADDRESS_MASTERDATA_trans.csv"
    print(drop_masterdata_full_path,current_masterdata_full_path,backup_masterdata_original_full_path,backup_masterdata_transformd_full_path)
    
    flag_file_name = check_file_presence(drop_masterdata_full_path, current_masterdata_full_path)
    if flag_file_name:
        DROP_VENDOR_MASTER = pd.read_excel(drop_masterdata_full_path)
        CURRENT_VENDOR_MASTER = pd.read_csv(current_masterdata_full_path)
        print("File Name is correct")
        list_needed = ['BP Name', 'BP Code', 'GST Registration Number', 'Wtax Code','State Code','Shipping From']
        list_got = list(DROP_VENDOR_MASTER.columns)
        flag_column_name = check_column_presence(list_needed, list_got)
        if flag_column_name == True:
            print("Matched columns")
            ## Comment to be removed in production
            ## Copying the current file to backup folder
            try:
                DROP_VENDOR_MASTER.to_excel(backup_masterdata_original_full_path, index = False)
            except:
                DROP_VENDOR_MASTER.to_csv(backup_masterdata_original_full_path, index = False)
            print("Original File copied to Backup folder.")
            # remove_file(drop_masterdata_full_path)
            ## Renaming the columns
            DROP_VENDOR_MASTER.rename(columns={'BP Code': 'VENDOR_ID'}, inplace=True)
            DROP_VENDOR_MASTER.rename(columns={'BP Name': 'VENDOR_NAME'}, inplace=True)
            DROP_VENDOR_MASTER.rename(columns={'GST Registration Number': 'VENDOR_GSTIN'}, inplace=True)

            DROP_VENDOR_MASTER, flag_validate = validate_masterdata(DROP_VENDOR_MASTER,CURRENT_VENDOR_MASTER,data_modified)
            if flag_validate:
                print("Proceeding to transform file")
                list_id_text = []
                # list_doc_text = []
                for row_ind, row in DROP_VENDOR_MASTER.iterrows():
                    # cnt+=1  
                    if str(row["VENDOR_GSTIN"]).lower() == "nan":
                        text = (str(row["VENDOR_NAME"])) + " "+ (str(row["VENDOR_NAME"])) +" "+ (str(row["VENDOR_NAME"])) +" "+ (str(row["VENDOR_NAME"]))
                    else:    
                        text = (str(row["VENDOR_GSTIN"]) + " "+str(row["VENDOR_NAME"])) + " "+ (str(row["VENDOR_GSTIN"]) + " "+str(row["VENDOR_NAME"])) +" "+ (str(row["VENDOR_GSTIN"]) + " "+str(row["VENDOR_NAME"])) +" "+ (str(row["VENDOR_GSTIN"]) + " "+str(row["VENDOR_NAME"]))
                    if len(text.split())<10:
                        if str(row["VENDOR_GSTIN"]).lower() != "nan":
                            text = text + " " + str(row["VENDOR_GSTIN"])
                        #print(text,len(text.split()))
                    if len(text.split())<10:
                        text = text + " " + str(row["VENDOR_NAME"])
                    #print(len(text.split()),text)
                    list_id_text.append(text)

                DROP_VENDOR_MASTER["IDENTIFIER_TEXT"] = list_id_text
                DROP_VENDOR_MASTER["DOCUMENT_TEXT"] = list_id_text
                DROP_VENDOR_MASTER["MATCH_SCORE"]= 0

                DROP_VENDOR_MASTER["CLIENT"] = "PIERIAN"

                print("Copying transformed file")
                ## Comment to be removed in production
                DROP_VENDOR_MASTER.to_csv(current_masterdata_full_path,index = False)
                DROP_VENDOR_MASTER.to_csv(backup_masterdata_transformd_full_path,index = False)
            else:
                print("Unsuccessful validations. No changes done")
        else:
            print("Columns name are not matching")
            send_email(content= "Dear Swaroop,\n\nColumns name are not matching in vendor masterdata for blinkit.\n\nBest regards,\nSahil")
            print("Mail sent")   
    else:
        print("File does not exists")
        send_email(content= "Dear Swaroop,\n\nVendor masterdata file does not exist in shared drive for blinkit.\n\nBest regards,\nSahil")
        print("Mail sent")       
    if log_file is not None:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_file.close() 
