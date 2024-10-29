import pandas as pd
import pytz
from datetime import datetime, timedelta
from pymongo import MongoClient 
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders



def setup_logger():
    # Create a logger
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    
    # Create a file handler
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    log_file = f"logs/log_{current_time}.txt"
    file_handler = logging.FileHandler(log_file)
    
    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(file_handler)
    
    return logger

def send_email(sender_email, sender_password, recipient_email_list, subject, body, csv_filename):
    # Create a multipart message
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = ", ".join(recipient_email_list)
        message["Subject"] = subject

        # Add body to email
        message.attach(MIMEText(body, "plain"))

        # Open the CSV file in binary mode
        with open(csv_filename, "rb") as attachment:
            # Add file as application/octet-stream
            # Email client can usually download this automatically as attachment
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())

        # Encode file in ASCII characters to send by email    
        encoders.encode_base64(part)

        # Add header as key/value pair to attachment part
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {csv_filename}",
        )

        # Add attachment to message and convert message to string
        message.attach(part)
        text = message.as_string()
        
        # Log in to SMTP server and send email
        with smtplib.SMTP("smtp.office365.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email_list, text)
   
        
def get_ui_data(collection):
    # Get the current date and time in UTC
    current_date_utc = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=pytz.UTC)

    # Calculate the 1st day of the current month in UTC
    start_of_month_utc = current_date_utc.replace(day=1)

    # Calculate yesterday's date in UTC
    yesterday_utc = current_date_utc - timedelta(days=1)

    # Calculate end of yesterday
    end_of_yesterday_utc = datetime(yesterday_utc.year, yesterday_utc.month, yesterday_utc.day, 23, 59, 59, 999999, tzinfo=pytz.UTC)

    # Convert date range to UNIX timestamps (in milliseconds)
    start_timestamp_utc = int(start_of_month_utc.timestamp() * 1000)
    end_timestamp_utc = int(end_of_yesterday_utc.timestamp() * 1000)

    # print("Start of month (UTC) timestamp:", start_timestamp_utc)
    # print("Yesterday (UTC) timestamp:", end_timestamp_utc)


    # Define the query
    query = {
        "status": {"$nin": ["NEW", "EXTRACTION_INPROGRESS", "FAILED"]},
        "submittedOn": {"$gte": start_timestamp_utc, "$lte": end_timestamp_utc}
    }

    # Projection: specify only the fields you want to retrieve
    projection = {"submittedOn": 1, "documentId": 1, "status": 1, "_id": 0}

    # Execute the query with projection
    results= collection.find(query, projection)

    # Initialize an empty list to store documents
    documents = []

    # Iterate over the cursor to retrieve documents
    for doc in results:
        documents.append(doc)

    # Convert list of documents to DataFrame
    df = pd.DataFrame(documents)

    return df, current_date_utc, start_of_month_utc, end_of_yesterday_utc


# Connection-

conn_string = "mongodb://blinkitmouser:Blinkitmo102@10.0.0.46:27017/?directConnection=true&appName=mongosh+1.10.2"
client = MongoClient(conn_string) 

# Database connection-
db = client['blinkit_prod_db']

# UI Data table-
collection = db['document_metadata']

ui_data, current_date, start_date, end_date = get_ui_data(collection)

# Connect to SQL database
engine = create_engine("postgresql://blinkit_sql@blinkittappsql:9+px|,rJ&Jt&FZJ@blinkittappsql.postgres.database.azure.com:5432/blinkit-tapp")

# Query to fetch the data from Consumption DB
query = '''
SELECT
    master.sub_id,
    master.allotted,
    master.end_date,
    master.type,
    auth.call_id AS docs_requested,
    auth.authenticated_time AT TIME ZONE 'UTC' AS authenticated_time_utc,
    MAX(calls.pages_requested) AS pages_requested
FROM
    subscription_master master
LEFT JOIN
    subscription_auth_tokens auth USING (sub_id)
LEFT JOIN
    subscription_call_records calls USING (auth_token)
WHERE
    calls.status = 1
    AND auth.authenticated_time >= date_trunc('month', CURRENT_DATE AT TIME ZONE 'UTC')
    AND auth.authenticated_time <= (CURRENT_DATE AT TIME ZONE 'UTC' - INTERVAL '1 second')
GROUP BY
    master.sub_id,
    master.allotted,
    master.end_date,
    master.type,
    auth.call_id,
    authenticated_time_utc;

'''

# Setup Logger
logger = setup_logger()

cons_db_data = pd.read_sql_query(query, engine)
logger.info(f'"MongoDB data:" {ui_data.shape}, "Consumption DB data:" {cons_db_data.shape} for date range: {start_date} to {end_date}')

# The documentIds present in mongodb data but not in Consumption DB
In_Mongodb_Not_In_Cons = list(set(ui_data['documentId'])-set(cons_db_data['docs_requested']))
logger.info(f"{len(In_Mongodb_Not_In_Cons)}, The extracted documents which are present in Mongodb data but not in consumption DB for date range: {start_date} to {end_date}")

# The documentIds present in Consumption DB but not in mongodb data
In_Cons_Not_In_Mongodb = list(set(cons_db_data['docs_requested'])-set(ui_data['documentId']))
logger.info(f"{len(In_Cons_Not_In_Mongodb)}, The extracted documents which are present in consumption DB but not in Mongodb data for date range: {start_date} to {end_date}")

print(In_Mongodb_Not_In_Cons)
print(In_Cons_Not_In_Mongodb)

# Create a Session class to interact with the database
Session = sessionmaker(bind=engine)

# Define your queries using SQLAlchemy's text construct
cp_stage_is_present_or_not_query = text("""
    SELECT COUNT(*) AS stage_count
    FROM call_log
    WHERE auth_token = :auth_token
    AND stage = 'client_processing'
""")

latest_auth_token_query = text("""
    SELECT auth_token, authenticated_time
    FROM subscription_auth_tokens 
    WHERE call_id = :call_id 
    ORDER BY authenticated_time DESC 
    LIMIT 1
""")

sum_of_status_query = text("""
    SELECT SUM(a.status) s 
    FROM (
        SELECT MAX(status) status, stage 
        FROM call_log 
        WHERE auth_token = :auth_token
        AND stage IN ('submit', 'input_preprocess', 'extraction') 
        GROUP BY stage
    ) a
""")

cp_status_zero_query = text("""
    SELECT * 
    FROM call_log 
    WHERE auth_token = :auth_token 
    AND status = 0
    AND stage = 'client_processing'
""")

upd_query = text("UPDATE call_log SET status = 1 WHERE stage='client_processing' and auth_token = :auth_token")

get_previous_time_query = text("""
    SELECT "time" 
    FROM call_log
    WHERE auth_token = :auth_token
    AND stage = 'extraction'
    AND status = 1
    ORDER BY "time" DESC
    LIMIT 1
""")

insert_query = text("""
    INSERT INTO call_log (auth_token, stage, "time", status, is_start, is_end)
    VALUES (:auth_token, 'client_processing', :time, 1, 0, 1)
""")

# Create a session
session = Session()

updated_docs = []
inserted_stage_docs = []

try:
    if In_Mongodb_Not_In_Cons:
        for doc_id in In_Mongodb_Not_In_Cons:
            auth_token = session.execute(latest_auth_token_query, {"call_id": doc_id}).fetchone()[0]
            status = session.execute(sum_of_status_query, {"auth_token": auth_token}).fetchone()[0]
            logger.info(f"Sum of status' of first three stages: {status} for auth_token: {auth_token}")
            
            if status is not None and status >= 3:
                cp_present = session.execute(cp_stage_is_present_or_not_query, {"auth_token": auth_token}).fetchone()[0]
                if cp_present:
                    cp_status = session.execute(cp_status_zero_query, {"auth_token": auth_token}).fetchone()
                    if cp_status and cp_status[0] == 0:
                        session.execute(upd_query, {"auth_token": auth_token})
                        session.commit()
                        logger.info(f"Update query executed successfully! for document_id: {doc_id}")
                        updated_docs.append({"doc_id": doc_id, "auth_token": auth_token})
                    else:
                        updated_docs.append(None)
                        inserted_stage_docs.append(None)
                        logger.info(f"Client Processing stage is not zero for auth_token: {auth_token}\n\n Might be the timing mismatch!")
                else:
                    previous_stage_time = session.execute(get_previous_time_query, {"auth_token": auth_token}).fetchone()[0]
                    if previous_stage_time:
                        new_time = previous_stage_time + timedelta(seconds=3)
                        session.execute(insert_query, {"auth_token": auth_token, "time": new_time})
                        session.commit()
                        logger.info(f"Insert query executed successfully! for document_id: {doc_id}") 
                        inserted_stage_docs.append({"doc_id": doc_id, "auth_token": auth_token})     
                    else:
                        updated_docs.append(None)
                        inserted_stage_docs.append(None)
                        logger.error(f"ERROR: No previous stage time found for auth_token while running insert query: {auth_token}")                    
                   
            else:
                logger.info(f"Sum of the status of first three stages is None!")  
                     
except Exception as e:
    logger.error(f"ERROR: {str(e)}")
finally:
    session.close()


# Create dataframe
data = {'Updated_Documents': updated_docs, 'Stage_Inserted_Documents': inserted_stage_docs}

# the CSV file path
csv_file_path = f"updated_data_for_{current_date.strftime('%Y-%m-%d')}.csv"
df = pd.DataFrame(data)

# Write DataFrame to CSV file (even if it's empty)
df.to_csv(csv_file_path, index=False)
logger.info(f"CSV file created at {csv_file_path}")

# Send csv file in the email-
sender_email = 'paiges.admin@taoautomation.com'
sender_password = 'XdFgHjUv_09'
sending_list = ['hariharamoorthy.theriappan@taoautomation.com', 'rupesh.alluri@taoautomation.com', 'divya.maggu@taoautomation.com', 'sahil.aggarwal@taoautomation.com']
subject = 'Daily Metrics Report for BLINKIT Procduction'
body = f"Hello,\n\nI hope this email finds you well. Please find the attached file for updated documents. \nDate: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}  \n\nThanks"

send_email(sender_email, sender_password, sending_list, subject, body, csv_file_path)
print("Email sent successfully!")

engine.dispose()



