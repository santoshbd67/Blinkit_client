import traceback
import requests
import json
import preProcUtilities as putil
import TAPPconfig as cfg
from polling import TimeoutException, poll
from  logging_module import get_logger
logger = get_logger()

def send_request(auth_token, document_id, sub_id):
    try:
        url = cfg.getExtractionGetAPI()
        enc_token = putil.encryptMessage(json.dumps({
            "auth_token": auth_token,
            "documentId": document_id,
            "sub_id": sub_id
        }))
        enc_token = enc_token.decode("utf-8")
        message = json.dumps({"message": enc_token})
        headers = {"Content-Type": "application/json"}

        response = requests.post(url=url, headers=headers, data=message)
        if response.status_code != 200:
            return None
        
        resp_obj = response.json()
        resp_obj = putil.decryptMessage(resp_obj["message"])
        resp_obj = json.loads(resp_obj)
        
        if resp_obj["status_code"] != 200:
            return None
        
        return resp_obj
    except Exception as e:
        logger.error(f"Error occurred in send_request: {str(e)}")
        return None

def check_response(response):
    if response is None:
        return True

    ext_status = response["status"]
    ext_status_code = response.get("status_code")
    error_message = response.get("error_message")
    logger.debug(f"Response status: {ext_status}")
    logger.debug(f"Response status code: {ext_status_code}")
    logger.debug(f"Error message: {error_message}")
    if (ext_status == "Processing" or ext_status == "Submitted"):
        return False
    else:
        return True

@putil.timing
def poll_status(auth_token, delta, document_id, sub_id):
    try:
        polling_result = poll(
            lambda: send_request(auth_token, document_id, sub_id),
            check_success=check_response,
            step=5,
            timeout=delta
        )
        return polling_result
    except TimeoutException as te:
        # traceback.print_exc()
        logger.error("Polling TimeOutException", exc_info=True)
        return None
