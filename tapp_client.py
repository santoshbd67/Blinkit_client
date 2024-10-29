# -*- coding: utf-8 -*-
# from socket import timeout
import traceback
import os
import time
import pandas as pd
import math
import preProcUtilities as putil
import TAPPconfig as cfg
import uuid
import json
import post_processor as pp
import numpy as np
import dateparser
import datetime
import sys
from celery import Celery
from poller import poll_status
from client_rules import is_equal_subTotal_TotalAmount,check_field_stp_score,check_multi_invoices_stp
import requests
import sys
from datetime import datetime as dt
from  logging_module import get_logger, setup_logging
import warnings
import copy
warnings.filterwarnings("ignore")

broker = cfg.getTaskBroker()
task_name = cfg.getTaskName()
celeryapp = Celery(task_name,broker=broker)
vendorMasterDataPath = cfg.getVendorMasterData()
masterFilePath = os.path.join(vendorMasterDataPath)
VENDOR_MASTERDATA = pd.read_csv(masterFilePath, encoding='unicode_escape')
app_name = cfg.getAppName()

#Get values for request/response
docType = cfg.getDocumentType()
sysUser = cfg.getSystemUser()
tappVer = cfg.getTappVersion()
rootFolderPath = cfg.getRootFolderPath()

docUpdApi = cfg.getDocUpdApi()
docResAddApi = cfg.getDocResAddApi()

paramStatusSuccess = cfg.getParamStatusSuccess()
paramStatusFailed = cfg.getParamStatusFailed()

statusRevComp = cfg.getStatusReviewCompleted()
statusProcessed = cfg.getStatusProcessed()
statusReview = cfg.getStatusReview()
statusFailed = cfg.getStatusFailed()

errmsgExtractionUpdateFail = cfg.getErrmsgExtractResNotUpd()
errCode = cfg.getErrcodeError()

stgExtract = cfg.getStageExtract()

statusmsgExtractSuccess = cfg.getStatusmsgExtractSuccess()

def timing(f):
    """
    Function decorator to get execution time of a method
    :param f:
    :return:
    """
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('Time taken to execute {:s} is: {:.3f} sec'.format(f.__name__, (time2 - time1)))
        return ret

    return wrap

def apiInit(documentId):
    docApiInfo = {}
    docApiInfo["id"] = docUpdApi
    docApiInfo["ver"] = tappVer
    docApiInfo["ts"] = math.trunc(time.time())
    docParams = {}
    docRequest = {}
    docApiInfo["params"] = docParams
    docApiInfo["request"] = docRequest
    docRequest["documentId"] = documentId
    docParams["msgid"] = str(uuid.uuid1())
    docRequest["documentType"] = docType
    docRequest["lastUpdatedBy"] = sysUser

    docApiInfo = updDocInfo(None,
                            None,
                            None,
                            None,
                            None,
                            paramStatusFailed,
                            docRequest,
                            docParams,
                            docApiInfo)

    return docApiInfo

def updDocInfo(reqStatus,
               stage,
               statusMsg,
               err,
               errMsg,
               prmStatus,
               docRequest,
               docParams,
               docApiInfo):
    docRequest["status"] = reqStatus
    docRequest["stage"] = stage
    docRequest["statusMsg"] = statusMsg
    docRequest["lastProcessedOn"] = math.trunc(time.time())

    docParams["err"] = err
    docParams["errmsg"] = errMsg
    docParams["status"] = prmStatus

    docApiInfo["request"] = docRequest
    docApiInfo["params"] = docParams
    return docApiInfo
@putil.timing
def prepare_request_ML(docRequest,
                       docInfo):
    #Form the TAPP API request object for document result update
    resultApiInfo = {}
    resultApiInfo["id"] = docResAddApi
    resultApiInfo["ver"] = tappVer
    resultApiInfo["ts"] = math.trunc(time.time())

    resRequest = {}
    resultApiInfo["request"] = resRequest
    resRequest["documentId"] = docRequest["documentId"]
    resRequest["processingEngine"] = "ML"
    resRequest["documentInfo"] = docInfo["result"]["documentInfo"]
    resRequest["documentLineItems"] = docInfo["result"]["documentLineItems"]

    for key in docInfo.keys():
        if key != "result":
            docRequest[key] = docInfo[key]

    return docRequest,resultApiInfo

@putil.timing
def updateStatusToCloud(auth_token,
                        documentId,
                        success):
    uploaded = False
    try:
        localPaths = []
        #update Cloud with status against Auth-token
        cloud_dict = {}
        cloud_dict["auth_token"] = auth_token
        cloud_dict["sub_id"] = cfg.getSubscriberId()
        cloud_dict["document_id"] = documentId
        cloud_dict["body"] = {}
        cloud_dict["stage"] = "client_processing"
        cloud_dict["create_time"] = str(datetime.datetime.now())
        cloud_dict["success"] = success
        cloud_dict["is_start"] = 0
        cloud_dict["is_end"] = 1
        cloud_json = json.dumps(cloud_dict)

        localPath = os.path.join(rootFolderPath,
                                 documentId + "_auth:" + auth_token + "__islaststage.json")
        localPaths.append(localPath)

        with open(localPath,"w") as f:
            f.write(cloud_json)

        uploaded = putil.uploadFilesToBlobStore(localPaths)
        if uploaded:
            for localPath in localPaths:
                try:
                    os.remove(localPath)
                except:
                    pass
        return uploaded
    except:
        print("updateStatusToCloud",
              traceback.print_exc())
        return False
@putil.timing
def updateSuccess(status,
                  stage,
                  docResult,
                  docApiInfoInp,
                  documentId,
                  auth_token,
                  callbackUrl):

    #Init API Jsons
    # docRequest = docApiInfo["request"]
    # docParams = docApiInfoInp["params"]
    # print(docParams)
    docApiInfo = apiInit(documentId)
    docApiInfo["request"]["status"] = status
    docApiInfo["request"]["stage"] = stage
    docApiInfo["request"]["statusMsg"] = statusmsgExtractSuccess
    docApiInfo["request"]["lastProcessedOn"] = math.trunc(time.time())
    docApiInfo["request"]["extractionCompletedOn"] = math.trunc(time.time())
    #Not storing invoice data in document metadata due to security concerns from Swiggy
    # docApiInfo["request"]["invoiceNumber"] = docApiInfoInp["request"].get("invoiceNumber","")
    # docApiInfo["request"]["invoiceDate"] = docApiInfoInp["request"].get("invoiceDate","")
    # docApiInfo["request"]["totalAmount"] = docApiInfoInp["request"].get("totalAmount","")
    #Not storing invoice data in document metadata due to security concerns from Swiggy
    docApiInfo["request"]["currency"] = docApiInfoInp["request"].get("currency","")
    docApiInfo["request"]["overall_score"] = docApiInfoInp["request"]["overall_score"]
    docApiInfo["request"]["stp"] = docApiInfoInp["request"]["stp"]
    docApiInfo["request"]["vendorId"] = docApiInfoInp["request"].get("vendorId","")
    docApiInfo["request"]["qualityScore"] = docApiInfoInp.get("qualityScore",0.0)
    docApiInfo["request"]["pages"] = docApiInfoInp["request"]["pages"]
    #Aug 05 2022 - Add number of pages OCRed
    docApiInfo["request"]["pages_ocred"] = docApiInfoInp.get("pages_ocred",0)
    #Aug 05 2022 - Add number of pages OCRed
    downloadUrl = cfg.getPreprocServer() + "/" + cfg.getDownloadResultsApi() + "/" + documentId
    docApiInfo["request"]["resultDownloadLink"] = downloadUrl
    #Mark that extraction is completed
    docApiInfo["request"]["extraction_completed"] = 1
    #download png file to local folder and update path
    # docApiInfo["request"]["rawPrediction"] = docApiInfoInp['result'].get("rawPrediction","")

    docApiInfo["params"]["err"] = None
    docApiInfo["params"]["errmsg"] = None
    docApiInfo["params"]["status"] = paramStatusSuccess
    # docResult["params"] = {}
    # docResult["params"]["msgid"] = docApiInfo["params"]["msgid"]
    
    #Jul 28 2022 - When we retry a failed document, check if results were already created.
    #It happens sometimes that only status update has failed, so it is better to update only the status
    docInfo = putil.getDocumentApi(documentId,
                                       callbackUrl)
    docResultPresent = putil.getDocumentResultApi(documentId,
                                                  callbackUrl)
    sys.stdout.flush()
    """
        
    #Get document metadata to check the oringinal invoice's document ID and the STP status
    ## document/get
    docInfo_Org = putil.getDocumentApi(orgDocumentId,
                                       callbackUrl)
    ## document/result/get
    docInfoRes_Org = putil.getDocumentResultApi(orgDocumentId,
                                                callbackUrl)
    #This status must be "Review". 
    org_status = docInfo_Org["status"]
    #Check if the original invoice is already corrected
    org_corrected = False
    #Parse the results to check if any correction is done. If yes, we cannot update discrepancy note and the original invoice to review completed
    org_stp = #This you will get it from discrepancy note's document metadata. you can do a getDocumentApi of documentId.
    #This will give you the original invoice's STP status, which you passed it to the discrepancy note while extracing original invoice
    """
    docResultUpdated = True
    if not docResultPresent:
        docResultUpdated = putil.createDocumentResultsApi(documentId,
                                                          docResult,
                                                          callbackUrl)
    print("Document Update Request for successful transaction:: ",docResultUpdated, docApiInfo)
    
    if docResultUpdated:
        uploaded = updateStatusToCloud(auth_token,
                                       documentId,1)
        if not uploaded:
            docApiInfo["request"]["pp_cloud_update"] = 0
        else:
            docApiInfo["request"]["pp_cloud_update"] = 1
    
        # docApiInfo["request"] = docRequest
        # print("sahil testing2", type(docApiInfo), docApiInfo)
        print("Update Document Api", docApiInfo)
        if "result" in docApiInfo.keys():
            del docApiInfo["result"]

        #Jul 05 2022 - set stp true only if all business rules satisfy
        from client_rules import bizRuleValidateForUi as biz_rl
        from client_rules import isMasterDataPresent
        from client_rules import isInvoiceNumberAnAmount
        from client_rules import isTotOrSubCalc
        from client_rules import getBuyerStateCode as getBSC
        callbackUrl = cfg.getUIServer()
        if not documentId.endswith("_DISCR"):
            result, document_result_updated,_  = biz_rl(documentId, callbackUrl, UI_validation= False, VENDOR_ADDRESS_MASTERDATA = VENDOR_MASTERDATA)
            print("Get Buyer State Code", documentId)
            buyerStateCode = getBSC(documentId,
                                    callbackUrl)
            if buyerStateCode:
                docApiInfo["request"]["buyer_state_code"] = buyerStateCode
                print("Buyer State Code",
                    documentId,
                    buyerStateCode)
            else:
                docApiInfo["request"]["buyer_state_code"] = ''
            print("Business Rules Result",result)
            if result is not None:
                if len(result) > 0:
                    stp = False
                else:
                    stp = True
                    #After biz rule validation passed through,
                    #please also check if invoiceNumber is not an Amount field
                    
                    #After biz rule validation passed through,
                    #please also check if invoiceNumber is not an Amount field

                    flag_isMasterDataPresent = isMasterDataPresent(documentId,
                                                            cfg.getUIServer())
                    stp = flag_isMasterDataPresent
                    print("isMasterDataPresent stp flag :",flag_isMasterDataPresent)
                    # if stp:
                    #     stp = not isInvoiceNumberAnAmount(documentId,
                    #                                       cfg.getUIServer())
                    #     print("stp flag  After isInvoiceNumberAnAmount :",stp)
                    # if stp:
                    #     # "totalAmount" and "subtotal calculate field value check"
                    #     stp = not isTotOrSubCalc(documentId,
                    #                              cfg.getUIServer())
                    #     print("stp flag  After isTotOrSubCalc :",stp)
                    #     #Check if total or subtotal is calculated and do not allow for stp
                    # Checking total == subtotal if true making stp false
                    # sub_tot_match = is_equal_subTotal_TotalAmount(documentId,
                    #                                               cfg.getUIServer())
                    # print("sub_tot_match :",sub_tot_match)
                    # if sub_tot_match is not None:
                    #     stp = sub_tot_match
                
                    if stp:
                    # "invoiceDate","invoiceNumber","totalAmount" confidence check
                        invdt_stp = check_field_stp_score(documentId,
                                                            cfg.getUIServer())
                        print("date field stp Check :",invdt_stp)
                        stp = invdt_stp
                    if stp:
                        multi_invoices = check_multi_invoices_stp(documentId,
                                                            cfg.getUIServer())
                        print("Mult invoice stp check :",multi_invoices)
                        stp = multi_invoices
                    # 23 May 2023 Commented document quality score for blinkit
                    # if stp:
                    #     docQualityCutOff = cfg.GET_DOCUMENT_QUALITY_CUT_OFF_SCORE()
                    #     docQualityScore = docApiInfoInp.get("qualityScore",0.0)
                    #     print("docQualityCutOff score :",docQualityCutOff)
                    #     print("docQualityScore :",docQualityScore)
                    #     if (docQualityCutOff > 0) and (docQualityScore < docQualityCutOff):
                    #         stp = False
                    if stp:
                        if docApiInfo.get("request").get("stp") == "False" or docApiInfo.get("request").get("stp") == False:
                            print("Invoice not set STP from Post-Processor")
                            stp = False
                    
            else:
                print("Business Rules Result is empty",False)
                stp = False
            if stp:
                if docApiInfo.get("request").get("stp") == "False" or docApiInfo.get("request").get("stp") == False:
                    print("Invoice not set STP from Post-Processor")
                    stp = False
            print("STP after validating rules : ",stp)
            docApiInfo["request"]["stp"] = stp
        if documentId.endswith("_DISCR"):
            print("checking stp for invoice and discr note")
            # documentIdInvoice = str(documentId)[:-6]
            # docInfoInvoice = putil.getDocumentApi(documentIdInvoice,
            #                             callbackUrl)
            # docInfoResInvoice = putil.getDocumentResultApi(documentIdInvoice,
            #                                         callbackUrl)
            # invoice_corrected = False
            # for item in docInfoResInvoice["result"]["document"]["documentInfo"]:
            #     if item.get("correctedValue")!= None:
            #         invoice_corrected = True
            #         break
            # original_invoice_stp = docInfo["result"]["document"]["linked_document"]["original_invoice_stp"]
            stp_discr = docApiInfo.get("request").get("stp")
            # print("Original Invoice stp:",original_invoice_stp)
            # print("Status of Invoice:",docInfoInvoice["result"]["document"]["status"])
            # print("Is Invoice Corrected", invoice_corrected)
            print("Discr note stp status",stp_discr)
            # result = biz_rl(documentIdInvoice, callbackUrl)
            # print("Business Rules Result",result)
            # if result is not None:
            #     if len(result) > 0:
            #         stp_invoice = False
            #     else:                    
            #         flag_isMasterDataPresent = isMasterDataPresent(documentIdInvoice,
            #                                                 cfg.getUIServer())
            #         stp_invoice = flag_isMasterDataPresent
            #         print("isMasterDataPresent stp flag :",flag_isMasterDataPresent)
            #     if stp_invoice:
            #         invdt_stp = check_field_stp_score(documentIdInvoice,
            #                                             cfg.getUIServer())
            #         print("date field stp Check :",invdt_stp)
            #         stp_invoice = invdt_stp
            #     if stp_invoice:
            #         multi_invoices = check_multi_invoices_stp(documentIdInvoice,
            #                                             cfg.getUIServer())
            #         print("Mult invoice stp check :",multi_invoices)
            #         stp_invoice = multi_invoices
            # else:
            #     stp_invoice = False

            ## 7 June 2023 Added STP as false bcz of FP
            
            #if docInfoInvoice["result"]["document"]["status"] == "REVIEW" and invoice_corrected == False and original_invoice_stp == True and stp_discr == True and stp_invoice == True:
            if stp_discr==True:
                print("All conditions satisfied, changing staus to review completed")
                # reqdata = {}
                # reqdata["documentId"] = documentIdInvoice
                # reqdata["status"] = statusRevComp
                # reqdata["stp"] = True
                # reqdata["lastUpdatedOn"] = math.trunc(time.time())
                # data = {
                #     "ver":"1.0",
                #     "request":reqdata
                # }
                # updated_invoice,resp_code = putil.updateDocumentApi_New(documentIdInvoice,
                #                                         data,
                #                                         callbackUrl)
                # print("Updated Invoice status",updated_invoice)
                docApiInfo["request"]["status"] = statusRevComp
                docApiInfo["request"]["stp"] = True
            else:
                docApiInfo["request"]["status"] = statusReview
                docApiInfo["request"]["stp"] = False
        else:
            ## 7 June 2023 Added STP as false bcz of FP
            print("checking not _DISCR ",docApiInfo)
            if stp:
                docApiInfo["request"]["status"] = statusRevComp
            else:
                docApiInfo["request"]["status"] = statusReview
        

        #Jul 05 2022 - set stp true only if all business rules satisfy
        print("Doc Status before update: ",
              docApiInfo.get("request").get("status"))
        
        #write code for quality score here
        # docApiInfo["request"]["status"] = statusRevComp
        #Jul 18 2022 Call new function for updateDocument that returns response code as well
        # updated = putil.updateDocumentApi(documentId,
        #                                   docApiInfo,
        #                                   callbackUrl)
        # print("Is RawPrediction there:")
        # if "rawPrediction" in docApiInfo.get("request"):
        #     print("YES")
        # else:
        #     print("NO")
        # if docApiInfo.get("request").get("rawPrediction"):
        #     del docApiInfo["request"]["rawPrediction"]
        # print("Is RawPrediction there after deletion:")
        # if "rawPrediction" in docApiInfo.get("request"):
        #     print("YES")
        # else:
        #     print("NO")

        """
        #Update stp for both disr note and original invoice
        if stp == True and org_stp == True:
            if org_status == "REVIEW" and org_corrected == False:
                updated,resp_code = putil.updateDocumentApi_New(documentId,
                                                                docApiInfo,
                                                                callbackUrl)
                #Create a document update json for original invoice with only orgdocumentId, stp = True, status = Review_completed
                org_update,org_resp_code = putil.updateDocumentApi_New(orgDocumentId,
                                                                       orgDocApiInfo,
                                                                       callbackUrl)
            else:
                stp = False
        else:
            stp = False
        """
        updated,resp_code = putil.updateDocumentApi_New(documentId,
                                                        docApiInfo,
                                                        callbackUrl)
        sys.stdout.flush()
        # if not updated and (resp_code == 413):
        #     print("Inside resp_code 413, deleting rawPrediction")
        #     if docApiInfo.get("request").get("rawPrediction"):
        #         del docApiInfo["request"]["rawPrediction"]
        #         updated,resp_code = putil.updateDocumentApi_New(documentId,
        #                                                         docApiInfo,
        #                                                         callbackUrl)
        if not updated:
            updated = updateFailure(stgExtract,
                                    statusFailed,
                                    errCode,
                                    errmsgExtractionUpdateFail,
                                    documentId,
                                    callbackUrl,
                                    auth_token)

        #Jul 18 2022 Call new function for updateDocument that returns response code as well

    else:
        updated = updateFailure(stgExtract,
                                statusFailed,
                                errCode,
                                errmsgExtractionUpdateFail,
                                documentId,
                                callbackUrl,
                                auth_token)
    sys.stdout.flush()
    return updated

def updateFailure(stage,
                  statusMsg,
                  error,
                  errorMsg,
                  documentId,
                  callbackUrl,
                  auth_token):

    #Check doc status
    status = putil.getDocumentStatus(documentId,callbackUrl)
    if not((status == statusReview) or (status == statusRevComp)) and (status == "NEW" or status == "EXTRACTION_INPROGRESS"):
        #Init API Jsons
        docApiInfo = apiInit(documentId)
        docRequest = docApiInfo["request"]
        docParams = docApiInfo["params"]
        docApiInfo = updDocInfo(statusFailed, #reqStatus
                                stage, #stage
                                statusMsg, #statusMsg
                                error, #err
                                errorMsg, #errMsg
                                paramStatusFailed, #paramStatus
                                docRequest,
                                docParams,
                                docApiInfo)
        print("Document Update Request for failed transaction: ",docApiInfo)
        #print("Document Update Request for failed transaction: ")
        uploaded = updateStatusToCloud(auth_token,
                                       documentId,0)
        docRequest = docApiInfo["request"]
        if not uploaded:
            docRequest["pp_cloud_update"] = 0
        else:
            docRequest["pp_cloud_update"] = 1
        
        #Mark that extraction is completed
        docRequest["extraction_completed"] = 1
    
        docApiInfo["request"] = docRequest
    
        updated = putil.updateDocumentApi(documentId,
                                          docApiInfo,
                                          callbackUrl)
    
        return updated
    return False

@putil.timing
def updateDeleted(documentId,
                  deleteReason,
                  callbackUrl,
                  auth_token):

    
        #Init API Jsons
        docApiInfo = apiInit(documentId)
        docRequest = docApiInfo["request"]
        docParams = docApiInfo["params"]
        
        docRequest["status"] = "DELETED"
        docRequest["stage"] = "DELETED"
        docRequest["statusMsg"] = deleteReason
        docRequest["deleteReason"] = deleteReason
        docRequest["lastProcessedOn"] = math.trunc(time.time())
        docRequest["deleteTime"] = math.trunc(time.time())
        docRequest["deletedBy"] = "AGqCDJY5rbTVsaMxKIVABg==" ## Encrypted balue for SYSTEM
        docRequest["totalReviewedTime"] = 1
        docParams["status"] = "DELETED"

        docApiInfo["request"] = docRequest
        docApiInfo["params"] = docParams
        
        print("Document Update Request for deleted transaction: ",docApiInfo)
        #print("Document Update Request for failed transaction: ")
        uploaded = updateStatusToCloud(auth_token,
                                       documentId,0)
        docRequest = docApiInfo["request"]
        if not uploaded:
            docRequest["pp_cloud_update"] = 0
        else:
            docRequest["pp_cloud_update"] = 1
        
        #Mark that extraction is completed
        docRequest["extraction_completed"] = 1
    
        docApiInfo["request"] = docRequest
    
        updated = putil.updateDocumentApi(documentId,
                                          docApiInfo,
                                          callbackUrl)
    
        return updated


@putil.timing
def processExtRes(output,docInfo):
    try:
        inpFlds = ['index', 'token_id', 'page_num', 'line_num', 'line_text', 'line_left',
        'line_top', 'line_height', 'line_width', 'line_right', 'line_down', 'word_num',
        'text', 'conf', 'left', 'top', 'height', 'width', 'right', 'bottom', 'image_height', 
        'image_widht', 'wordshape', 'W1Ab', 'W2Ab', 'W3Ab', 'W4Ab', 'W5Ab', 'd1Ab', 'd2Ab', 'd3Ab', 'd4Ab', 
        'd5Ab', 'W1Lf', 'W2Lf', 'W3Lf', 'W4Lf', 'W5Lf', 'd1Lf', 'd2Lf', 'd3Lf', 'd4Lf', 'd5Lf',
        'predict_label', 'prediction_probability']
        if output is not None:
            docObj = output[0]
            rawPrediction = output[1][inpFlds]
            overall_score = output[2]
            stp = output[3]
            vendor_id = output[4]
            docInfo["result"] = docObj

            #update docInfo with invoice number, invoice date, currency and total Amount
            documentInfo = docObj["documentInfo"]
            reqFlds = ["invoiceNumber","invoiceDate","currency","totalAmount"]
            for fld in documentInfo:
                if fld["fieldId"] in reqFlds:
                    try:
                        if fld["fieldId"] == "invoiceDate":
                            dtparsed = dateparser.parse(fld["fieldValue"])
                            if dtparsed is not None:
                                timeval = dtparsed.hour + dtparsed.minute + dtparsed.second
                                if timeval > 0:
                                    docInfo[fld["fieldId"]] = str(datetime.datetime.timestamp(dtparsed))
                                else:
                                    docInfo[fld["fieldId"]] = str(datetime.datetime.timestamp(dtparsed) * 1000)
                            else:
                                docInfo[fld["fieldId"]] = fld["fieldValue"]
                    except Exception as e:
                        print("Exception occured in processExtRes while processing invoiceDate",e)
                        docInfo[fld["fieldId"]] = fld["fieldValue"]
                    else:
                        docInfo[fld["fieldId"]] = fld["fieldValue"]

            docInfo["overall_score"] = overall_score
            docInfo["stp"] = stp
            docInfo["vendorId"] = vendor_id
            #Jul 20 2022 - stop updating rawPrediction
            docInfo['result']["rawPrediction"] = rawPrediction.to_json(orient = "records")
            # docInfo['result']["rawPrediction"] = json.dumps({})
            #Jul 20 2022 - stop updating rawPrediction
            #Aug 05 2022 - find number of pages OCRed
            no_pages_ocred = 0
            try:
                no_pages_ocred = len(list(rawPrediction["page_num"].unique()))
            except:
                pass
            docInfo['pages_ocred'] = no_pages_ocred
            #Aug 05 2022 - find number of pages OCRed
            return docInfo
        else:
            return None
    except:
        print("Error in extraction or in processing post processor result",
              traceback.print_exc())
        return None
def transform_page_num(df):
    """
    Transforms the page_num column to have consecutive integers starting from 0
    based on the unique values in the column. Returns a copy of the original dataframe
    in case of an exception.
    This is done specifically for blinkit discrepancy note when avoiding extraction code.
    for other cases it cause no impact
    """
    df_copy = copy.deepcopy(df)
    try:  
        df_copy['page_num'] = pd.factorize(df_copy['page_num'])[0]
        return df_copy
    except Exception as e:
        print(f"An error occurred: {e}")
        return df


@timing
def processExtraction(pred_file_path,
                      documentId,
                      callbackUrl):
    #1. It calls the image preprocess method for image enhancement and extract invoice number,
    #Date, Currency and Amount fields from Invoice
    #2. It updates the status of a preprocessed image to "Ready for Extraction", if the
    # processes are successfully completed. In case of failure, it is updated as "Failed"
    callbackUrl = cfg.getUIServer()
    def convert(o):
        if isinstance(o, np.int64): return int(o)

    try:

        DF_PRED = pd.read_csv(pred_file_path,
                              index_col = 0)
        print(DF_PRED['page_num'].drop_duplicates().tolist())
        DF_PRED = transform_page_num(DF_PRED)
        print(DF_PRED['page_num'].drop_duplicates().tolist())
        #Save the pred file to Blob storage
        #Post Processor On Premise
        #Call post processor
        try:
            
            qualScore = putil.document_quality_analysis(DF_PRED)
        except:
            qualScore = 0.0
            pass

        #get vendorname if available and pass it to post-processor
        print("CallbackUrl: ",callbackUrl)
        docMetaData = putil.getDocumentApi(documentId, callbackUrl)
        print("docMetaData is \n{}".format(docMetaData))
        ### 28 March 2023 added discrepancy not identification & removing form df 
        if docMetaData:
            import ProcessDiscrepancyNote as discr
			#changes made here to reject blank disc
            DF_PRED,discr_note,discrMetaData,blank_disc_note = discr.process_discr_note(DF_PRED,docMetaData)
            #Added a flag to check blank_disc_note
            if blank_disc_note:
                if 'result' in docMetaData:
                    if 'document' in docMetaData['result']:
                        if 'documentId' in docMetaData['result']['document']:
                            blank_disc_doc_id = docMetaData['result']['document']['documentId']
                            docApiInfo = apiInit(documentId)
                            docApiInfo["request"]["discNoteAutoDeleted"] = 1
                            
                            #Deleting unwanted keys
                            if 'status' in docApiInfo['request']:
                                del docApiInfo["request"]["status"]
                            if 'statusMsg' in docApiInfo['request']:
                                del docApiInfo["request"]["statusMsg"]
                            if 'stage' in docApiInfo['request']:
                                del docApiInfo["request"]["stage"]
                            # print(docApiInfo,"Info")
                            updated_invoice,resp_code = putil.updateDocumentApi_New(blank_disc_doc_id,
                                                                                    docApiInfo,
                                                                                    callbackUrl)
                            print("Updated Blank Disc Note Status",resp_code)




                print("Blank disc note identified",)
            if discr_note:
                status,path = putil.uploadFilesToBlobStore([discr_note])
                if status:
                    print("path :",path, type(path))
                    discr_note = "/".join(path[0].split("/")[1:])
                    print("discr note path:",discr_note)
                    try:
                        if 'result' in docMetaData:
                            if 'document' in docMetaData['result']:
                                if 'documentId' in docMetaData['result']['document']:
                                    invoice_doc_id = docMetaData['result']['document']['documentId']
                                    docApiInfo = apiInit(documentId)
                                    docApiInfo["request"]["discrNotePresent"] = 1
                                    docApiInfo["request"]["discrNoteFilePath"] = path[0]
                                    docApiInfo["request"]["discrNoteDocumentId"] = invoice_doc_id + "_DISCR"
                                    #Deleting unwanted keys
                                    if 'status' in docApiInfo['request']:
                                        del docApiInfo["request"]["status"]
                                    if 'statusMsg' in docApiInfo['request']:
                                        del docApiInfo["request"]["statusMsg"]
                                    if 'stage' in docApiInfo['request']:
                                        del docApiInfo["request"]["stage"]
                                    # print(docApiInfo,"Info")
                                    updated_invoice,resp_code = putil.updateDocumentApi_New(invoice_doc_id,
                                                                                            docApiInfo,
                                                                                            callbackUrl)
                                    print("Updated Disc Note Present Status",resp_code)
                    except Exception as e:
                        print("Exception in updating status:", e)
        ### 28 March 2023 added discrepancy not identification & removing form df 

        ## PBAIP-27 - Sahil 5-June-2024 (Added logic for handling only DN Invoice) -Code starts
        if discr_note:
            print("DN is present. DataFrame shape is:", DF_PRED.shape, "and discr note is:", discr_note)
            if DF_PRED.shape[0] == 0:
                ## Logic to move Invoice to deleted section
                ## Raise an exception, Only DN is present
                raise Exception("Only DN is Present")
        ## PBAIP-27 - Sahil 5-June-2024 (Added logic for handling only DN Invoice) -Code ends    

        if not docMetaData:
            dict_final,stp_score,overall_score, vendor_id = pp.post_process(DF_PRED)
        else:
            dict_final,stp_score,overall_score, vendor_id = pp.post_process(DF_PRED,
                                                                            docMetaData=docMetaData)
        print("Post Processor Executed")
        stp = False
        if stp_score > 0:
            stp = True
        #30 March Added Ingestion of discrepency note after Extraction of invoice 
        if discr_note:
            # discrMetaData["result"]["document"]["linked_document"]["vendorGSTIN"] = dict_final.get("vendorGSTIN")
            # discrMetaData["result"]["document"]["linked_document"]["documentId"] = documentId
            upload = discr.ingest_discreopancy_note(discr_file=discr_note,meta_data=discrMetaData)
            print("dict_final type :",type(dict_final))
            if docMetaData :
                if docMetaData.get("result").get("document").get("invNumber")!= None and docMetaData.get("result").get("document").get("invNumber")!= "":
                    doc_invNumber = docMetaData.get("result").get("document").get("invNumber")
                else:
                    doc_invNumber = None
            upload_status = upload.send_request(dict_final,documentId,doc_invNumber,stp)
            print("upload_status :",upload_status)
            if (upload_status.get("result") != None) and (upload_status.get("result").get("status_code") != None) and (upload_status.get("result").get("status_code") != 200):
                print("Returning None since Discr note is not uploaded")
                # updated = updateFailure(stgExtract,
                #                         statusFailed,
                #                         errCode,
                #                         errmsgExtractionUpdateFail,
                #                         documentId,
                #                         callbackUrl,
                #                         auth_token)
                return None
            # Setting stp to false till discr is not processed
            print("Not changing STP flag for Invoice")
            # stp = False
        print("Post Processor Build Json Called ", dict_final)
        json_dict = pp.build_final_json(dict_final)
        print("Post Processor Build Json Executed",
              json_dict)
        
        ## Added STP conditions for Discrepancy Note based on Invoice slabs
        if (docMetaData != None) and (docMetaData.get("result")!= None) and (docMetaData.get("result").get("document")!= None):
            if docMetaData.get("result").get("document").get("docType") == "Discrepancy Note":
                print(f"DocType is Discrepancy Note. Checking STP conditions")
                count = 0
                if (docMetaData.get("result").get("document").get("linked_document")!= None) and (docMetaData.get("result").get("document").get("linked_document").get("slabs")!= None):
                    subTotal_5_doc = float(dict_final.get("subTotal_5%").get("text"))
                    subTotal_12_doc = float(dict_final.get("subTotal_12%").get("text"))
                    subTotal_18_doc = float(dict_final.get("subTotal_18%").get("text"))
                    subTotal_28_doc = float(dict_final.get("subTotal_28%").get("text"))
                    subTotal_5_invoice = 0
                    subTotal_12_invoice = 0
                    subTotal_18_invoice = 0
                    subTotal_28_invoice = 0 
                    print("checking slabs")
                    for _idx, value in (docMetaData.get("result").get("document")["linked_document"]["slabs"]).items():
                        if _idx == "subTotal_5":
                            subTotal_5_invoice = float(value)
                        if _idx == "subTotal_12" :
                            subTotal_12_invoice = float(value)
                        if _idx == "subTotal_18" :
                            subTotal_18_invoice = float(value)
                        if _idx == "subTotal_28" :
                            subTotal_28_invoice = float(value)   
                        if float(value) > 0:
                            count+=1
                    if count>1:
                        print("Invoice Contains multiple slabs, so marking stp as false for discrepancy note")
                        stp = False
                    elif count == 1:
                        if (subTotal_5_invoice > 0) and ( (subTotal_12_doc > 0) or (subTotal_18_doc > 0) or (subTotal_28_doc > 0)):
                            print("Different slabs in both Invoice and discr note.")
                            stp = False
                        if (subTotal_12_invoice > 0) and ((subTotal_5_doc >0) or (subTotal_18_doc > 0) or (subTotal_28_doc > 0)):
                            print("Different slabs in both Invoice and discr note.")
                            stp = False
                        if (subTotal_18_invoice > 0 ) and ((subTotal_5_doc > 0) or (subTotal_12_doc > 0) or (subTotal_28_doc > 0)):
                            print("Different slabs in both Invoice and discr note.")
                            stp = False
                        if (subTotal_28_invoice > 0) and ((subTotal_5_doc > 0) or (subTotal_12_doc > 0) or (subTotal_18_doc > 0)):
                            print("Different slabs in both Invoice and discr note.")
                            stp = False
                    elif count == 0:
                        print("No tax slabs are present in Invoice, so marking STP as False.")
                        stp = False
                    
                    # if docMetaData.get("result").get("document").get("linked_document")
                if (docMetaData.get("result").get("document").get("linked_document")!= None) and (docMetaData.get("result").get("document").get("linked_document").get("cgstpresent")!= None):
                    cgst_present = docMetaData.get("result").get("document").get("linked_document").get("cgstpresent")
                    if cgst_present == -1:
                        print("Marking STP as False since slabs are not predicted in Invoice.")
                        stp =False
                    elif cgst_present == 1:
                        IGSTAmount_5_doc = float(dict_final.get("IGSTAmount_5%").get("text"))
                        IGSTAmount_12_doc = float(dict_final.get("IGSTAmount_12%").get("text"))
                        IGSTAmount_18_doc = float(dict_final.get("IGSTAmount_18%").get("text"))
                        IGSTAmount_28_doc = float(dict_final.get("IGSTAmount_28%").get("text"))
                        if (IGSTAmount_5_doc > 0) or (IGSTAmount_12_doc > 0) or (IGSTAmount_18_doc > 0) or (IGSTAmount_28_doc > 0):
                            print("invoice contains CGST, and Discr Note contain IGST. Marking STP as False")
                            stp = False
                    elif cgst_present == 0:
                        CGSTAmount_25_doc = float(dict_final.get("CGSTAmount_2.5%").get("text"))
                        CGSTAmount_6_doc = float(dict_final.get("CGSTAmount_6%").get("text"))
                        CGSTAmount_9_doc = float(dict_final.get("CGSTAmount_9%").get("text"))
                        CGSTAmount_14_doc = float(dict_final.get("CGSTAmount_14%").get("text"))
                        if (CGSTAmount_25_doc > 0) or (CGSTAmount_6_doc > 0) or (CGSTAmount_9_doc > 0) or (CGSTAmount_14_doc > 0):
                            print("invoice contains IGST, and Discr Note contain CGST. Marking STP as False")
                            stp = False 
        output = [json_dict, DF_PRED, overall_score * 100, stp, vendor_id, qualScore]

        print("Extraction and Post-Processor completed with overall score and stp {} and {}".format(overall_score,stp))
        return output

    except Exception as e:
        print("processExtraction",
              traceback.print_exc())
        if str(e) == "Only DN is Present":
            print("In Exception Block of processExtraction, raising another exception")
            raise Exception((str(e)))
        return None

def timeExpired(delta):

    # from datetime import datetime as dt
    from datetime import timedelta as td
    expired = False
    exp_time = dt.now()
    try:
        print("Delta type:", type(delta))
        if isinstance(delta, str):
            delta_time = dt.strptime(delta, '%H:%M:%S')
            time_delta = td(hours=delta_time.hour,
                            minutes=delta_time.minute,
                            seconds=delta_time.second)
            exp_time = dt.now() + time_delta
        elif isinstance(delta,dt):
            exp_time = delta
    
        t = dt.fromtimestamp(time.time())
        expired = t > exp_time

        print(t,exp_time,type(t),type(exp_time))

        return expired,exp_time
    except:
        print(traceback.print_exc())
        return True, exp_time



#new function to add raw prediction in new collection and deleted from doc info

# url ="http://106.51.73.100:9595/rawPrediction/add"


def form_request(metaData:dict=None)->dict:
    request = {"ver": "1.0",
               "params": {"msgid": ""},
               "request": {"documentId": "",
                           "rawPrediction":"[]",
                           "submittedOn": 1676526381000}}
    try:
        if metaData:
            meta = metaData.get("request",{})
            metaData = metaData.get("result",{})
            
            
            request["request"]['documentId']= meta.get("documentId","")
            request["request"]["rawPrediction"]= metaData.get("rawPrediction","[]")
            request["request"]['submittedOn']= meta.get("lastProcessedOn")
            print("payload formed success: ")
            return request
        else:
            print("metadata is none")
            return None
    except:
        print("form_request exception :",traceback.print_exc())
        return None
@putil.timing    
def add_raw_prediction(url:str,docInfo):
    
    payload=form_request(docInfo)
    if payload:
        Documentid = payload.get("request").get("documentId")
        print("add_raw_prediction Documentid",Documentid)
        retires = 0
        while retires <2:
            try:
                response = requests.post(url,json=payload,timeout=20)
                print("raw prediction added :",response.text)
                if response.status_code in [200,413,409]:
                    break   
                else:
                    retires = retires +1
                return response.json()
            except:
                retires = retires +1
    return 
                
def reorder_pages(data, pred_file_path):
    deep_copied_data = copy.deepcopy(data)
    try:
        DF_PRED = pd.read_csv(pred_file_path,
                                  index_col = 0)
        order = DF_PRED['page_num'].drop_duplicates().tolist()
        # Create a dictionary from the order list to map index to its position
        order_dict = {str(index): position for position, index in enumerate(order)}
        # Separate pages that are in the order list and those that are not
        ordered_pages = []
        remaining_pages = []
        for page in data:
            if page['index'] in order_dict:
                ordered_pages.append(page)
            else:
                remaining_pages.append(page)
        # Sort the ordered pages according to the order list
        ordered_pages.sort(key=lambda x: order_dict[x['index']])
        # Combine the sorted pages with the remaining pages in their original order
        data = ordered_pages + remaining_pages   
        return data
    except:
        print("exception in reorder_pages")
        return deep_copied_data
def addpagesfordisplay(callbackUrl,documentId,pages):
    deep_copied_data = copy.deepcopy(pages)
    try:
        doc_id = documentId.split('_DISCR')[0]
        url = f"{callbackUrl}/document/get/{doc_id}"
        response = requests.get(url)
        if response.status_code == 200:
            res = response.json()
            addpages =res['result']['document']['pages']
            #print(addpages)
            for ad_page in addpages:
                if ad_page not in deep_copied_data:
                   deep_copied_data.append(ad_page)
                else:
                    print(ad_page,'123')
        return deep_copied_data
    except:
        print("exception in addpagesfordisplay")
        return pages
# GetExtraction result 
@timing
def getExtractionResults(auth_token,
                         delta,
                         documentId,
                         callbackUrl,
                         sub_id):

    try:
        # import logging
        import pytz
        import datetime
        local_timezone = pytz.timezone('Asia/Kolkata')
        current_directory = os.getcwd()
        # file_name = str(documentId)+"_kafka.log"
        log_folder = os.path.join(current_directory, 'logs')
        putil.create_log_folder(log_folder)
        # log_file_kafka = os.path.join(log_folder,file_name)

        # app_log = setup_logging(log_file_kafka)
        file_name = str(documentId)+"_auth_token_"+str(auth_token)+".log"
        document_log_file = os.path.join(log_folder,file_name)
        document_log = get_logger(logger_name="document_log",log_file = document_log_file)

        dt_utc = datetime.datetime.utcnow()
        dt_local = dt_utc.replace(tzinfo=pytz.utc).astimezone(local_timezone)
        document_log.info(f"In tapp_client request received for doucment id {documentId} at time {dt_local}")
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        log_file_path = os.path.join(rootFolderPath,
                                     documentId + "_auth:" + auth_token  + ".log")
        log_file = open(log_file_path,"a")
        sys.stdout = log_file
        sys.stderr = log_file
    
        print("Log file kafka path is:", document_log_file)
        print("Time when request is received in tapp_client:", dt.now())

        document_log.debug(f"Starting get extraction result for document ID: {documentId} , with auth_token: {documentId}")

        #Check if extraction request has expired

        expired,exp_time = timeExpired(delta)
        dt_utc = dt.utcnow()
        import pytz
        local_timezone = pytz.timezone('Asia/Kolkata')
        dt_local = dt_utc.replace(tzinfo=pytz.utc).astimezone(local_timezone)
        print("Time when received req is in tapp_client:", dt_local)
        print("Expired time : ",expired, "Expected time :",exp_time)
        document_log.debug(f"Extraction expiraation time : {exp_time}, Status : {expired}")
        if expired:
            dt_utc = datetime.datetime.utcnow()
            dt_local = dt_utc.replace(tzinfo=pytz.utc).astimezone(local_timezone)
            document_log.info(f"In tapp_client for doucmentId '{documentId}' time is expired at time {dt_local}")
            updated = updateFailure(stgExtract,
                                    statusFailed,
                                    errCode,
                                    errmsgExtractionUpdateFail,
                                    documentId,
                                    callbackUrl,
                                    auth_token)
            return False
        docApiInfo = apiInit(documentId)    
        docRequest = docApiInfo["request"]
        print("Delta input",delta, "Current date time :")
        delta = datetime.datetime.strptime(delta, "%H:%M:%S")
        print("Delta strptime",delta)
        delta = delta - datetime.datetime(1900, 1, 1)
        print("Delta diff",delta)
        delta = delta.total_seconds()
        print("Delta time in Seconds", delta)
        delta = delta + 60
        print("Delta plus + 1 minutes : ",delta)
        dt_utc = datetime.datetime.utcnow()
        dt_local = dt_utc.replace(tzinfo=pytz.utc).astimezone(local_timezone)
        document_log.info(f"In tapp_client sending request for poll_status for doucment id {documentId} at time {dt_local}")
        document_log.debug(f"Polling time delta plus 1 minutes : {str(delta)}")
        Poll_result = poll_status(auth_token,
                                delta,
                                documentId,
                                sub_id)
        dt_utc = datetime.datetime.utcnow()
        dt_local = dt_utc.replace(tzinfo=pytz.utc).astimezone(local_timezone)
        document_log.info(f"In tapp_client poll_status returned with status code {Poll_result} at time {dt_local}")
        print("Poll_result ", Poll_result)
        dt_utc = dt.utcnow()
        dt_local = dt_utc.replace(tzinfo=pytz.utc).astimezone(local_timezone)
        print("Time after polling is done",dt_local)
        time_total = time.time()
        if Poll_result == None:
            # document_log.info(f"Poll_result is None : {Poll_result}")
            #Update document metadata to failure status
            updated = updateFailure(stgExtract,
                                    statusFailed,
                                    errCode,
                                    errmsgExtractionUpdateFail,
                                    documentId,
                                    callbackUrl,
                                    auth_token)
            print("Error updated ")
            document_log.info(f"Polling stautus is None failed updaated : {updated}")
            
            return False
        else:
            print("Result generated", Poll_result)
            ext_status = Poll_result["status"]
            document_log.debug(f"Polling Status: {ext_status}")
            print("Extraction Status:", ext_status)
            if ext_status == "Extracted":
                document_log.info(f"Processing result started")

                #Extraction is complete. So, get the result from "body"
                result_json = Poll_result["result"]
                #This result must be having the pred file path
                result = result_json
                #Download the pred file
                blob_pred_path = result["pred_file"]
                fileName = os.path.basename(blob_pred_path)
                local_pred_path = os.path.join(rootFolderPath,
                                            os.path.splitext(fileName)[0] +
                                            "_pred" +
                                            os.path.splitext(fileName)[1])
                pages = result["pages"]

                storageType = cfg.getStorageType()
                if storageType.upper() == "BLOB":
                    for page_index,page in enumerate(pages):
                        pngUrl = page["url"]
                        pages[page_index]["pngURI"] = pngUrl
                        # pages[page_index]["url"] = "preprocessor/" + filename
                else:
                    pngLocations = []
                    pngLocalPaths = []
                    print("Current Working Directory 1:",
                        os.getcwd())
                    uiRoot = cfg.getUIRootFolder()
                    print("Current Working Directory 2:",
                        os.getcwd())
                    png_folder = os.path.join(uiRoot,
                                            "preprocessor")
                    print("Current Working Directory:", os.getcwd())
                    print("png folder:",
                        png_folder)
                    os.makedirs(png_folder,
                                exist_ok=True)
                    for page_index,page in enumerate(pages):
                        pngUrl = page["url"]
                        filename = os.path.basename(pngUrl)
                        pngLocations.append(pngUrl)
                        localFilePath = os.path.join(png_folder,
                                                    filename)
                        pngLocalPaths.append(localFilePath)
                        pages[page_index]["pngURI"] = "preprocessor/" + filename
                        pages[page_index]["url"] = "preprocessor/" + filename
    
                    blob_downloads = [blob_pred_path] + pngLocations
                    local_downloads = [local_pred_path] + pngLocalPaths
                    print("download folders: ",
                        blob_downloads,
                        local_downloads)
                    downloads = zip(blob_downloads,local_downloads)
    
                    downloaded = putil.downloadFilesFromBlob(downloads)
                    print("Pred File Downloaded", downloaded)
                    if not downloaded:
                        #Update document metadata to failure status
                        updated = updateFailure(stgExtract,
                                                statusFailed,
                                                errCode,
                                                errmsgExtractionUpdateFail,
                                                documentId,
                                                callbackUrl,
                                                auth_token)
                        return False

                #Update pred file with new rules to extract critical header fields
                #predHdrUsingFuzzy(df)

                #If it's a successful call, go to post-processor
                print(pages,'pages123')
                pages = reorder_pages(pages,local_pred_path)
                pages = addpagesfordisplay(callbackUrl,documentId,pages)
                print(pages,'pages123')
                docApiInfo["request"]["pages"] = pages
                dt_utc = datetime.datetime.utcnow()
                dt_local = dt_utc.replace(tzinfo=pytz.utc).astimezone(local_timezone)
                if (Poll_result.get("result") != None) and (Poll_result.get("result").get("error") != None) and (Poll_result.get("result").get("error") == "Blank Document"):
                    print("Blank Document Uploaded")
                    ## Logic to auto-delete the documents
                    updateDeleted(
                        documentId,
                        "Auto Deleted: Blank or Blur Document",
                        callbackUrl,
                        auth_token
                    )
                    return True
                document_log.info(f"In tapp_client sending request for processExtraction for doucment id {documentId} at time {dt_local}")
                output = processExtraction(local_pred_path,
                                        documentId,
                                        callbackUrl)
                docInfo = processExtRes(output,
                                        docApiInfo)
                
            
                #print("o/p of process Extraction Response",docInfo)

                # Remove rawPrediction from docInfo
                # Add rawPrediction to new collection if Page Count is less than equal to 3
                # Delete rawPrediction form docInfo['result']


                if "rawPrediction" in docInfo['result']:
                    print("Adding rawPrediction to new collection")
                    #By hari and chaitra, False has been added to avoid raw prediction getting added even if extraction is failed -19/10/2023
                    
                    if docInfo["pages_ocred"]<=3 and False:
                        print("More than 2 pages in document!!!")
                        callbackUrl_ = cfg.getUIServer()
                        callbackUrl_add = callbackUrl_+"/rawPrediction/add"

                        added_detail=add_raw_prediction(callbackUrl_add,docInfo)
                        print("Return from add_raw_prediction")
                    else:
                        print("page limit is more, raw prediction will not be added")
                        pass
                        
                    del docInfo["result"]["rawPrediction"]
                    print("Raw prediction Deleted",docInfo)

                if docInfo is None:
                    updated = updateFailure(stgExtract,
                                            statusFailed,
                                            errCode,
                                            errmsgExtractionUpdateFail,
                                            documentId,
                                            callbackUrl,
                                            auth_token)
                    return False

                print("output json:",output,
                    "docInfo json:",docInfo,
                    "docRequest:",docRequest)
                docInfo["qualityScore"]=output[5]
                docRequest, resultApiInfo = prepare_request_ML(docRequest,
                                                            docInfo)
            
                # print("prepare request ML",docRequest,resultApiInfo)
                if (docRequest["stp"] == True) or (str(docRequest["stp"]).lower() == "true"):
                    status = statusRevComp
                else:
                    status = statusReview
                #Aug 05 2022 - Add number of pages OCRed in mongo db
                # docInfo["request"]["pages_ocred"] = len(pages)
                #Aug 05 2022 - Add number of pages OCRed in mongo db
                dt_utc = datetime.datetime.utcnow()
                dt_local = dt_utc.replace(tzinfo=pytz.utc).astimezone(local_timezone)
                document_log.info(f"In tapp_client sending request for updateSuccess for doucment id {documentId} at time {dt_local}")
                updated = updateSuccess(status,
                                        stgExtract,
                                        resultApiInfo,
                                        docInfo,
                                        documentId,
                                        auth_token,
                                        callbackUrl)
                print("Success Api Update", updated)
                document_log.debug(f"Result uodate status : {updated}")
                document_log.info(f"Total time taken for generating big auth file after polling is: {time.time()-time_total}")
                if not updated:
                    dt_utc = datetime.datetime.utcnow()
                    dt_local = dt_utc.replace(tzinfo=pytz.utc).astimezone(local_timezone)
                    document_log.info(f"In tapp_client sending request for updateFailure for Scenario1 for doucment id {documentId} at time {dt_local}")
                    fail_updated = updateFailure(stgExtract,
                                                statusFailed,
                                                errCode,
                                                errmsgExtractionUpdateFail,
                                                documentId,
                                                callbackUrl,
                                                auth_token)
                    return False
                return True
            else:
                dt_utc = datetime.datetime.utcnow()
                dt_local = dt_utc.replace(tzinfo=pytz.utc).astimezone(local_timezone)
                document_log.info(f"In tapp_client sending request for updateFailure for Scenario2 for doucment id {documentId} at time {dt_local}")
                updated = updateFailure(stgExtract,
                                        statusFailed,
                                        errCode,
                                        errmsgExtractionUpdateFail,
                                        documentId,
                                        callbackUrl,
                                        auth_token)
                document_log.debug(f"Update extraction failure : {updated}")
                return False

    except Exception as e:
        print("getExtractionResults",
            traceback.print_exc())
        dt_utc = datetime.datetime.utcnow()
        dt_local = dt_utc.replace(tzinfo=pytz.utc).astimezone(local_timezone)
        document_log.info(f"In tapp_client sending request for updateFailure for Scenario3 for doucment id {documentId} at time {dt_local}")
        
        if str(e) == "Only DN is Present":
            ## Logic to auto-delete the documents
            updateDeleted(
                documentId,
                "Auto Deleted: Only DN is present",
                callbackUrl,
                auth_token
            )
            return True
        updated = updateFailure(stgExtract,
                                statusFailed,
                                errCode,
                                errmsgExtractionUpdateFail,
                                documentId,
                                callbackUrl,
                                auth_token)
        print("System Exception", updated)
        return False
    finally:
        document_log.info(f"Get extraction result completed")
        pass
        try:
            if log_file is not None:
                # print("Time taken to uplaod file is:",time.time()-time_1)
                dt_utc = datetime.datetime.utcnow()
                dt_local = dt_utc.replace(tzinfo=pytz.utc).astimezone(local_timezone)
                document_log.info(f"In tapp_client sending request for uploadFilesToBlobStore for doucment id {documentId} at time {dt_local}")
                uploaded,orgFileLocation = putil.uploadFilesToBlobStore([log_file_path])
                if uploaded:
                    os.remove(log_file_path)
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                log_file.close()
        except:
            dt_utc = datetime.datetime.utcnow()
            dt_local = dt_utc.replace(tzinfo=pytz.utc).astimezone(local_timezone)
            document_log.info(f"In tapp_client got exception in finally block for doucment id {documentId} at time {dt_local}")
            pass
        finally:
            # document_log.close_file_handler(log_file_kafka)
            pass
        #Remove other files from local drive
        try:
            os.remove(local_pred_path)
        except:
            pass

