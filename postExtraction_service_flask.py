# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 14:52:45 2022

@author: DELL
"""

import traceback
from sys import argv
import json
import TAPPconfig as cfg

from client_rules import bizRuleValidateForUi as BR
import corrections as corr
import path_finder as pfi
import format_identifier as fmi
import collect_metadata as cmda
import dashboard as dsh
import ui_downloads
import os
import pandas as pd
# from klein import Klein
# app = Klein()

from flask import Flask, request, Response
# import flask
app = Flask(__name__)

appPort = cfg.getPostExtractionSvcPort()
svcIp = cfg.getPostExtractionSvcIP()
UI_url = cfg.getUIServer()
vendorMasterDataPath = cfg.getVendorMasterData()
masterFilePath = os.path.join(vendorMasterDataPath)
VENDOR_MASTERDATA = pd.read_csv(masterFilePath, encoding='unicode_escape')
@app.route('/document/BizRuleValidate',methods = ['POST'])
def docBizRuleValidate():
    
    def exceptionResponse(document_result_updated = None, docResult = None):
        return json.dumps({"status_code":200,
                           "list_fields":[],
                           "document_result_updated":document_result_updated,
                           "docResult":docResult},
                          indent = 4,
                          sort_keys = False,
                          default = str)

    def successResponse(document_result_updated = None, docResult = None):
        return json.dumps({"status_code":200,
                           "list_fields":[],
                           "document_result_updated":document_result_updated,
                           "docResult":docResult},
                          indent = 4,
                          sort_keys = False,
                          default = str)

    try:
        # flask.Response()
        # request.responseHeaders.addRawHeader(b"Content-Type",
        #                                      b"application/json")
        # rawContent = request.content.read()
        # encodedContent = rawContent.decode("utf-8")
        # content = json.loads(encodedContent)
        content = json.loads(request.data,
                             strict = False)
        print("Input from the UI:\n ", content)

        documentId = content["documentId"]
        # 12 sep 22  CR to make Vendor GSTIN and PAN optional
        UI_validation = content["UI_validation"]
        # 12 sep 22  CR to make Vendor GSTIN and PAN optional

        #Jul 05 2022 - DO NOT USE Callback Url to update back to UI.
        #Use a configured Url to write back to UI
        # callBackUrl = content["callBackUrl"]
        callBackUrl = UI_url
        #Jul 05 2022 - DO NOT USE Callback Url to update back to UI.
        #Use a configured Url to write back to UI
        print("Input from the UI:\n ", content)
        # 12 sep 22  CR to make Vendor GSTIN and PAN optional
        # resp = BR(documentId,
        #     callBackUrl)
        document_result_updated = 0
        docResult = None
        resp, document_result_updated, docResult = BR(documentId,
            callBackUrl,UI_validation, VENDOR_MASTERDATA)
        # 12 sep 22  CR to make Vendor GSTIN and PAN optional
        
        if resp is not None:
            if len(resp) == 0:
                print("response of biz rule validation",resp)
                return Response(successResponse(document_result_updated, docResult),
                                mimetype="application/json")
            else:
                resp_ = {}
                resp_["status_code"] = 500
                resp_["document_result_updated"] = document_result_updated
                resp_["docResult"] = docResult
                resp_["list_fields"] = []
                for res in resp:
                    r = {}
                    r["fieldId"] = res[0]
                    r["error_message"] = res[1]
                    ## 11 Jan 2024 Added warning flag
                    if len(res)==3:
                        r["warning_flag"] = res[2]
                    resp_["list_fields"].append(r)
                resp_ = json.dumps(resp_,
                                   indent = 4,
                                   sort_keys = False,
                                   default = str)
                return Response(resp_,
                                mimetype="application/json")
        else:
            return Response(exceptionResponse(document_result_updated, docResult),
                            mimetype="application/json")
    except:
        print("docBizRuleValidate",
              traceback.print_exc())
        return Response(exceptionResponse(document_result_updated, docResult),
                        mimetype="application/json")

@app.route('/corrections/getOCRLines', methods=['POST'])
def get_ocr_lines():
    try:
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
        content = json.loads(request.data,
                             strict = False)
        # response = corr.get_ocr_lines(request)
        response = corr.get_ocr_lines(content)
        # return response
        return Response(response,
                        mimetype="application/json")
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
    except:
        print("get_ocr_lines",
              traceback.print_exc())
        response_object = {}
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
        # request.responseHeaders.addRawHeader(b"content-type", b"application/json")
        response = json.dumps(response_object,
                              indent=4,
                              sort_keys=False,
                              default=str)
        # return response
        return Response(response,
                        mimetype="application/json")
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app

@app.route('/corrections/getOCRLinesTemp', methods=['POST'])
def get_ocr_lines_temp():
    try:
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
        print("All we received",
              request.data)
        content = json.loads(request.data,
                             strict = False)
        # response = corr.get_ocr_lines_temp(request)
        response = corr.get_ocr_lines_temp(content)
        # return response
        return Response(response,
                        mimetype="application/json")
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
    except:
        print("get_ocr_lines_temp",
              traceback.print_exc())
        response_object = {}
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
        # request.responseHeaders.addRawHeader(b"content-type", b"application/json")
        response = json.dumps(response_object,
                              indent = 4,
                              sort_keys = False,
                              default=str)
        # return response
        return Response(response,
                        mimetype="application/json")
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app


@app.route('/path_finder/get_list_templates', methods=['POST'])
def get_list_templates(request):
    try:
        content = json.loads(request.data,
                             strict = False)
        response = pfi.get_list_templates(content)
        return Response(response,
                        mimetype="application/json")
    except:
        print("get_templates",
              traceback.print_exc())
        response_object = {}
        response_object['status'] = "Failure"
        response_object['responseCode'] = 500
        response_object['templates'] = []
        response_object['message'] = "Failure"
        response = json.dumps(response_object,
                              indent = 4,
                              sort_keys = False,
                              default = str)
        # return response
        return Response(response,
                        mimetype="application/json")


@app.route('/path_finder/get_templates', methods=['POST'])
def get_templates():
    try:
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
        content = json.loads(request.data,
                             strict = False)
        # response = pfi.get_templates(request)
        response = pfi.get_templates(content)
        # return response
        return Response(response,
                        mimetype="application/json")
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
    except:
        print("get_templates",
              traceback.print_exc())
        response_object = {}
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        response_object['templates'] = []
        response_object['message'] = "Failure"
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
        # request.responseHeaders.addRawHeader(b"content-type",
        #                                      b"application/json")
        response = json.dumps(response_object,
                              indent = 4,
                              sort_keys = False,
                              default = str)
        # return response
        return Response(response,
                        mimetype="application/json")
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app

@app.route('/path_finder/test_templates', methods=['POST'])
def test_templates():
    try:
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
        content = json.loads(request.data,
                             strict = False)
        # response = pfi.test_templates(request)
        response = pfi.test_templates(content)
        # return response
        return Response(response,
                        mimetype="application/json")
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
    except:
        print("test_templates",
              traceback.print_exc())
        response_object = {}
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        response_object["extracted_value"] = []
        response_object['message'] = "Failure"
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
        # request.responseHeaders.addRawHeader(b"content-type",
        #                                      b"application/json")
        response = json.dumps(response_object,
                              indent = 4,
                              sort_keys = False,
                              default = str)
        # return response
        return Response(response,
                        mimetype="application/json")
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
        
@app.route('/path_finder/validate_template', methods=['POST'])
def validate_template():
    try:
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
        content = json.loads(request.data,
                             strict = False)
        # response = pfi.validate_template(request)
        response = pfi.validate_template(content)
        # return response
        return Response(response,
                        mimetype="application/json")
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
    except:
        print("validate_template",
              traceback.print_exc())
        response_object = {}
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        response_object['extracted_value'] = {}
        response_object['message'] = "Failure"
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
        # request.responseHeaders.addRawHeader(b"content-type",
        #                                      b"application/json")
        response = json.dumps(response_object,
                              indent = 4,
                              sort_keys = False,
                              default = str)
        # return response
        return Response(response,
                        mimetype="application/json")
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app

@app.route('/path_finder/create_templates', methods=['POST'])
def insert_template(request):
    try:
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
        content = json.loads(request.data,
                             strict = False)
        # response = pfi.insert_template(request)
        response = pfi.insert_template(content)
        # return response
        return Response(response,
                        mimetype="application/json")
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
    except:
        print("insert_template",
              traceback.print_exc())
        response_object = {}
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        response_object['message'] = "Failure"
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
        # request.responseHeaders.addRawHeader(b"content-type",
        #                                      b"application/json")
        response = json.dumps(response_object,
                              indent = 4,
                              sort_keys = False,
                              default = str)
        # return response
        return Response(response,
                        mimetype="application/json")
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app

@app.route('/format_identifier/refresh_format', methods=['POST'])
def refresh_format():
    try:
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
        content = json.loads(request.data,
                             strict = False)
        # response = fmi.refresh_format(request)
        response = fmi.refresh_format(content)
        # return response
        return Response(response,
                        mimetype="application/json")
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
    except:
        print("refresh_format",
              traceback.print_exc())
        response_object = {}
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        response_object['refreshed_result'] = None
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
        # request.responseHeaders.addRawHeader(b"content-type",
        #                                      b"application/json")
        response = json.dumps(response_object,
                              indent = 4,
                              sort_keys = False,
                              default = str)
        # return response
        return Response(response,
                        mimetype="application/json")
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app

@app.route('/format_identifier/get_suggestion', methods=['POST'])
def get_suggested_masterdata():
    try:
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
        content = json.loads(request.data,
                             strict = False)
        # response = fmi.get_suggested_masterdata(request)
        response = fmi.get_suggested_masterdata(content)
        # return response
        return Response(response,
                        mimetype="application/json")
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
    except:
        print("get_suggested_masterdata",
              traceback.print_exc())
        response_object = {}
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        response_object['master_data'] = None
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
        # request.responseHeaders.addRawHeader(b"content-type",
        #                                      b"application/json")
        response = json.dumps(response_object,
                              indent = 4,
                              sort_keys = False,
                              default = str)
        # return response
        return Response(response,
                        mimetype="application/json")
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app

@app.route('/format_identifier/create', methods=['POST'])
def insert_masterdata_wrapper():
    try:
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
        content = json.loads(request.data,
                             strict = False)
        # response = fmi.insert_masterdata_wrapper(request)
        response = fmi.insert_masterdata_wrapper(content)
        # return response
        return Response(response,
                        mimetype="application/json")
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
    except:
        print("insert_template",
              traceback.print_exc())
        response_object = {}
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        response_object['message'] = "Failure"
        response_object['master_data'] = None
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
        # request.responseHeaders.addRawHeader(b"content-type",
        #                                      b"application/json")
        response = json.dumps(response_object,
                              indent = 4,
                              sort_keys = False,
                              default = str)
        # return response
        return Response(response,
                        mimetype="application/json")
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app


@app.route('/format_identifier/validate', methods=['POST'])
def validate_masterdata_wrapper():
    try:
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
        content = json.loads(request.data,
                             strict = False)
        # response = fmi.validate_masterdata_wrapper(request)
        response = fmi.validate_masterdata_wrapper(content)
        # return response
        return Response(response,
                        mimetype="application/json")
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
    except:
        print("validate_masterdata_wrapper",
              traceback.print_exc())
        response_object = {}
        response_object['validate_result'] = "INVALID"
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        response_object['score'] = 0.0
        response_object['message'] = "Failure"
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
        # request.responseHeaders.addRawHeader(b"content-type",
        #                                      b"application/json")
        response = json.dumps(response_object,
                              indent = 4,
                              sort_keys = False,
                              default = str)
        # return response
        return Response(response,
                        mimetype="application/json")
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app

@app.route('/collect_metadata', methods=['POST'])
def collect_metadata_wrapper():
    try:
        content = json.loads(request.data,
                             strict = False)
        response = cmda.collect_metadata_wrapper(content)
        return Response(response,
                        mimetype="application/json")
    except:
        print("collect_metdata_wrapper",
              traceback.print_exc())
        response_object = {}
        response_object['validate_result'] = "INVALID"
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        response_object['message'] = "Failure"
        response = json.dumps(response_object,
                              indent = 4,
                              sort_keys = False,
                              default = str)
        # return response
        return Response(response,
                        mimetype="application/json")
        

@app.route('/populate_dashboard', methods=['POST'])
def populate_dashboard_wrapper():
    try:
        content = json.loads(request.data,
                             strict = False)
        response = dsh.populate_dashboard_wrapper(content)
        return Response(response,
                        mimetype="application/json")
    except:
        print("populate_dashboard_wrapper",
              traceback.print_exc())
        response_object = {}
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        response_object['message'] = "Failure"
        response = json.dumps(response_object,
                              indent = 4,
                              sort_keys = False,
                              default = str)
        
        return Response(response,
                        mimetype="application/json")


@app.route('/populate_dashboard/get_vendors', methods=['POST'])
def get_vendors_wrapper():
    try:
        content = json.loads(request.data,
                             strict = False)
        response = dsh.get_vendors_wrapper(content)
        return Response(response,
                        mimetype="application/json")
    except:
        print("populate_dashboard_wrapper",
              traceback.print_exc())
        response_object = {}
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        response_object['message'] = "Failure"
        response = json.dumps(response_object,
                              indent = 4,
                              sort_keys = False,
                              default = str)
        
        return Response(response,
                        mimetype="application/json")


@app.route('/populate_dashboard/get_billing_units', methods=['POST'])
def get_billing_units_wrapper():
    try:
        content = json.loads(request.data,
                             strict = False)
        response = dsh.get_billing_units_wrapper(content)
        return Response(response,
                        mimetype="application/json")
    except:
        print("get_billing_units_wrapper",
              traceback.print_exc())
        response_object = {}
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        response_object['message'] = "Failure"
        response = json.dumps(response_object,
                              indent = 4,
                              sort_keys = False,
                              default = str)
        
        return Response(response,
                        mimetype="application/json")


@app.route('/ui_downloads/list_view', methods=['POST'])
def download_list_view_data():
    try:
        content = json.loads(request.data,
                             strict = False)
        response = ui_downloads.download_list_view_data(content)
        return response
    except:
        print("download_list_view_data",
              traceback.print_exc())
        response_object = {}
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        response_object['message'] = "Failure"
        response = json.dumps(response_object,
                              indent = 4,
                              sort_keys = False,
                              default = str)
        return Response(response,
                        mimetype="application/json")


if __name__ == "__main__":
    if len(argv) > 1:
        appPort = int(argv[1])
        print(appPort)
    #Jun 23, 2022 - run the service only in localhost
    # app.run("0.0.0.0", appPort)
    app.run(svcIp, appPort)
    #Jun 23, 2022 - run the service only in localhost
