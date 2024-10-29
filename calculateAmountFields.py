#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 14:00:34 2022

@author: Parmesh
"""
import pandas as pd
import math
import traceback
import re
from operator import itemgetter
import copy
import string
# def taxOnlyAmts(max_amt,cgst,amts):

#     second_max = amts[0]

#     for i in range(2):
#         if second_max < 0.84 * max_amt:
#             if len(amts) > 1 and i == 0:
#                 second_max = amts[1]
#                 continue
#             return {}
#         elif (max_amt != second_max) and math.isclose(max_amt,
#                                                       second_max,
#                                                       abs_tol = 0.8):
#             if len(amts) > 1 and i == 0:
#                 second_max = amts[1]
#                 continue
#             return {}

#         totalGst = max_amt - second_max
#         # print("CGST",
#         #       cgst,
#         #       totalGst,
#         #       max_amt,
#         #       second_max)
#         # commented after first 15 test
#         # if totalGst < second_max * .024:
#         #     if len(amts) > 1 and i == 0:
#         #         second_max = amts[1]
#         #         continue
#         #     return {}

#         totalGstPresent = True in (math.isclose(totalGst,
#                                                 amt,
#                                                 abs_tol = 1.0) for amt in amts)
#         gst = totalGst
#         if cgst == 1:
#             gst = totalGst / 2.0
#         elif cgst == -1:
#             gstPresent = True in (math.isclose(gst,
#                                                amt,
#                                                abs_tol = 1.0) for amt in amts)
#             if not gstPresent:
#                 gst = totalGst / 2.0
#                 gstPresent = True in (math.isclose(gst,
#                                                    amt,
#                                                    abs_tol = 1.0) for amt in amts)
#                 if gstPresent:
#                     cgst = 1
#             else:
#                 cgst = 0

#         # print("GST",cgst,gst,gstPresent)

#         gstPresent = True in (math.isclose(gst,
#                                            amt,
#                                            abs_tol = 1.0) for amt in amts)
#         # print("CGST",
#         #       cgst,
#         #       totalGst,
#         #       max_amt,
#         #       second_max,
#         #       gstPresent)
#         if not gstPresent and not totalGstPresent:
#             if len(amts) > 1 and i == 0:
#                 second_max = amts[1]
#                 continue
#             return {}
#         else:
#             if cgst:
#                 return {"total":max_amt,
#                         "subTotal":second_max,
#                         "cgst":gst,
#                         "sgst":gst,
#                         "igst":0.0,
#                         "totalGst":totalGst}
#             else:
#                 return {"total":max_amt,
#                         "subTotal":second_max,
#                         "cgst":0.0,
#                         "sgst":0.0,
#                         "igst":gst,
#                         "totalGst":totalGst}
vendorGSTIN_list = ['27AAFCD3317F1ZY','37AAFCD3317F1ZX','24AAFCD3317F1Z4','23AAFCD3317F1Z6','19AAFCD3317F1ZV',
'04AAFCD3317F1Z6','32AAFCD3317F1Z7','33AAFCD3317F1Z5','29AAFCD3317F1ZU','36AAFCD3317F1ZZ','06AAFCD3317F1Z2']

## Issue# 170 - Sahil -26 April-2024 (Improvement in Total Amount Accuracy) -Code starts
vendor_PAN_other_max_amts = {"AAACW4202F": "due"}
## Issue# 170 - Sahil -26 April-2024 (Improvement in Total Amount Accuracy) -Code ends

def taxOnlyAmts(max_amt,cgst,amts):

    for i in range(3):
        second_max = amts[i]
        if second_max < 0.84 * max_amt:
            if len(amts) > 1 and i <= 1:
                continue
            return {}
        elif (max_amt != second_max) and math.isclose(max_amt,
                                                      second_max,
                                                      abs_tol = 0.8):
            if len(amts) > 1 and i <= 1:
                continue
            return {}

        totalGst = max_amt - second_max
        print("CGST",
              cgst,
              totalGst,
              max_amt,
              second_max,
              totalGst < second_max * .024)

        totalGstPresent = True in (math.isclose(totalGst,
                                                amt,
                                                abs_tol = 1.0) for amt in amts)
        gst = totalGst
        if cgst == 1:
            gst = totalGst / 2.0
        elif cgst == -1:
            gstPresent = True in (math.isclose(gst,
                                               amt,
                                               abs_tol = 1.0) for amt in amts)
            if not gstPresent:
                gst = totalGst / 2.0
                gstPresent = True in (math.isclose(gst,
                                                   amt,
                                                   abs_tol = 1.0) for amt in amts)
                if gstPresent:
                    cgst = 1
            else:
                cgst = 0

        gstPresent = True in (math.isclose(gst,
                                           amt,
                                           abs_tol = 1.0) for amt in amts)
        print("CGST",
              cgst,
              totalGst,
              max_amt,
              second_max,
              gstPresent)
        if not gstPresent and not totalGstPresent:
            if len(amts) > 1 and i <= 1:
                continue
            return {}
        else:
            if cgst:
                return {"total":max_amt,
                        "subTotal":second_max,
                        "cgst":gst,
                        "sgst":gst,
                        "igst":0.0,
                        "totalGst":totalGst}
            else:
                return {"total":max_amt,
                        "subTotal":second_max,
                        "cgst":0.0,
                        "sgst":0.0,
                        "igst":gst,
                        "totalGst":totalGst}
# 3 march 2023 Added functionality for prediction of cgst if there are more than 2 GSTIN before final_prediction
def check_if_cgst_v1(df,prediction=None,docMetaData= None):
    # 20 March 2023, Added for discrepancy note only
    #print("DocMetadata for Identifying CGST/IGST:",docMetaData)
    
    
    if docMetaData != None and docMetaData.get("result")!= None and docMetaData.get("result").get("document").get("docType") == "Discrepancy Note":
        discr_data = docMetaData.get("result").get("document").get("linked_document")
        if discr_data != None:
            cgst = -1
            vendorGSTIN = discr_data.get("vendorGSTIN")
            shippingGSTIN = discr_data.get("shippingGSTIN")
            billingGSTIN = discr_data.get("billingGSTIN")
            if vendorGSTIN == "" or vendorGSTIN == "N/A" or vendorGSTIN == None:
                isVendorGSTIN = False
            else:
                isVendorGSTIN = True
            if shippingGSTIN == "" or shippingGSTIN == "N/A" or shippingGSTIN == None:
                isShippingGSTIN = False
            else:
                isShippingGSTIN = True
            if billingGSTIN == "" or billingGSTIN == "N/A" or billingGSTIN == None:
                isBillingGSTIN = False
            else:
                isBillingGSTIN = True
            #isVendorGSTIN = (vendorGSTIN and (vendorGSTIN.get("text")!= '' and vendorGSTIN.get("text") is not None) and (vendorGSTIN.get("text") != "N/A"))
            #isBillingGSTIN = (billingGSTIN and (billingGSTIN.get("text")!= '' and billingGSTIN.get("text") is not None) and (billingGSTIN.get("text") != "N/A"))
            if (isVendorGSTIN and isShippingGSTIN):
                if vendorGSTIN[:2] == shippingGSTIN[:2]:
                    cgst = 1
                else:
                    cgst = 0 
            elif (isVendorGSTIN and isBillingGSTIN):
                if vendorGSTIN[:2] == billingGSTIN[:2]:
                    cgst = 1
                else:
                    cgst = 0 
            return cgst

    from business_rules import get_gstin_of
    import preProcUtilities as putil
    if prediction == None:
        cgst = -1
        try:
            #Cgst or Igst
            gst_list = list(set([putil.correct_gstin(s) for s in list(df[df["is_gstin_format"]==1]["text"].unique())]))
            print("Unique GSTIN in pred.csv",gst_list)
            noOfGSTIN = len(gst_list)

            if noOfGSTIN >= 2:
                print("Two or more GSTIN Prasent")
                if((noOfGSTIN==2)and(gst_list[0][:2] == gst_list[1][:2])):
                    cgst = 1
                elif((noOfGSTIN==2)and(gst_list[0][:2] != gst_list[1][:2])):
                    cgst = 0
                elif (noOfGSTIN >2):
                    VendorGSTIN = get_gstin_of(df,"vendorGSTIN")
                    ShippingGSTIN = get_gstin_of(df,"shippingGSTIN")
                    BillingGSTIN = get_gstin_of(df,"billingGSTIN")
                    ## PBAIP-23 - Sahil 5-June-2024 (Pick GSTIN from masterdata file if not in prediction) -Code starts
                    if VendorGSTIN == None or ShippingGSTIN == None or BillingGSTIN == None:
                        import os
                        import TAPPconfig as cfg
                        vendorMasterDataPath = cfg.getVendorMasterData()
                        buyerMasterDataPath = cfg.getBuyerMasterData()
                        masterFilePath = os.path.join(vendorMasterDataPath)
                        addressFilePath = os.path.join(buyerMasterDataPath)
                        VENDOR_MASTERDATA = pd.read_csv(masterFilePath, encoding='unicode_escape')
                        VENDOR_ADDRESS_MASTERDATA = VENDOR_MASTERDATA.applymap(lambda x: x.strip() if isinstance(x, str) else x)
                        BUYER_ADDRESS_MASTERDATA = pd.read_csv(addressFilePath, encoding='unicode_escape')
                        BUYER_ADDRESS_MASTERDATA = BUYER_ADDRESS_MASTERDATA.applymap(lambda x: x.strip() if isinstance(x, str) else x)
                        
                        # VENDOR_ADDRESS_MASTERDATA = pd.read_csv(r"E:\GitHub\pAIges_Client\pAIges_Client\Utilities\VENDOR_ADDRESS_MASTERDATA.csv")
                        VENDOR_ADDRESS_MASTERDATA['VENDOR_GSTIN'] = VENDOR_ADDRESS_MASTERDATA['VENDOR_GSTIN'].str.capitalize()
                        # BUYER_ADDRESS_MASTERDATA = pd.read_csv(r"E:\GitHub\pAIges_Client\pAIges_Client\Utilities\BUYER_ADDRESS_MASTERDATA.csv")
                        BUYER_ADDRESS_MASTERDATA['GSTIN'] = BUYER_ADDRESS_MASTERDATA['GSTIN'].str.capitalize()
                        for gstin in gst_list:
                            gstin = gstin.capitalize()
                            print("Testing GSTIN", gstin)
                            vendor_master = VENDOR_ADDRESS_MASTERDATA[VENDOR_ADDRESS_MASTERDATA["VENDOR_GSTIN"] == gstin]
                            print("Vendor Masterdata shape:", vendor_master.shape)
                            if vendor_master.shape[0]>0:
                                VendorGSTIN = gstin
                                print("Vendor GSTIN Assigned")
                            buyer_master = BUYER_ADDRESS_MASTERDATA[BUYER_ADDRESS_MASTERDATA["GSTIN"] == gstin]
                            print("Buyer Masterdata shape:", buyer_master.shape)
                            if buyer_master.shape[0]>0:
                                ShippingGSTIN = gstin
                                print("Billing GSTIN Assigned")
                    ## PBAIP-23 - Sahil 5-June-2024 (Pick GSTIN from masterdata file if not in prediction) -Code ends     
                    print("VendorGSTIN",VendorGSTIN,"ShippingGSTIN :",ShippingGSTIN,"BillingGSTIN :",BillingGSTIN)
                    if(VendorGSTIN is not None) & (ShippingGSTIN is not None):
                        if(VendorGSTIN[:2] == ShippingGSTIN[:2]):
                            cgst = 1
                        else:
                            cgst = 0  
                    if cgst == -1:
                        if(VendorGSTIN is not None) & (BillingGSTIN is not None):
                            if(VendorGSTIN[:2] == BillingGSTIN[:2]):
                                cgst = 1
                            else:
                                cgst = 0 
            return cgst
        except:
            print("exception",traceback.print_exc())
            return cgst
    else:
        cgst = -1
        try:
            vendorGSTIN = prediction.get("vendorGSTIN")
            billingGSTIN = prediction.get("billingGSTIN")
            shippingGSTIN = prediction.get("shippingGSTIN")
            #a = vendorGSTIN.get("text")
            #print(f"vendor GSTIN = {a} billingGSTIN = {billingGSTIN} shippingGSTIN = {shippingGSTIN}")
            #print("----------------")
            isVendorGSTIN = (vendorGSTIN and (vendorGSTIN.get("text")!= '' and vendorGSTIN.get("text") is not None) and (vendorGSTIN.get("text") != "N/A"))
            isBillingGSTIN = (billingGSTIN and (billingGSTIN.get("text")!= '' and billingGSTIN.get("text") is not None) and (billingGSTIN.get("text") != "N/A"))
            isShippingGSTIN = (shippingGSTIN and (shippingGSTIN.get("text")!= '' and shippingGSTIN.get("text") is not None) and (shippingGSTIN.get("text") != "N/A"))
            print(f"vendor GSTIN = {isVendorGSTIN} billingGSTIN = {isBillingGSTIN} shippingGSTIN = {isShippingGSTIN}")
            print("=====================")
            if (isVendorGSTIN and isBillingGSTIN):
                vendorGSTIN = vendorGSTIN.get("text")
                billingGSTIN = billingGSTIN.get("text")
                #Cgst or Igst
                if vendorGSTIN[:2] == billingGSTIN[:2]:
                    cgst = 1
                else:
                    cgst = 0  
            if cgst == -1:
                if (isVendorGSTIN and isShippingGSTIN):
                    vendorGSTIN = vendorGSTIN.get("text")
                    shippingGSTIN = shippingGSTIN.get("text")
                    #Cgst or Igst
                    if vendorGSTIN[:2] == shippingGSTIN[:2]:
                        cgst = 1
                    else:
                        cgst = 0     
            return cgst
        except:
            print("check_if_cgst exception :",traceback.print_exc())
            return cgst
               
def check_if_cgst_1(df):
    """
    Parameters
    ----------
    df : TYPE - Dataframe
        DESCRIPTION.

    Returns
    -------
    cgst : TYPE integer flag default -1, 1 true, 0 false
        DESCRIPTION.

    """
    cgst = -1
    try:    
        unqGSTins = list(df[df["is_gstin_format"] == 1]["text"])
        unqGSTins = list(set(unqGSTins))
        GSTIN_PATTERN = r"\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}"
        #Cgst or Igst
        if len(unqGSTins) > 1:
            firstGSTIN = unqGSTins[0]
            secondGSTIN = unqGSTins[1]
            first_span = re.search(GSTIN_PATTERN, firstGSTIN).span()
            firstGSTIN = firstGSTIN[first_span[0]:first_span[1]]
            second_span = re.search(GSTIN_PATTERN, secondGSTIN).span()
            secondGSTIN = secondGSTIN[second_span[0]:second_span[1]]
    
            # if vendorGstin[:2] == billingGstin[:2]:
            if firstGSTIN[:2] == secondGSTIN[:2]:
                cgst = 1
            else:
                cgst = 0
        return cgst
    except:
        print("exception",traceback.print_exc())
        
        return cgst

def check_if_cgst(prediction:dict):
    """
    Parameters
    ----------
    df : TYPE - dictionary
        DESCRIPTION.

    Returns
    -------
    cgst : TYPE integer flag default -1, 1 true, 0 false
        DESCRIPTION.

    """

    cgst = -1
    try:
        vendorGSTIN = prediction.get("vendorGSTIN")
        billingGSTIN = prediction.get("billingGSTIN")
        #isVendorGSTIN = (vendorGSTIN and (vendorGSTIN.get("text")!= '' or vendorGSTIN.get("text") is not None))
        isBillingGSTIN = (billingGSTIN and (billingGSTIN.get("text")!= '' or billingGSTIN.get("text") is not None))
        if (isVendorGSTIN and isBillingGSTIN):
            vendorGSTIN = vendorGSTIN.get("text")
            billingGSTIN = billingGSTIN.get("text")
            #Cgst or Igst
            if vendorGSTIN[:2] == billingGSTIN[:2]:
                cgst = 1
            else:
                cgst = 0  
        if cgst == -1:
            shippingGSTIN = prediction.get("shippingGSTIN")
            isVendorGSTIN = (vendorGSTIN and (vendorGSTIN.get("text")!= '' or vendorGSTIN.get("text") is not None))
            isShippingGSTIN = (shippingGSTIN and (shippingGSTIN.get("text")!= '' or shippingGSTIN.get("text") is not None))
            if (isVendorGSTIN and isShippingGSTIN):
                vendorGSTIN = vendorGSTIN.get("text")
                shippingGSTIN = shippingGSTIN.get("text")
                #Cgst or Igst
                if vendorGSTIN[:2] == shippingGSTIN[:2]:
                    cgst = 1
                else:
                    cgst = 0
                
        return cgst
    except:
        print("check_if_cgst exception :",traceback.print_exc())
        return cgst

    
# def calcAmountFields(df,prediction:dict):
#     try:
#         #Get vendor and buyer gstin
#         # vendorGstins = list(df[df["predict_label"] == 'vendorGstin']["text"])
#         # vendorGstins = list(set(vendorGstins))
#         # vendorGstin = vendorGstins[0]
#         # billingGstins = list(df[df["predict_label"] == 'billingGstin']["text"])
#         # billingGstins = list(set(billingGstins))
#         # billingGstin = billingGstins[0]

#         # Get tax structure from prediction file
#         cgst = check_if_cgst(prediction)

#         # # Get tax structure from dataframe file
#         # cgst = check_if_cgst_1(df)
        
#         #Get all extracted amounts
#         max_line_row = max(list(df["line_row"]))

#         df_amt = df[(df["extracted_amount"] > 0) & (df["extracted_amount"] < 1000000) & 
#                     ((df["line_row"] == 0) | (df["line_row"] == max_line_row))]
#         ocr_amts = list(df_amt["extracted_amount"])
#         unq_amts = list(set(ocr_amts))
#         unq_amts = sorted(unq_amts,
#                           reverse = True)

#         #loop through top 5 highest amounts only
#         max_range = min(3,len(unq_amts))
#         for i in range(0,max_range):
#             max_amt = unq_amts[i]
#             amts = unq_amts[i+1:]
#             if len(amts) > 0:
#                 result = taxOnlyAmts(max_amt,
#                                      cgst,
#                                      amts)
#                 # print("max",max_amt,result)
#                 if result == {}:
#                     continue
#                 else:
#                     # print(result)
#                     return result
#             else:
#                 break
#         # result = {"total":unq_amts[0],
#         #           "subTotal":unq_amts[0],
#         #           "cgst":0.0,
#         #           "sgst":0.0,
#         #           "igst":0.0,
#         #           "totalGst":0.0}
#         # print(result)
#         # return result
#         return None
#     except:
#         print(traceback.print_exc(),
#               "Calc Amount Fields")
#         return None

def calcAmountFields(df,vendor_masterdata,prediction=None):
    try:
        # #Get tax structure from prediction file
        # cgst = check_if_cgst(prediction)

        # Get tax structure from dataframe file
        cgst = check_if_cgst_v1(df)
        print("cgst_present",cgst)
        if cgst == -1:
            print("Can't identify tax structure")
            return None
        try:
            ## PBAIP-23 - Sahil 5-June-2024 (Added logic for removal of HSN Code) -Code starts
            import numpy as np
            from tax_slab_analysis import find_gst_table
            table_start, table_end = find_gst_table(df)
            # print("sahil1 testing", table_start, table_end)
            df["potential_amount"] = np.where(((df.tableLineNo_New >= table_start) & (df.tableLineNo_New <= table_end)), 0, df.potential_amount)
        except Exception as e:
            print("Exception in chenging potential_amount feature",e)
        #Get all extracted amounts
        max_line_row = max(list(df["line_row_new"]))
        threshold_amount = 999999999
        # Commented the limit total amount extraction limit 25 Nov
        df_amt = df[((df["extracted_amount"] > 0) | (((df["potential_amount"] > 0)) & (df["val"] > 0) & (df["val"] < threshold_amount))) 
                    # & 
                    # ((df["line_row_new"] == 0) | (df["line_row_new"] == max_line_row))
                    ]
        ## PBAIP-23 - Sahil 5-June-2024 (Added logic for removal of HSN Code) -Code ends
        # Commented the limit  total amount extraction limit 25 Nov

        # amts_token_id = list(zip(df_amt["token_id"],df_amt["extracted_amount"]))
        # ocr_amts = list(df_amt["extracted_amount"])
        
        amts_token_id = [(token_id, extracted_amount if extracted_amount != 0 else val) for token_id, extracted_amount, val in zip(df_amt["token_id"], df_amt["extracted_amount"], df_amt["val"])]
        ocr_amts = [extracted_amount if extracted_amount != 0 else val for extracted_amount, val in zip(df_amt["extracted_amount"], df_amt["val"])]
        
        print("ocr_amts",ocr_amts)
        unq_amts = sorted(list(set(ocr_amts)),reverse = True)
        amts_token_id = sorted(amts_token_id,reverse = True,key=itemgetter(1))
        # print("ocr_amts_token_id",amts_token_id)

        #loop through top 5 highest amounts only
        max_range = min(5,len(unq_amts))
        for i in range(0,max_range):
            max_amt = unq_amts[i]
            amts = unq_amts[i+1:]
            if len(amts) > 0:
                print("checking for ",max_amt,amts)
                result = taxOnlyAmts(max_amt,
                                     cgst,
                                     amts)
                print("result",result)
                if result == {}:
                    continue
                else:
                    result_with_token = {}
                    print("taxOnlyAmts result",result)
                    total_amt_val_tokens = [tup for tup in amts_token_id if abs(tup[1]-result.get("total"))<0.9]
                    total_amt_val_tokens = sorted(total_amt_val_tokens,reverse=True,key=itemgetter(0))
                    print("total_amt_val_tokens :",total_amt_val_tokens)
                    ## Issue# 170 - Sahil -26 April-2024 (Improvement in Total Amount Accuracy) -Code starts
                    # if (vendor_masterdata != None):
                    #     if vendor_masterdata.get('VENDOR_GSTIN') in vendorGSTIN_list:
                    #         if check_totalAmount(df,total_amt_val_tokens):
                    #             result_with_token["totalAmount"] = total_amt_val_tokens[0]
                    #             print("Total Amount has Keyword")
                    #         else:
                    #             print("Total Amount has No Keyword")
                    #             # return None
                    #             continue
                    #     else:
                    #         result_with_token["totalAmount"] = total_amt_val_tokens[0]
                    # else:
                    ## Issue# 170 - Sahil -26 April-2024 (Improvement in Total Amount Accuracy) -Code ends
                    result_with_token["totalAmount"] = total_amt_val_tokens[0]
                    subtotal_amt_val_tokens = [tup for tup in amts_token_id if abs(tup[1]-result.get("subTotal"))<0.9]
                    subtotal_amt_val_tokens = sorted(subtotal_amt_val_tokens,reverse=True,key=itemgetter(0))
                    print("subtotal_amt_val_tokens :",subtotal_amt_val_tokens)
                    result_with_token["subTotal"] = subtotal_amt_val_tokens[0]                    
                    cgst_amt_val_tokens = [tup for tup in amts_token_id if abs(tup[1]- result.get("cgst")) < 0.9]
                    print("cgst_amt_val_tokens :",cgst_amt_val_tokens)
                    if (result.get("cgst")> 0) & (len(cgst_amt_val_tokens)>=2):
                        ## Issue# 170 - Sahil -26 April-2024 (Improvement in Total Amount Accuracy) -Code starts
                        # if (vendor_masterdata != None):
                        #     if vendor_masterdata.get('VENDOR_GSTIN') in vendorGSTIN_list:
                        #         if check_cgst(df,cgst_amt_val_tokens):
                        #             print("GST has Keyword")
                        #             result_with_token["CGSTAmount"] = cgst_amt_val_tokens[0]                   
                        #             result_with_token["SGSTAmount"] = cgst_amt_val_tokens[1]
                        #         else:
                        #             print('Gst amount has No Keyword')
                        #             # return None
                        #             continue
                        #     else:
                        #         result_with_token["CGSTAmount"] = cgst_amt_val_tokens[0]                   
                        #         result_with_token["SGSTAmount"] = cgst_amt_val_tokens[1]
                        # else:
                        ## Issue# 170 - Sahil -26 April-2024 (Improvement in Total Amount Accuracy) -Code ends
                        result_with_token["CGSTAmount"] = cgst_amt_val_tokens[0]                   
                        result_with_token["SGSTAmount"] = cgst_amt_val_tokens[1]
                    else :
                        ## Issue# 170 - Sahil -26 April-2024 (Improvement in Total Amount Accuracy) -Code starts
                        # if (vendor_masterdata != None):
                        #     if vendor_masterdata.get('VENDOR_GSTIN') in vendorGSTIN_list:
                        # return None
                        pass
                        ## Issue# 170 - Sahil -26 April-2024 (Improvement in Total Amount Accuracy) -Code ends
                    igst_amt_val_tokens = [tup for tup in amts_token_id if abs(tup[1]-result.get("igst"))<0.9]
                    print("igst_amt_val_tokens :",igst_amt_val_tokens)
                    if result.get("igst")> 0:
                        ## Issue# 170 - Sahil -26 April-2024 (Improvement in Total Amount Accuracy) -Code starts
                        # if (vendor_masterdata != None):
                        #     if vendor_masterdata.get('VENDOR_GSTIN') in vendorGSTIN_list:
                        #         print("check_igst(df,igst_amt_val_tokens) :",check_igst(df,igst_amt_val_tokens))
                        #         if check_igst(df,igst_amt_val_tokens):
                        #             result_with_token["IGSTAmount"] = igst_amt_val_tokens[0]
                        #             #return result_with_token
                        #         else:
                        #             # return None
                        #             continue
                        #     else:
                        #         result_with_token["IGSTAmount"] = igst_amt_val_tokens[0]
                        # else:
                        ## Issue# 170 - Sahil -26 April-2024 (Improvement in Total Amount Accuracy) -Code ends
                        result_with_token["IGSTAmount"] = igst_amt_val_tokens[0]                 
                    print("result_with_token:",result_with_token)
                    return result_with_token
            else:
                break
        return None
    except:
        print(traceback.print_exc(),
              "Calc Amount Fields")
        return None
## Issue# 170 - Sahil -26 April-2024 (Improvement in Total Amount Accuracy) -Code starts
def calcAmountFields_due_amount(df,vendor_masterdata,prediction=None):
    try:
        cgst = check_if_cgst_v1(df)
        print("cgst_present",cgst)
        if cgst == -1:
            print("Can't identify tax structure")
            return None
        #Get all extracted amounts
        max_line_row = max(list(df["line_row_new"]))
        # Commented the limit total amount extraction limit 25 Nov
        df_amt = df[((df["extracted_amount"] > 0) | (((df["potential_amount"] > 0)) & (df["val"] > 0))) 
                    # & 
                    # ((df["line_row_new"] == 0) | (df["line_row_new"] == max_line_row))
                    ]
        # df_amt = DF[((DF["extracted_amount"] > 0) | (((DF["potential_amount"] > 0)) & (DF["val"] > 0)))]
        df_amt['new_amount'] = df_amt.apply(lambda row: row['extracted_amount'] if row['potential_amount'] == 0 else row['val'], axis=1)
        df_amt = df_amt.sort_values(by='new_amount', ascending=False)
        df_amt = df_amt.reset_index(drop=True)  # Reset index without adding a new column for the old index

        # df_amt.to_csv("test1.csv")
        
        # Remove rows with values within a tolerance of 1 Rs.
        tolerance = 1
        tolerance_amount = 100000000000
        rows_to_drop = []
        for index, row in df_amt.iterrows():
            if index >= 0:
                if abs(row["new_amount"] > tolerance_amount ):
                    rows_to_drop.append(index)
                elif abs(row['new_amount'] - df_amt.loc[index - 1, 'new_amount']) <= tolerance:
                    rows_to_drop.append(index)
                

        df_amt = df_amt.drop(rows_to_drop)

        # Sort the DataFrame again based on 'new_amount' before saving to CSV
        df_amt = df_amt.sort_values(by='new_amount', ascending=False)
        # df_amt.to_csv("test2.csv")
        amts_token_id = [(token_id, extracted_amount if extracted_amount != 0 else val) for token_id, extracted_amount, val in zip(df_amt["token_id"], df_amt["extracted_amount"], df_amt["val"])]
        ocr_amts = [extracted_amount if extracted_amount != 0 else val for extracted_amount, val in zip(df_amt["extracted_amount"], df_amt["val"])]
        
        print("ocr_amts",ocr_amts)
        unq_amts = sorted(list(set(ocr_amts)),reverse = True)
        amts_token_id = sorted(amts_token_id,reverse = True,key=itemgetter(1))

        max_range = min(7,len(unq_amts))
        for i in range(0,max_range):
            max_amt = unq_amts[i]
            amts = unq_amts[i+1:]
            if len(amts) > 0:
                print("checking for ",max_amt,amts)
                result = taxOnlyAmts(max_amt,
                                     cgst,
                                     amts)
                print("result",result)
                if result == {}:
                    continue
                else:
                    result_with_token = {}
                    print("taxOnlyAmts result",result)
                    total_amt_val_tokens = [tup for tup in amts_token_id if abs(tup[1]-result.get("total"))<0.9]
                    total_amt_val_tokens = sorted(total_amt_val_tokens,reverse=True,key=itemgetter(0))
                    print("total_amt_val_tokens :",total_amt_val_tokens)
                    if (vendor_masterdata != None):
                        if str(vendor_masterdata.get('VENDOR_GSTIN'))[2:12] in vendor_PAN_other_max_amts:
                            keyword_to_search = vendor_PAN_other_max_amts[str(vendor_masterdata.get('VENDOR_GSTIN'))[2:12]]
                            if checkAmountWithKeywordForOtherMaxAmount(df,total_amt_val_tokens, keyword_to_search):
                                print("Due Amount is present")
                                continue
                            else:
                                result_with_token["totalAmount"] = total_amt_val_tokens[0]
                                # return None   
                        else:
                            result_with_token["totalAmount"] = total_amt_val_tokens[0]
                    else:
                        print("No GSTIN found in masterdata")
                        if checkAmountWithKeywordForOtherMaxAmount(df,total_amt_val_tokens):
                            print("Due Amount is present")
                            continue
                        else:
                            result_with_token["totalAmount"] = total_amt_val_tokens[0]
                    subtotal_amt_val_tokens = [tup for tup in amts_token_id if abs(tup[1]-result.get("subTotal"))<0.9]
                    subtotal_amt_val_tokens = sorted(subtotal_amt_val_tokens,reverse=True,key=itemgetter(0))
                    print("subtotal_amt_val_tokens :",subtotal_amt_val_tokens)
                    result_with_token["subTotal"] = subtotal_amt_val_tokens[0]                    
                    cgst_amt_val_tokens = [tup for tup in amts_token_id if abs(tup[1]- result.get("cgst")) < 0.9]
                    print("cgst_amt_val_tokens :",cgst_amt_val_tokens)
                    if (result.get("cgst")> 0) & (len(cgst_amt_val_tokens)>=2):
                        if (vendor_masterdata != None):
                            if vendor_masterdata.get('VENDOR_GSTIN') in vendorGSTIN_list:
                                if check_cgst(df,cgst_amt_val_tokens):
                                    print("GST has Keyword")
                                    result_with_token["CGSTAmount"] = cgst_amt_val_tokens[0]                   
                                    result_with_token["SGSTAmount"] = cgst_amt_val_tokens[1]
                                else:
                                    print('Gst amount has No Keyword')
                                    # return None
                                    continue
                            else:
                                result_with_token["CGSTAmount"] = cgst_amt_val_tokens[0]                   
                                result_with_token["SGSTAmount"] = cgst_amt_val_tokens[1]
                        else:
                            result_with_token["CGSTAmount"] = cgst_amt_val_tokens[0]                   
                            result_with_token["SGSTAmount"] = cgst_amt_val_tokens[1]
                    else :
                        if (vendor_masterdata != None):
                            if vendor_masterdata.get('VENDOR_GSTIN') in vendorGSTIN_list:
                                return None
                    igst_amt_val_tokens = [tup for tup in amts_token_id if abs(tup[1]-result.get("igst"))<0.9]
                    print("igst_amt_val_tokens :",igst_amt_val_tokens)
                    if result.get("igst")> 0:
                        if (vendor_masterdata != None):
                            if vendor_masterdata.get('VENDOR_GSTIN') in vendorGSTIN_list:
                                print("check_igst(df,igst_amt_val_tokens) :",check_igst(df,igst_amt_val_tokens))
                                if check_igst(df,igst_amt_val_tokens):
                                    result_with_token["IGSTAmount"] = igst_amt_val_tokens[0]
                                    #return result_with_token
                                else:
                                    # return None
                                    continue
                            else:
                                result_with_token["IGSTAmount"] = igst_amt_val_tokens[0]
                        else:
                            result_with_token["IGSTAmount"] = igst_amt_val_tokens[0]                 
                    print("result_with_token:",result_with_token)
                    return result_with_token
            else:
                break
        return None
    except:
        print("calcAmountFields_due_amount exception", traceback.print_exc())
        return None

def enchant_distance(text):
    try :
        import enchant
        tokens = str(str(text).lower()).translate(str.maketrans('', '', string.punctuation)).split()
        # print("tokens11 :",tokens,type(tokens))
        for w in tokens:
            distance = enchant.utils.levenshtein("total", w)
            # print("dist :",distance)
            if (distance == 0 or distance ==1):
                return distance
        return distance
    except:
        print("enchant_distance exception",traceback.print_exc())
        return -1

def check_totalAmount(df,total_amt_val_tokens):
    print("total_amt_val_tokens :",total_amt_val_tokens[0][0])
    # Updated logic to loop over all the tiken and check Dec 9
    # df_token=df[df['token_id']==total_amt_val_tokens[0][0]]
    try:
        for i  in total_amt_val_tokens:
            print(" i tuples :",i)
            df_token=df[df['token_id']==i[0]]
            print("df_token :",df_token.shape)
            for row in df_token.itertuples():
                # to fix ocr issue with label extraction amd matching added enchant dist cal.
                # if ("total" in row.left_processed_ngbr.lower()):
                distance = enchant_distance(row.left_processed_ngbr)
                if (("total" in row.left_processed_ngbr.lower()) | (distance in [0,1])):
                    return True
                # to fix ocr issue with label extraction amd matching added enchant dist cal.
                # Fixed the bug iterrating over all the token Dec 9
                # else:
                #     return False
                # Fixed the bug iterrating over all the token Dec 9
        return False
    # Updated logic to loop over all the tiken and check Dec 9
    except :
        print("check_totalAmount Exception",traceback.print_exc())
        return False
## Issue# 170 - Sahil -26 April-2024 (Improvement in Total Amount Accuracy) -Code starts
def checkAmountWithKeywordForOtherMaxAmount(df,total_amt_val_tokens, keyword_to_search = None):
    print("total_amt_val_tokens :",total_amt_val_tokens[0][0])
    try:
        for i  in total_amt_val_tokens:
            df_token=df[df['token_id']==i[0]]
            print("df_token :",df_token.shape)
            for row in df_token.itertuples():
                if keyword_to_search != None:
                    if (keyword_to_search in row.left_processed_ngbr.lower()):
                        return True
                else:
                    if ("due" in row.left_processed_ngbr.lower()):
                        return True
        return False
    except :
        print("checkAmountWithKeywordForOtherMaxAmount Exception",traceback.print_exc())
        return False
## Issue# 170 - Sahil -26 April-2024 (Improvement in Total Amount Accuracy) -Code ends
   
def check_igst(df,igst_amt_val_tokens):
    try:
        # Updated logic to loop over all the tiken and check Dec 9
        # df_token=df[df['token_id']==igst_amt_val_tokens[0][0]]
        for tuples in igst_amt_val_tokens:
            df_token=df[df['token_id']== tuples[0]]
            print("df_token shape :",df_token.shape)
            for row in df_token.itertuples():
                left_ngb_words = str(row.left_processed_ngbr.lower()).translate(str.maketrans('', '', string.punctuation))
                print("left_ngb_words withiut punctuations :",left_ngb_words)
                above_ngb_words = str(row.above_processed_ngbr.lower()).translate(str.maketrans('', '', string.punctuation))
                print("above_ngb_words withiut punctuations :",above_ngb_words)
                A = any("igst" == word.lower() in word for word in left_ngb_words.split()) or any("igst" == word.lower() in word for word in above_ngb_words.split())
                B = any("integrated" == word.lower() in word for word in left_ngb_words.split()) or any("integrated" == word.lower() in word for word in above_ngb_words.split())

                # A= ("igst" in row.left_processed_ngbr.lower()) or ("igst" in row.above_processed_ngbr.lower())
                # # B= ("integrated" in row.left_processed_ngbr.lower()) or ("integrated" in row.above_processed_ngbr.lower())
                if A or B:
                    return True
                # else:
                #     return False
        return False
        # Updated logic to loop over all the tiken and check Dec 9
    except :
        print("check_igst Exception :",traceback.print_exc())
        return False
        
def check_cgst(df,cgst_amt_val_tokens):
    try:
        # print("cgst_amt_val_tokens 1:",cgst_amt_val_tokens)
        # Updated logic to loop over all the tiken and check Dec 9
        #df_token=df[df['token_id']==cgst_amt_val_tokens[0][0]]
        for tuples in cgst_amt_val_tokens:
            df_token=df[df['token_id']==tuples[0]]            
            print("df_token shape :",df_token.shape)
            for row in df_token.itertuples():
                # A= ("gst" in row.left_processed_ngbr.lower()) or ("gst" in row.above_processed_ngbr.lower())
                # B = ("central" in row.above_processed_ngbr.lower()) or ("state" in row.above_processed_ngbr.lower())
                left_ngb_words = str(row.left_processed_ngbr.lower()).translate(str.maketrans('', '', string.punctuation))
                print("left_ngb_words withiut punctuations :",left_ngb_words)
                above_ngb_words = str(row.above_processed_ngbr.lower()).translate(str.maketrans('', '', string.punctuation))
                print("above_ngb_words withiut punctuations :",above_ngb_words)
                A = any("cgst" == word.lower() in word for word in left_ngb_words.split()) or any("sgst" == word.lower() in word for word in above_ngb_words.split())
                B = any("central" == word.lower() in word for word in above_ngb_words.split()) or any("state" == word.lower() in word for word in above_ngb_words.split())
                print("condition A ,B",A,B)    
                if A or B:
                    print("row.left_processed_ngbr.lower() :",row.left_processed_ngbr.lower())
                    print("row.above_processed_ngbr.lower() :",row.above_processed_ngbr.lower())
                    return True
                # else:
                #     return False
        return False
    # Updated logic to loop over all the tiken and check Dec 9
    except :
        print("check_cgst Exception",traceback.print_exc())
        return False

def assignVavluesToDf(col_name,col_vals,df,
                      base_col = "token_id"):
    import numpy as np
    new_col = col_name + "_new"
    df[new_col] = df[base_col].map(col_vals)
    df[col_name] = np.where(df[new_col].isnull(),
                            df[col_name],
                            df[new_col])
    return df

def reduce_confidence(df,amount_fields):
    try:
        for field in amount_fields.keys():
            tokens = df[df["prob_"+field]>0.9]['token_id']
            tokens = tokens.to_list()
            if len(tokens)>0:
                print("tokens to reduce confidence :",tokens)
                for tk in tokens:
                    # print("befor Reducing confidence :",df[df["token_id"]==tk]["prob_"+field])
                    # print("befor changing label :",df[df["token_id"]==tk]["predict_label"])
                    df = assignVavluesToDf("predict_label",{tk:"unknown"},df)
                    df = assignVavluesToDf(("prob_" + field),{tk:0.8},df)
                    # print("After Reducing confidence :",df[df["token_id"]==tk]["prob_"+field])
                    # print("After changing label :",df[df["token_id"]==tk]["predict_label"])
        return df
    except:
        print("reduce_confidence exception",traceback.print_exc())
        return df
    
def updated_amount_field_prob_label(df,vendor_masterdata):
    copy_df = df.copy(deep = True)
    try:
        ## Issue# 170 - Sahil -26 April-2024 (Improvement in Total Amount Accuracy) -Code starts
        if (vendor_masterdata != None):
            if str(vendor_masterdata.get('VENDOR_GSTIN'))[2:12] in vendor_PAN_other_max_amts:
                print("checking for due amounts")
                amount_fields = calcAmountFields_due_amount(df,vendor_masterdata)
            else:
                amount_fields = calcAmountFields(df,vendor_masterdata)
        else:
            amount_fields = calcAmountFields(df,vendor_masterdata)
        ## Issue# 170 - Sahil -26 April-2024 (Improvement in Total Amount Accuracy) -Code ends    
        print("calcAmountFields result :",amount_fields)
        if amount_fields:
            df = reduce_confidence(df,amount_fields)
            for key, val in amount_fields.items():
                # print("tkn lbl",{val[0]:key},{val[0]:1.0})
                df = assignVavluesToDf("predict_label",{val[0]:key},df)
                df = assignVavluesToDf(("prob_" + key),{val[0]:1.0},df)
        return df
    except:
        print("updated_amount_field_prob_label exception",traceback.print_exc())
        return copy_df


if __name__ == "__main__":
    import pandas as pd
    path = r"C:\Users\OVE\Downloads\2a030359-504b-11ed-99bd-80ce62234e48_pred.csv"
    df=pd.read_csv(path)
    df = updated_amount_field_prob_label(df)
    
    
    
    
    