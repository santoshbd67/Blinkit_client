
import traceback
import pandas as pd
import re
import operator
from helper_post_processor import wordBoundingText
from difflib import SequenceMatcher
import math
from collections import Counter
from collections import OrderedDict
# import json
import numpy
import os
# import pickle
import collections
from dateutil.parser import parse
from price_parser import parse_price
import preProcUtilities as putil
import TAPPconfig as cfg
import copy
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors
from modify_prediction import modify_prediction, validate_invnum, predictPoNumber_from_metadata
from rule_based_validation import validate_final_output
from business_rules import apply_business_rules
from business_rules import client_required_fields
import business_rules as biz
import client_rules as cl_rules
from  client_rules import makeDefaultPredValToNA, vendor_name_validation,sanitize_invoice_date_text
from client_rules import apply_client_rules, adding_mandatory_fieldFlag, custom_sorting_prediction
from calculateAmountFields import taxOnlyAmts
# from client_rules import build_final_QRCode_json

# for insurance extraction only
from client_rules import present_doc_output, validating_amount_fields_increasing_confidence

# for scaling the bounding boxes in UI
from business_rules import scale_bounding_boxes
# from format_identifier import get_vendor Return another fuction get vendor by extract matching GSTIN
from path_finder import extract_vendor_specific_extra_fields
from tax_slab_analysis import calculateandassignslab, amountslablist
from datetime import datetime
#import preProcUtilities as putil

# Define Constants
#with open('constant_labels.pkl', 'rb') as hand:
#    CONST= pickle.load(hand)
CONST = putil.getPostProcessConstantLabels()
# fields
#with open('field_labels.pkl', 'rb') as handle:
#    FIELD= pickle.load(handle)
FIELD = putil.getPostProcessFieldLabels()

SCORE = putil.getPostProcessScoring()
#with open('scoring.pkl', 'rb') as han:
#    SCORE= pickle.load(han)

#with open('check.pkl', 'rb') as ha:
#    CHECK= pickle.load(ha)
CHECK = putil.getPostProcessDtTypeCheck()

STP_CONFIGURATION = putil.getSTPConfiguration()
PREDICTION_THRESHOLD = {"CGSTAmount":0.16, "SGSTAmount":0.16, "IGSTAmount":0.16,
                        "poNumber" :0.23, "invoiceNumber" : 0.24, "invoiceDate" : 0.24,
                        "totalAmount" : 0.24, "TCSAmount" : 0.24, "freightAmount" :0.24,
                        "subTotal" : 0.24, "dueDate" : 0.24,"CessAmount":0.24,"additionalCessAmount":0.24}
FIELD = {'date_fields': ['invoiceDate', 'dueDate'],
'header_fields': ['invoiceNumber', 'poNumber', 'paymentTerms'],
'vendor_specific_addrefinal_candidates_fields': ['vendorAddress'],
'address_fields': ['shippingAddress', 'billingAddress'],
# 'address_fields': [],
'vendor_names_fields': ['vendorName','billingName','shippingName' ],
'vendor_specific_header_fields': ['vendorGSTIN', 'vendorEmail', 'currency','billingGSTIN','shippingGSTIN'],
'amount_fields': ['totalAmount', 'taxAmount', 'subTotal','SGSTAmount', 'CGSTAmount', 'IGSTAmount',
'freightAmount','discountAmount','TCSAmount','CessAmount','insuranceAmount','additionalCessAmount'],
'tax_rate_fields': ['taxRate'],
'lineitem_header_labels': ['hdr_itemCode', 'hdr_itemDescription', 'hdr_itemQuantity', 'hdr_unitPrice', 'hdr_itemValue',
'hdr_taxAmount', 'hdr_taxRate','hdr_CGSTAmount','hdr_SGSTAmount','hdr_IGSTAmount', 'hdr_UOM', 'hdr_HSNCode','hdr_CessAmount'],
'lineitem_value_labels': ['LI_itemQuantity', 'LI_unitPrice','LI_UOM', 'LI_itemValue', 'LI_HSNCode', 'LI_taxAmount', 'LI_taxRate', 'LI_itemCode',
'LI_itemDescription','LI_CGSTAmount', 'LI_SGSTAmount', 'LI_IGSTAmount','LI_CessAmount'],
'total_fields': ['totalAmount', 'subTotal'], 'lbl_total_fields': ['lblTotalAmount', 'lblSubTotal']}
vendorGSTIN_list = ['27AAFCD3317F1ZY','37AAFCD3317F1ZX','24AAFCD3317F1Z4','23AAFCD3317F1Z6','19AAFCD3317F1ZV',
'04AAFCD3317F1Z6','32AAFCD3317F1Z7','33AAFCD3317F1Z5','29AAFCD3317F1ZU','36AAFCD3317F1ZZ','06AAFCD3317F1Z2']
CHECK = {'date': ['invoiceDate','dueDate'],
'amount': ['totalAmount','SGSTAmount','CGSTAmount','IGSTAmount','taxAmount',
           'subTotal','unitPrice','itemValue','freightAmount','discountAmount','CessAmount','additionalCessAmount'],
'number': []}
# {"LOCATION": (left, right, top, bottom)}
# LOCATION_COORDINATES = {"TOP LEFT": (0.0, 0.5, 0.0, 0.5),
# "TOP RIGHT": (0.5, 1.0, 0.0, 0.5),
# "BOTTOM LEFT": (0.0, 0.5, 0.5, 1.0),
# "BOTTOM RIGHT": (0.5, 1.0, 0.5, 1.0),
# "NONE": (0.0, 1.0, 0.0, 1.0)}

# Read Vendor MasterData and form a Dictionary

script_dir = os.path.dirname(__file__)
print(script_dir)
vendorMasterDataPath = cfg.getVendorMasterData()
buyerMasterDataPath = cfg.getBuyerMasterData()
# Changed for reading path from file share 16/12/22
# addressFilePath = os.path.join(script_dir,buyerMasterDataPath)
addressFilePath = os.path.join(buyerMasterDataPath)
REFERENCE_MASTER_DATA = cfg.getReferenceMasterData()
# ADDRESS_MASTERDATA = pd.read_csv(addressFilePath, encoding='unicode_escape')

#masterFilePath = os.path.join(script_dir,vendorMasterDataPath)
masterFilePath = os.path.join(vendorMasterDataPath)
                             # r"Utilities/VENDOR_ADDRESS_MASTERDATA.csv")
# VENDOR_MASTERDATA = pd.read_csv(masterFilePath, encoding='unicode_escape')
#REFERENCE_MASTER_DATA_PATH = os.path.join(script_dir,REFERENCE_MASTER_DATA)
REFERENCE_MASTER_DATA_PATH = os.path.join(REFERENCE_MASTER_DATA)
# Changed for reading path from file share 16/12/22
REFERENCE_DATA = pd.read_csv(REFERENCE_MASTER_DATA_PATH, encoding='unicode_escape')

"""
def refresh_vendor_masterdata(preprocess_vendor):
    
    VENDOR_MASTERDATA = pd.read_csv(masterFilePath, encoding='unicode_escape')
    
    preprocess_vendor = str(preprocess_vendor).strip().upper()
    VENDOR_MASTERDATA["PREPROCESS_VENDOR"] = VENDOR_MASTERDATA["PREPROCESS_VENDOR"].str.strip()
    VENDOR_MASTERDATA["PREPROCESS_VENDOR"] = VENDOR_MASTERDATA["PREPROCESS_VENDOR"].str.upper()
    print("preprocess_vendor:", preprocess_vendor)
    print(VENDOR_MASTERDATA)
    list_of_vendors = VENDOR_MASTERDATA['PREPROCESS_VENDOR'].to_list()
    print("List of vendors: {}".format(list_of_vendors))
    print(preprocess_vendor in set(list_of_vendors))
    if preprocess_vendor != "UNKNOWN" and preprocess_vendor in set(list_of_vendors):
        VENDOR_MASTERDATA = VENDOR_MASTERDATA.loc[VENDOR_MASTERDATA["PREPROCESS_VENDOR"] == preprocess_vendor]
    else:
        VENDOR_MASTERDATA = VENDOR_MASTERDATA.loc[VENDOR_MASTERDATA["VENDOR_SPECIFIC"] == 0]
    print("VENDOR_MASTERDATA after filtering", VENDOR_MASTERDATA)
    VENDOR_MASTERDATA['hdr_CGSTAmount'] = ""
    VENDOR_MASTERDATA['hdr_SGSTAmount'] = ""
    VENDOR_MASTERDATA['hdr_IGSTAmount'] = ""
    VENDOR_MASTERDATA['hdr_HSNCode'] = ""

    vendor_master_selected = VENDOR_MASTERDATA[['VENDOR_ID', 'Client',
                                                'vendorName', 'vendorAddress',
                                                'vendorEmail', 'vendorGSTIN']]
    dict_vendor = {}
    for idx, row in vendor_master_selected.iterrows():
        final_list = []
        f = row['VENDOR_ID']

        vendor_name = re.split(r'[^\w]', str(row['vendorName']))
        vendor_address = re.split(r'[^\w]', str(row['vendorAddress']))
        vendor_email = re.split(r'[^\w]', str(row['vendorEmail']))
        vendor_gstin = re.split(r'[^\w]', str(row['vendorGSTIN']))
    #    vendor_pan = re.split(r'[^\w]', str(row['vendorPAN']))

        final_list.extend(vendor_name)
        final_list.extend(vendor_address)
        final_list.extend(vendor_email)
        final_list.extend(vendor_gstin)
    #    final_list.extend(vendor_pan)
        final_list = [x.upper() for x in final_list if ((x != '') and (x != 'nan'))]
        final_list = [x for x in final_list if (len(x) > 1)]
        dict_vendor[f] = final_list
    return dict_vendor, VENDOR_MASTERDATA
dict_vendor = refresh_vendor_masterdata("UNKNOWN")[0]
"""
# dict_vendor = {}
# for idx, row in vendor_master_selected.iterrows():
#     final_list = []
#     f = row['VENDOR_ID']

#     vendor_name = re.split(r'[^\w]', str(row['vendorName']))
#     vendor_address = re.split(r'[^\w]', str(row['vendorAddress']))
#     vendor_email = re.split(r'[^\w]', str(row['vendorEmail']))
#     vendor_gstin = re.split(r'[^\w]', str(row['vendorGSTIN']))
#    vendor_pan = re.split(r'[^\w]', str(row['vendorPAN']))

#     final_list.extend(vendor_name)
#     final_list.extend(vendor_address)
#     final_list.extend(vendor_email)
#     final_list.extend(vendor_gstin)
#    final_list.extend(vendor_pan)

#     final_list = [x.upper() for x in final_list if ((x != '') and (x != 'nan'))]
#     final_list = [x for x in final_list if (len(x) > 1)]

#     dict_vendor[f] = final_list

print("************ Vendor MasterData Read ***************")

script_dir = os.path.dirname(__file__)
customFieldPath = os.path.join(script_dir,
                              "Utilities/VENDOR_CUSTOM_FIELD.csv")

# VENDOR_SPECIFIC_FIELD = pd.read_csv(customFieldPath, encoding='unicode_escape')


tableFieldPath = os.path.join(script_dir,
                              "Utilities/TABLE_CUSTOM_FIELD.csv")

TABLE_CUSTOM_FIELD = pd.read_csv(tableFieldPath, encoding='unicode_escape')

# Code ends






    #    vendor_pan = re.split(r'[^\w]', str(row['vendorPAN']))

    #    final_list.extend(vendor_pan)




# Helper Methods for LineItem Extraction
def sort_tuple_by_first_value(tup):
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    tup.sort(key=lambda x: x[0])
    return tup


def convert_float(x):
    try:
        return float(x)
    except:
        return ''


def round_decimals_up(number: float, decimals: int = 2):
    """
    Returns a value rounded up to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.ceil(number)

    factor = 10 ** decimals
    return math.ceil(number * factor) / factor


def get_top_label(g):
    c = g['predict_label'].value_counts()
    res = c.iloc[np.lexsort((c.index, -c.values))]
    return res.idxmax()

def logistic_function(x, steepness=6, midpoint=0.15):
    return 1 / (1 + math.exp(-steepness * (x - midpoint)))


def logistic_function_(x, steepness=4, midpoint=0.2):
    return 1 / (1 + math.exp(-steepness * (x - midpoint)))


def get_addresses_from_masterdata(DF, prediction):
    """
    """
    # TEMP = pd.DataFrame()
    print("Inside get_addresses_from_masterdata")
    global ADDRESS_MASTERDATA
    ADDRESS_MASTERDATA = pd.read_csv(addressFilePath, encoding='unicode_escape')

    dict_address = {}
    for idx, row in ADDRESS_MASTERDATA.iterrows():
        final_list = []
        f = row['ID']

        address = re.split(r'[^\w]', str(row['ADDRESS']))
        gstin = re.split(r'[^\w]', str(row['GSTIN']))
        name = re.split(r'[^\w]', str(row['NAME']))

        final_list.extend(address)
        final_list.extend(gstin)
        final_list.extend(name)
        final_list = [x.upper() for x in final_list if ((x != '') and (x != 'nan'))]
        final_list = [x for x in final_list if (len(x) > 1)]

        dict_address[f] = final_list

    l = list(DF['text'])
    l = [str(x) for x in l]
    s = " ".join(l)
    s = s.upper()

    score = {}
    for key, value in dict_address.items():
        unit_ = key
        list_sub = value
        if len(list_sub) == 0:
            continue
        matches = [x for x in list_sub if x in s]
        ss = len(matches) / len(list_sub)
        if ss >= 0.7:
            score[unit_] = ss

    print("SCORE:", score)
    # Get Model prediction for billing and shipping address
    predicted_billing_address = None
    predicted_shipping_address = None
    if "billingAddress" in prediction and prediction["billingAddress"] is not None:
        predicted_billing_address = prediction["billingAddress"]['text']
    if "shippingAddress" in prediction and prediction["shippingAddress"] is not None:
        predicted_shipping_address = prediction["shippingAddress"]['text']

    print("Model pred Billing:", predicted_billing_address)
    print("Model pred Shipping:", predicted_shipping_address)


    billing_address_scores = {}
    if predicted_billing_address is not None:
        s_address = predicted_billing_address.split()
        s_address = [str(x) for x in s_address]
        s_address = " ".join(s_address)
        s_address = s_address.upper()
        for unit_, v in score.items():
            adress = dict_address[unit_]
            matches = [x for x in adress if x in s_address]
            ss = len(matches) / len(adress)
            if ss >= 0.2:
                billing_address_scores[unit_] = ss

    shipping_address_scores = {}
    if predicted_shipping_address is not None:
        s_address = predicted_shipping_address.split()
        s_address = [str(x) for x in s_address]
        s_address = " ".join(s_address)
        s_address = s_address.upper()
        for unit_, v in score.items():
            adress = dict_address[unit_]
            matches = [x for x in adress if x in s_address]
            ss = len(matches) / len(adress)
            if ss >= 0.2:
                shipping_address_scores[unit_] = ss

    print("billing add score ",billing_address_scores)
    print("shipping address score",shipping_address_scores)


    # Find the predicted value
    # If single address in MasterData matches with the document
    # Return that address as billing as well as shipping address
    # If multiple address in MasterData matched with the document
    # Return the address which has highest match score with predicted billing and shipping address
    # If multiple address in MasterData matched with the document and no prediction for
    # billing and shipping address from the model, return nothing

    billing_unit = None
    shipping_unit = None

    if len(score) == 1:
        billing_unit = max(score.items(), key=operator.itemgetter(1))[0]
        shipping_unit = max(score.items(), key=operator.itemgetter(1))[0]
    elif len(score) > 1:
        if len(billing_address_scores) > 0:
            billing_unit = max(billing_address_scores.items(), key=operator.itemgetter(1))[0]
        if len(shipping_address_scores) > 0:
            shipping_unit = max(shipping_address_scores.items(), key=operator.itemgetter(1))[0]
    
    return_dict = {}
    if billing_unit is not None:
        dict_address = {}
        predicted_address = ADDRESS_MASTERDATA.loc[ADDRESS_MASTERDATA['ID'] == billing_unit].iloc[0].to_dict()

        """
        Validating Picked Bill to, ship to details with GSTIN present in the invoice.
        return predicted address only if the GSTIN matches in the invoice.
        """
        totalGSTINPresent = list(set([putil.correct_gstin(s) for s in list(DF[DF["is_gstin_format"]==1]["text"].unique())]))
        #print("Unique GSTIN in present :",totalGSTINPresent)
        if len(totalGSTINPresent) > 0 and predicted_address.get("GSTIN") not in totalGSTINPresent: # DF[DF["is_gstin_format"]==1]["text"].to_list():
            print("Extracted Billing Details GSTIN from buyers address master Not matching in the invoice",predicted_address)

        else:
            dict_address['line_num'] = 0
            dict_address["Label_Present"] = True
            dict_address['word_num'] = 0
            dict_address['left'] = 0
            dict_address['right'] = 1
            dict_address['conf'] = 1
            dict_address['top'] = 0
            dict_address['bottom'] = 1
            dict_address['page_num'] = 0
            dict_address['image_height'] = 1
            dict_address['image_widht'] = 1
            dict_address["label_confidence"] = None
            dict_address["wordshape"] = None
            dict_address["wordshape_confidence"] = None
            dict_address["Odds"] = None
            dict_address['model_confidence'] = 1
            dict_address['final_confidence_score'] = 1
            dict_address['vendor_masterdata_present'] = True
            dict_address['extracted_from_masterdata'] = False

            dict_address["prob_" + "billingAddress"] = 1
            extracted_value = predicted_address["ADDRESS"]
            dict_address['text'] = extracted_value
            return_dict["billingAddress"] = dict_address

            dict_GSTIN = dict_address.copy()
            del dict_GSTIN["prob_" + "billingAddress"]
            extracted_value = predicted_address["GSTIN"]
            dict_GSTIN['text'] = extracted_value
            dict_GSTIN["prob_" + "billingGSTIN"] = 1
            return_dict["billingGSTIN"] = dict_GSTIN

    if shipping_unit is not None:
        dict_address = {}
        predicted_address = ADDRESS_MASTERDATA.loc[ADDRESS_MASTERDATA['ID'] == shipping_unit].iloc[0].to_dict()

        # return predicted address only if the GSTIN mathes in the invoice 
        totalGSTINPresent = list(set([putil.correct_gstin(s) for s in list(DF[DF["is_gstin_format"]==1]["text"].unique())]))

        if len(totalGSTINPresent) > 0 and predicted_address.get("GSTIN") not in totalGSTINPresent: # DF[DF["is_gstin_format"]==1]["text"].to_list():
            print("Extracted Billing Details GSTIN from buyers address master Not matching in the invoice",predicted_address)

        else:
            dict_address['line_num'] = 0
            dict_address["Label_Present"] = True
            dict_address['word_num'] = 0
            dict_address['left'] = 0
            dict_address['right'] = 1
            dict_address['conf'] = 1
            dict_address['top'] = 0
            dict_address['bottom'] = 1
            dict_address['page_num'] = 0
            dict_address['image_height'] = 1
            dict_address['image_widht'] = 1
            dict_address["label_confidence"] = None
            dict_address["wordshape"] = None
            dict_address["wordshape_confidence"] = None
            dict_address["Odds"] = None
            dict_address['model_confidence'] = 1
            dict_address['final_confidence_score'] = 1
            dict_address['vendor_masterdata_present'] = True
            dict_address['extracted_from_masterdata'] = False

            dict_address["prob_" + "shippingAddress"] = 1
            extracted_value = predicted_address["ADDRESS"]
            dict_address['text'] = extracted_value
            return_dict["shippingAddress"] = dict_address

            dict_GSTIN = dict_address.copy()
            del dict_GSTIN["prob_" + "shippingAddress"]
            extracted_value = predicted_address["GSTIN"]
            dict_GSTIN['text'] = extracted_value
            dict_GSTIN["prob_" + "shippingGSTIN"] = 1
            return_dict["shippingGSTIN"] = dict_GSTIN

    print("final bill /shipping address from master data",return_dict)
    return return_dict

"""
def get_vendor(df, preprocess_vendor):

    l = list(df['text'])
    l = [str(x) for x in l]
    s = " ".join(l)
    s = s.upper()

    dict_score = {}
    dict_vendor, VENDOR_MASTERDATA = refresh_vendor_masterdata(preprocess_vendor)
    for key, value in dict_vendor.items():
        format_ = key
        list_sub = value
        if len(list_sub) == 0:
            continue
        matches = [x for x in list_sub if x in s]
        dict_score[format_] = len(matches) / len(list_sub)
    print(dict_score)
    predicted_format = max(dict_score.items(), key=operator.itemgetter(1))[0]
    max_score = dict_score[predicted_format]
    print(dict_score,"dict***")
    print(dict_vendor,"dictV***")
    print(s,"S***")
    # max_score = 1.0
    if max_score > CONST['vendor_master_data_cutoff_score']:
        # predicted_format = "KGS_0001"
        dict_vendor_data = VENDOR_MASTERDATA.loc[VENDOR_MASTERDATA['VENDOR_ID']
                                                 == predicted_format].iloc[0].to_dict()
        return (predicted_format, max_score, dict_vendor_data)
    else:
        return (None, None, None)
"""
# 15 May 2023 Added to get vendor_id from final_prediction
def GetVendorByPred(final_prediction,VENDOR_MASTERDATA):
    print("vendor data shape :",VENDOR_MASTERDATA.shape)
    if final_prediction.get("vendorGSTIN") != None and final_prediction.get("vendorGSTIN").get("text") != None:
        GSTIN = final_prediction.get("vendorGSTIN").get("text")
        matched_gstin = VENDOR_MASTERDATA[VENDOR_MASTERDATA["VENDOR_GSTIN"]== GSTIN]
        #print("GSTIN :",row.text,"\t Df shape :",VENDOR_MASTERDATA.shape)
        if matched_gstin.shape[0]>0:
            vendorMaster = matched_gstin.iloc[0].to_dict()
            print("matched records :",vendorMaster)
            format_ = vendorMaster["VENDOR_ID"]
            Match_Score = vendorMaster["MATCH_SCORE"]
            return format_, Match_Score,vendorMaster
    return None, None, None

def GetVendor(DF,VENDOR_MASTERDATA):
    # VENDOR_MASTERDATA = pd.read_csv(r"./Utilities/VENDOR_ADDRESS_MASTERDATA.csv")
    # VENDOR_MASTERDATA = VENDOR_MASTERDATA.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    print("vendor data shap :",VENDOR_MASTERDATA.shape)
    DF = DF[DF["is_gstin_format"]==1]
    print("GSTIN data shape :",DF.shape)
    for row in DF.itertuples():
        GSTIN = putil.correct_gstin(row.text)        
        matched_gstin = VENDOR_MASTERDATA[VENDOR_MASTERDATA["VENDOR_GSTIN"]== GSTIN]
        print("GSTIN :",row.text,"\t Df shape :",VENDOR_MASTERDATA.shape)
        if matched_gstin.shape[0]>0:
            vendorMaster = matched_gstin.iloc[0].to_dict()
            print("matched records :",vendorMaster)
            format_ = vendorMaster["VENDOR_ID"]
            Match_Score = vendorMaster["MATCH_SCORE"]
            return format_, Match_Score,vendorMaster
        print("moving to newt item")
    return None, None, None

def get_wordshape(text):
    if not (pd.isna(text)):
        t1 = re.sub('[A-Z]', 'X', text)
        t2 = re.sub('[a-z]', 'x', t1)
        return re.sub('[0-9]', 'd', t2)
    return text


def find_similarity_words(a, b):
    return SequenceMatcher(None, str(a), str(b)).ratio()
@putil.timing
def extract_line_items_new(DF, prediction):
    """
    Code to extract LineItems
    """

    DF['text']=DF['text'].astype(str)
    # Get LineItem Prediction form model
    TEMP = pd.DataFrame()
    pages = list(DF['page_num'].unique())
    for p in pages:
        a = DF.loc[(DF['page_num'] == p) & (DF["predict_label"].isin(FIELD['lineitem_header_labels']))]
        TEMP = pd.concat([TEMP, a], ignore_index=True)
        a = DF.loc[(DF['page_num'] == p) & (DF["predict_label"].isin(FIELD['lineitem_value_labels']))]
        TEMP = pd.concat([TEMP, a], ignore_index=True)

    if TEMP.shape[0]==0:
        return {}
    TEMP.sort_values(['page_num', 'line_num', 'word_num'], ascending=True, inplace=True)

    SS = TEMP.sort_values(['word_num'], ascending=True).groupby(['page_num', 'line_num',
                            'predict_label']).agg(
        {'text': lambda x: "%s" % ' '.join(x),
         'prediction_probability': 'mean',
         'word_num': 'count',
         'left': 'min',
         'right': 'max',
         'top': 'min',
         'bottom': 'max',
         'image_height': 'first',
         'image_widht': 'first'}).reset_index()

    SS = SS.loc[SS['prediction_probability'] > CONST['model_threshold']]
    if SS.shape[0]==0:
        return {}

    SS.sort_values(['page_num', 'predict_label', 'top'], ascending=[True, True, True], inplace=True)

    # Find top boundry
    # Code Starts
    pages = set(SS['page_num'])
    for p in pages:
        PAGE = SS.loc[SS['page_num'] == p]
        top = list(PAGE['top'])
        top.sort()
        indices = [i + 1 for (x, y, i) in zip(top, top[1:],
                   range(len(top))) if CONST['top_variance'] < abs(x - y)]
        result = [top[start:end] for start, end in zip([0] + indices, indices + [len(top)])]
        line_item_area_id = 1
        for res in result:
            SS.loc[SS['top'].isin(res), 'line_item_area_id'] = line_item_area_id
            line_item_area_id += 1

    count_LIHeader_labels_area = SS.loc[SS['predict_label'].isin(FIELD['lineitem_header_labels'])].groupby(
        ['page_num', 'line_item_area_id']).agg(LI_header_count = ('predict_label', 'count')).reset_index()
    count_LI_labels_area = SS.groupby(['page_num', 'line_item_area_id']).agg(LI_label_count = ('predict_label', 'count'),
        top_bounding_box = ('top', 'min'),
        bottom_bounding_box = ('bottom', 'max')).reset_index()
    Count_labels_area = count_LI_labels_area.merge(count_LIHeader_labels_area, on=['page_num', 'line_item_area_id'],
                                                   how='left')

    Count_labels_area.fillna(0, inplace=True)
    Count_labels_area.sort_values(['page_num', 'LI_header_count', 'LI_label_count', 'top_bounding_box'],
     ascending=[True, False, False, True], inplace=True)

    AA = Count_labels_area.groupby(['page_num']).first().reset_index()

    # Find top boundry
    # Code ends

    # Remove the code to extract bottom bounding box from total prediction
    '''
    # Find bottom boundary
    prediction_total = {l: {l: prediction[l] for l in prediction}[l] for l
                        in FIELD['amount_fields']}

    print(prediction_total)
    dict_bottom = {}
    for key, value in prediction_total.items():
        if value is not None:
            if (value['final_confidence_score'] < 0.5) | (key == 'taxAmount'):
                continue
            p = value['page_num']
            bottom = value['top']
            if p in dict_bottom:
                old_bottom = dict_bottom[p]
                if bottom < old_bottom:
                    dict_bottom[p] = bottom
            else:
                dict_bottom[p] = bottom

    # Remove pages after amount prediction
    if len(dict_bottom) > 0:
        max_lineitem_page = max(dict_bottom)
        AA = AA.loc[AA['page_num'] <= max_lineitem_page]

    AA['bottom_bounding_box'] = 1.0
    for key, value in dict_bottom.items():
        AA.loc[(AA['page_num'] == key), 'bottom_bounding_box'] = value
    '''

    # Remove pages after amount prediction
    prediction_total = {l: {l: prediction[l] for l in prediction}[l] for l in FIELD['amount_fields']}
    pages_amount = set()
    for key, value in prediction_total.items():
        if value is not None:
            if (value['final_confidence_score'] < 0.5):
                continue
            pages_amount.add(value['page_num'])

    if len(pages_amount) > 0:
        max_lineitem_page = min(pages_amount)
        AA = AA.loc[AA['page_num'] <= max_lineitem_page]

    DF_LINEITEMS = pd.DataFrame()
    # Extract LineItems
    DF_BOUNDRY = AA[['page_num', 'top_bounding_box', 'bottom_bounding_box']].drop_duplicates()

    for idx, row in DF_BOUNDRY.iterrows():
        p = row['page_num']
        top = row['top_bounding_box']
        bottom = row['bottom_bounding_box']
        PAGE = DF.loc[(DF['page_num'] == p) & (DF['top'] >= top) & (DF['top'] < bottom)]
        if PAGE.empty:
            continue

        PAGE.sort_values(['line_num', 'word_num'], ascending=[True, True], inplace=True)

        # Fix added for Azure Line Number
        PAGE['is_amount'] = (PAGE['text'].apply(is_amount_) | PAGE['text'].str.isnumeric())
        TEMP_LINES = PAGE.groupby(['line_num']).agg({'text': 'count',
            'is_amount': 'sum'}).reset_index()

        TEMP_LINES['percentage_numeric_words'] = TEMP_LINES['is_amount']/TEMP_LINES['text']
        TEMP_LINES = TEMP_LINES.loc[((TEMP_LINES['text'] > 1)
            & (TEMP_LINES['percentage_numeric_words'] >= 0.4))]

        line_num_updated = TEMP_LINES['line_num'].unique()
        for l in line_num_updated:
            PAGE.loc[(PAGE['line_num'] == l),'line_num'] = PAGE['line_num'] + PAGE['word_num']/10

        # Fix end  for Azure Line Number
        PAGE_LINES = PAGE.groupby(['line_num']).agg({'text': lambda x: "%s" % ' '.join(x),
                                                     'word_num': 'count',
                                                     'left': 'min',
                                                     'right': 'max',
                                                     'top': 'min',
                                                     'bottom': 'max',
                                                     'conf': 'mean',
                                                     'image_height': 'first',
                                                     'image_widht': 'first'}).reset_index()


        label_df = PAGE.groupby(['line_num']).apply(get_top_label).reset_index()
        label_df.columns = ['line_num', 'predict_label']
        # Get Model Probability
        label_df = label_df.merge(PAGE[['line_num', 'predict_label', 'prediction_probability']],
                                  on=['line_num', 'predict_label'], how='left')
        label_df = label_df.groupby(['line_num', 'predict_label'])[['prediction_probability']].mean().reset_index()

        old_shape = PAGE_LINES.shape
        PAGE_LINES = PAGE_LINES.merge(label_df, on='line_num', how='left')
        assert (old_shape[0] == PAGE_LINES.shape[0])
        assert (old_shape[1] + 2 == PAGE_LINES.shape[1])

        # Divide table into rows
        top_bottom = list(zip(PAGE_LINES['top'], PAGE_LINES['bottom']))
        ll = sort_tuple_by_first_value(top_bottom)
        indices = [i + 1 for (x, y, i) in zip(ll, ll[1:], range(len(ll))) if CONST['row_dividing_threshold'] < (y[0] - x[1])]
        result = [ll[start:end] for start, end in zip([0] + indices, indices + [len(ll)])]
        PAGE_LINES['row_num_extracted'] = ''
        row_num_extracted = 0
        for res in result:
            PAGE_LINES.loc[(pd.Series(list(zip(PAGE_LINES['top'], PAGE_LINES['bottom']))).isin(res)),
                           'row_num_extracted'] = row_num_extracted
            row_num_extracted += 1

        # Discard rows which have very few header predictions as False Positives
        count_LIHeader_labels_row = PAGE_LINES.loc[PAGE_LINES['predict_label'].isin(FIELD['lineitem_header_labels'])].groupby(
            ['row_num_extracted']).agg(LI_header_count = ('predict_label', 'count')).reset_index()
        count_LIValue_labels_row = PAGE_LINES.loc[PAGE_LINES['predict_label'].isin(FIELD['lineitem_value_labels'])].groupby(
            ['row_num_extracted']).agg(LI_label_count = ('predict_label', 'count')).reset_index()
        count_labels_row = count_LIValue_labels_row.merge(count_LIHeader_labels_row, on=['row_num_extracted'],
            how='outer')
        count_labels_row.fillna(0, inplace=True)
        header_prediction_row_threshold = 3
        count_labels_row = count_labels_row.loc[count_labels_row['LI_header_count'] >= header_prediction_row_threshold]
        count_labels_row.sort_values(['LI_header_count', 'row_num_extracted'], ascending=[False, True], inplace=True)

        if not count_labels_row.empty:
            header_row_num = count_labels_row.iloc[0]['row_num_extracted']
            PAGE_LINES = PAGE_LINES.loc[PAGE_LINES['row_num_extracted'] >= header_row_num]

        # Divide table into columns
        # Divide col based on left alignment
        left = list(PAGE_LINES['left'])
        left.sort()
        indices = [i + 1 for (x, y, i) in zip(left, left[1:], range(len(left))) if CONST['col_dividing_threshold'] < abs(x - y)]
        result = [left[start:end] for start, end in zip([0] + indices, indices + [len(left)])]
        PAGE_LINES['left_align_col_num_extracted'] = ''
        col_num_extracted = 0
        for res in result:
            PAGE_LINES.loc[(PAGE_LINES['left'].isin(res)), 'left_align_col_num_extracted'] = col_num_extracted
            col_num_extracted += 1

        # Correct col number based on right alignment
        right = list(PAGE_LINES['right'])
        right.sort()
        indices = [i + 1 for (x, y, i) in zip(right, right[1:], range(len(right))) if
                  CONST['col_dividing_threshold'] < abs(x - y)]
        result = [right[start:end] for start, end in zip([0] + indices, indices + [len(right)])]
        PAGE_LINES['right_align_col_num_extracted'] = ''
        col_num_extracted = 0
        for res in result:
            PAGE_LINES.loc[(PAGE_LINES['right'].isin(res)), 'right_align_col_num_extracted'] = col_num_extracted
            col_num_extracted += 1

        PAGE_LINES['col_num_extracted'] = PAGE_LINES['left_align_col_num_extracted']
        for right_col in list(PAGE_LINES['right_align_col_num_extracted'].unique()):
            left_cols = list(PAGE_LINES.loc[PAGE_LINES['right_align_col_num_extracted'] ==
                                            right_col]['col_num_extracted'])
            if len(set(left_cols)) > 1:
                left_cols.sort()
                new_col_number = max(left_cols, key=left_cols.count)
                if left_cols.count(new_col_number) > len(left_cols) / 2:
                    PAGE_LINES.loc[(PAGE_LINES['right_align_col_num_extracted'] == right_col),
                                   'col_num_extracted'] = new_col_number

        # Correct col number based on center alignment
        PAGE_LINES['center'] = (PAGE_LINES['left'] + PAGE_LINES['right']) / 2.0
        center = list(PAGE_LINES['center'])
        center.sort()
        indices = [i + 1 for (x, y, i) in zip(center, center[1:], range(len(center)))
                   if CONST['col_dividing_threshold'] < abs(x - y)]
        result = [center[start:end] for start, end in zip([0] + indices, indices + [len(center)])]
        PAGE_LINES['center_align_col_num_extracted'] = ''
        col_num_extracted = 0
        for res in result:
            PAGE_LINES.loc[(PAGE_LINES['center'].isin(res)), 'center_align_col_num_extracted'] = col_num_extracted
            col_num_extracted += 1

        for center_col in list(PAGE_LINES['center_align_col_num_extracted'].unique()):
            left_cols = list(PAGE_LINES.loc[PAGE_LINES['center_align_col_num_extracted'] ==
                                            center_col]['col_num_extracted'])
            if len(set(left_cols)) > 1:
                left_cols.sort()
                new_col_number = max(left_cols, key=left_cols.count)
                if left_cols.count(new_col_number) > len(left_cols) / 2:
                    PAGE_LINES.loc[(PAGE_LINES['center_align_col_num_extracted'] == center_col),
                                   'col_num_extracted'] = new_col_number

        # Take Column with Amount as reference to further divide rows
        count_amount_column = 0
        reference_column_number = None
        for c in list(PAGE_LINES['col_num_extracted'].unique()):
            texts = list(PAGE_LINES.loc[PAGE_LINES['col_num_extracted'] == c]['text'])
            texts = [str(x).replace(" ", "") for x in texts]
            texts = [str(x).replace(",", "") for x in texts]
            texts = [str(x).replace("$", "") for x in texts]
            amounts = [convert_float(x) for x in texts]
            amounts = [x for x in amounts if x != '']
            if len(amounts) > count_amount_column:
                count_amount_column = len(amounts)
                reference_column_number = c


        # Build Final Table
        row_top_values = list(PAGE_LINES.loc[PAGE_LINES['col_num_extracted'] == reference_column_number]['top'])
        row_top_values.sort()

        PAGE_LINES['new_row_number'] = 999

        new_row_num = 0
        # Divide rows based on amount
        # Anything below or in line of amount is considered as one row
        #row_top_values.append(1.0)
        # print("XXXXXXX:", row_top_values)

        if len(row_top_values) > 0:
            top_value = row_top_values[0] - 0.002
            PAGE_LINES.loc[((PAGE_LINES['top'] >= top_value)), 'new_row_number'] = new_row_num
            new_row_num += 1
            if len(row_top_values) > 1:
                for top_value in row_top_values[1:]:
                    top_value = top_value - 0.002
                    PAGE_LINES.loc[((PAGE_LINES['top'] >= top_value)), 'new_row_number'] = new_row_num
                    new_row_num += 1

        # print(PAGE_LINES[['text','top', 'predict_label', 'new_row_number']])

        # Divide rows based on amount
        # Anything above or in line of amount is considered as one row
        # previous_top_value = 0
        # for top_value in row_top_values:
        #     PAGE_LINES.loc[((PAGE_LINES['top'] > previous_top_value) & (PAGE_LINES['top'] <= top_value)),
        #                    'new_row_number'] = new_row_num
        #     previous_top_value = top_value
        #     new_row_num += 1

        # NOTE: Issue fixed: line items getting combined as one; Date 11-Dec-2020
        if len(list(PAGE_LINES['row_num_extracted'].unique())) > 1:
            PAGE_LINES.loc[(PAGE_LINES['row_num_extracted'] == 0), 'new_row_number'] = -1

        PAGE_LINES.sort_values(['new_row_number', 'col_num_extracted', 'row_num_extracted'], inplace=True)
        KK = PAGE_LINES.groupby(['new_row_number', 'col_num_extracted',
                                 'row_num_extracted']).agg({'text': lambda x: "%s" % ' '.join(x),
                                                            'word_num': 'sum',
                                                            'left': 'min',
                                                            'right': 'max',
                                                            'top': 'min',
                                                            'bottom': 'max',
                                                            'conf': 'mean',
                                                            'image_height': 'first',
                                                            'image_widht': 'first'}).reset_index()

        # Get Model Prediction and Prediction Probability
        label_df = PAGE_LINES.groupby(['new_row_number', 'col_num_extracted',
                                       'row_num_extracted']).apply(get_top_label).reset_index()
        label_df.columns = ['new_row_number', 'col_num_extracted', 'row_num_extracted', 'predict_label']
        # Get Model Probability
        label_df = label_df.merge(PAGE_LINES[['new_row_number', 'col_num_extracted', 'row_num_extracted',
                                              'predict_label', 'prediction_probability']],
                                  on=['new_row_number', 'col_num_extracted', 'row_num_extracted', 'predict_label'],
                                  how='left')
        label_df = label_df.groupby(['new_row_number', 'col_num_extracted', 'row_num_extracted',
                                     'predict_label'])[['prediction_probability']].mean().reset_index()
        old_shape = KK.shape
        KK = KK.merge(label_df, on=['new_row_number', 'col_num_extracted', 'row_num_extracted'], how='left')
        assert (old_shape[0] == KK.shape[0])
        assert (old_shape[1] + 2 == KK.shape[1])

        KK['FileName'] = list(DF['FileName'].unique())[0]
        KK['page_num'] = p

        DF_LINEITEMS = pd.concat([DF_LINEITEMS, KK], ignore_index=True)

    if DF_LINEITEMS.empty:
        return {}

    pages = list(DF_LINEITEMS['page_num'].unique())
    dict_df_pages = {}
    for p in pages:
        DF_LINEITEM_PAGE = DF_LINEITEMS.loc[(DF_LINEITEMS['page_num'] == p) & (DF_LINEITEMS['new_row_number'] != 999)]
        if DF_LINEITEM_PAGE.empty:
            continue
        DF_LINEITEM_PAGE.sort_values(['new_row_number', 'col_num_extracted'], inplace=True)
        AA = DF_LINEITEM_PAGE.groupby(['new_row_number', 'col_num_extracted']).agg(
            {'text': lambda x: "%s" % '\n'.join(x),
             'left': 'min',
             'right': 'max',
             'top': 'min',
             'bottom': 'max',
             'conf': 'mean',
             'image_height': 'first',
             'image_widht': 'first'}).reset_index()

        # Get Mode Prediction and Prediction Probability
        label_df = DF_LINEITEM_PAGE.groupby(['new_row_number',
                                             'col_num_extracted']).apply(get_top_label).reset_index()
        label_df.columns = ['new_row_number', 'col_num_extracted', 'predict_label']
        # Get Model Probability
        label_df = label_df.merge(DF_LINEITEM_PAGE[['new_row_number', 'col_num_extracted',
                                                    'predict_label', 'prediction_probability']],
                                  on=['new_row_number', 'col_num_extracted', 'predict_label'], how='left')

        label_df = label_df.groupby(['new_row_number', 'col_num_extracted',
                                     'predict_label'])[['prediction_probability']].mean().reset_index()
        old_shape = AA.shape
        AA = AA.merge(label_df, on=['new_row_number', 'col_num_extracted'], how='left')
        assert (old_shape[0] == AA.shape[0])
        assert (old_shape[1] + 2 == AA.shape[1])

        AA['page_num'] = p
        # BB = pd.pivot_table(AA, index='new_row_number', columns='col_num_extracted', values='text',
        #                     aggfunc='first')
        # BB.reset_index(inplace=True)
        # print(BB)
        # Make first row (new_row_num = -1 ) as column name
        # BB.columns = BB.iloc[0]
        # BB = BB.drop(BB.index[0])
        # Rename column named as -1 to new_row_num
        # BB.rename(columns={-1: 'new_row_num'}, inplace=True)
        dict_df_pages[p] = AA

    # Keep rows and columns which have predicted LineItems
    # Commented due to poor model prediction
    # Line-items are predicted sparsley from the model
    # for key, val in dict_df_pages.items():
    #     print("************************", key)
    #     rows = list(val['new_row_number'].unique())
    #     final_rows = []
    #     for r in rows:
    #         row_text = list(val.loc[val['new_row_number'] == r]['text'])
    #         row_predictions = list(val.loc[val['new_row_number'] == r]['predict_label'])
    #         print(r, row_text, row_predictions)
    #         lineitem_labels = FIELD['lineitem_header_labels'] + FIELD['lineitem_value_labels']
    #         ll = [x for x in row_predictions if x in lineitem_labels]
    #         if len(ll) > 0:
    #             final_rows.append(c)
    #     dict_df_pages[key] = val.loc[val['new_row_number'].isin(final_rows)]

    #     cols = list(val['col_num_extracted'].unique())
    #     final_cols = []
    #     for c in cols:
    #         col_text = list(val.loc[val['col_num_extracted'] == c]['text'])
    #         col_predictions = list(val.loc[val['col_num_extracted'] == c]['predict_label'])
    #         lineitem_labels = FIELD['lineitem_header_labels'] + FIELD['lineitem_value_labels']
    #         ll = [x for x in col_predictions if x in lineitem_labels]
    #         if len(ll) > 0:
    #             final_cols.append(c)
    #     dict_df_pages[key] = val.loc[val['col_num_extracted'].isin(final_cols)]

    # Rename column name as continuous Alphabet
    for key, val in dict_df_pages.items():
        cols = list(val['col_num_extracted'].unique())
        cols.sort()
        new_mapping = {}
        for idx, v in enumerate(cols):
            new_mapping[v] = chr(65 + idx)
        new_val = val.replace({"col_num_extracted": new_mapping})
        dict_df_pages[key] = new_val

    # Combine Columns further (if a lot of missing values in the column)
    for key, val in dict_df_pages.items():
        count_rows = len(list(val['new_row_number'].unique()))
        cols = list(val['col_num_extracted'].unique())
        cols.sort()
        cols_to_collapse = []
        for c in cols:
            col_text = list(val.loc[val['col_num_extracted'] == c]['text'])
            if len(col_text) < (count_rows/2): # If count values in columns is half of total rows(Edited for AzureLineFIx)

            # if len(col_text) < (count_rows - 2):  # Take into consideration two total rows
                # print(c, len(col_text), (count_rows - 2), col_text)
                cols_to_collapse.append(c)

        cols_to_keep = list(set(cols) - set(cols_to_collapse))
        cols_to_keep.sort()
        cols_to_collapse.sort()
        # print("Collapse:", cols_to_collapse)
        # print("Keep:", cols_to_keep)
        temp = val.groupby(['col_num_extracted']).agg({'left': 'min', 'right': 'max'}).reset_index()
        temp['left'] = temp['left'].astype(float)
        temp['right'] = temp['right'].astype(float)
        temp['col_span'] = list(zip(temp['left'], temp['right']))
        col_span = dict(zip(temp['col_num_extracted'], temp['col_span']))

        new_mapping = {}
        cols_to_collapse_left = []

        for c in cols_to_collapse:
            interval_collapse = col_span[c]
            dict_overlap_length = {}
            for k in cols_to_keep:
                interval_keep = col_span[k]
                overlap_length = min(interval_collapse[1], interval_keep[1]) - max(interval_collapse[0],
                                                                                   interval_keep[0])
                dict_overlap_length[k] = overlap_length
                dict_overlap_length = dict((k, v) for k, v in dict_overlap_length.items() if v >= 0)
            if len(dict_overlap_length) > 0:
                new_col = max(dict_overlap_length.items(), key=operator.itemgetter(1))[0]
                new_mapping[c] = new_col
            else:
                cols_to_collapse_left.append(c)

        cols_to_collapse_left.sort()
        # print(cols_to_collapse_left)
        dict_final_collapse = {}
        for c in cols_to_collapse_left:
            interval_collapse = col_span[c]
            if len(dict_final_collapse) == 0:
                # Take this column as final one if no collapsed column found
                dict_final_collapse[interval_collapse] = [c]
            else:
                # Find the column in which to collapse
                collapse_col_found = False
                for k in list(dict_final_collapse):
                    v = dict_final_collapse[k]
                    if ((k[0] <= interval_collapse[0]) & (interval_collapse[0] <= k[1])):
                        # Left of the collapse column lies in the interval
                        new_key = (min(k[0], interval_collapse[0]), max(k[1], interval_collapse[1]))
                        v.append(c)
                        del dict_final_collapse[k]
                        dict_final_collapse[new_key] = v
                        collapse_col_found = True
                        break
                    elif ((k[0] <= interval_collapse[1]) & (interval_collapse[1] <= k[1])):
                        # Right of the collapse column lies in the interval
                        new_key = (min(k[0], interval_collapse[0]), max(k[1], interval_collapse[1]))
                        v.append(c)
                        del dict_final_collapse[k]
                        dict_final_collapse[new_key] = v
                        collapse_col_found = True
                        break
                if not collapse_col_found:
                    dict_final_collapse[interval_collapse] = [c]
        # print(dict_final_collapse)

        for k, v in dict_final_collapse.items():
            if len(v) > 1:
                col = v[0]
                for index in range(1, len(v)):
                    col_to_collapse = v[index]
                    new_mapping[col_to_collapse] = col

        dict_df_pages[key] = val.replace({'col_num_extracted': new_mapping})



    # Mark the cells with HeaderLineItems
    for key, val in dict_df_pages.items():
        val.loc[(val['predict_label'].isin(FIELD['lineitem_header_labels'])), 'new_row_number'] = 777

    # Final Prediction
    page_prediction = {}
    for key, val in dict_df_pages.items():
        temp = val.loc[~val['new_row_number'].isin([888])]
        cols = list(temp['col_num_extracted'].unique())
        col_labels = {}
        col_counts = {}
        for c in cols:
            prediction = list(temp.loc[temp['col_num_extracted'] == c]['predict_label'])
            prediction = [x for x in prediction if '_' in x]
            if len(prediction)==0:
                col_labels[c] = 'UNKNOWN'
                continue
            prediction = dict(Counter([x[x.index('_') + 1:] for x in prediction]))
            if len(prediction) == 1:
                col_labels[c] = list(prediction.keys())[0]
            else:
                col_counts[c] = prediction

        cols_found = set(col_labels.values())
        remaining_cols = sorted(set(cols) - set(col_labels.keys()))
        for c in remaining_cols:
            prediction = col_counts[c]
            d_descending = dict(OrderedDict(sorted(prediction.items(), key=lambda kv: kv[1], reverse=True)))
            for k, v in d_descending.items():
                if k not in cols_found:
                    col_labels[c] = k
                    cols_found.add(k)
                    break

        remaining_cols = sorted(set(cols) - set(col_labels.keys()))
        for c in remaining_cols:
            col_labels[c] = "UNKNOWN"

        for k, v in col_labels.items():
            if v == "UNKNOWN":
                # Discard the column
                temp = temp.loc[temp['col_num_extracted'] != k]

        # Remove Header Rows
        temp = temp.loc[~temp['new_row_number'].isin([777])]
        # Remove NaN rows
        rows = list(temp['new_row_number'].unique())
        for r in rows:
            text = list(temp.loc[temp['new_row_number'] == r]['text'])
            if len(text) == 0:
                # Remove Row
                temp = temp.loc[temp['new_row_number'] != r]

        # Code to remove duplicate columns
        #print("Column Labels:", col_labels)
        cols = list(temp['col_num_extracted'].unique())
        col_label_confidence = {}
        for c in cols:
            text = list(temp.loc[temp['col_num_extracted'] == c]['text'])
            prediction = list(temp.loc[temp['col_num_extracted'] == c]['predict_label'])
            label_ = col_labels[c]
            count_labels = [x for x in prediction if x == ("LI_"+label_)]
            l = list(temp.loc[(temp['col_num_extracted'] == c)
                & (temp['predict_label'] == "LI_"+label_)]['prediction_probability'])
            prediction_probability = 0
            if len(l) > 0:
                prediction_probability = sum(l)/float(len(l))
            col_label_confidence[c] = (len(count_labels) / len(prediction), prediction_probability)

        # print(col_label_confidence)
        cols = list(temp['col_num_extracted'].unique())
        cols_to_keep = []
        for c in cols:
            col_label = col_labels[c]
            t = {k:v for k, v in col_labels.items() if v == col_label}
            if len(t) > 1:
                m = {k:v for k,v in col_label_confidence.items() if k in t.keys()}
                col_to_keep = max(m.items(), key=operator.itemgetter(1))[0]
                cols_to_keep.append(col_to_keep)
            else:
                cols_to_keep.append(c)
        cols_to_keep = list(set(cols_to_keep))
        temp = temp.loc[temp['col_num_extracted'].isin(cols_to_keep)]

        #Take itemValue as the benchmark
        anchor_col = 'UNKNOWN'
        cols_found = set(col_labels.values())
        if 'itemValue' in cols_found:
            anchor_col = (list(col_labels.keys())[list(col_labels.values()).index('itemValue')])
        elif 'unitPrice' in cols_found:
            anchor_col = (list(col_labels.keys())[list(col_labels.values()).index('unitPrice')])
        elif 'itemQuantity' in cols_found:
            anchor_col = (list(col_labels.keys())[list(col_labels.values()).index('itemQuantity')])

        # print("Anchor Column:", anchor_col)
        # print(col_labels)
        #
        # print(pd.pivot_table(temp, index='new_row_number', columns='col_num_extracted', values='text',aggfunc='first'))
        # Remove Columns with Sparse entry
        '''
        count_rows = len(rows)
        cols = list(temp['col_num_extracted'].unique())
        for c in cols:
            text = list(temp.loc[temp['col_num_extracted'] == c]['text'])
            if len(text) < count_rows / 2:
                # Remove Column
                temp = temp.loc[temp['col_num_extracted'] != c]

        temp = temp.loc[temp['col_num_extracted'].isin(cols_to_keep)]
        '''

        # Code Starts: Fix to pick final lineitem from multiple sections of lineitems based on horizontal gap in them
        # top_min = temp.groupby('new_row_number')[['top']].min().reset_index()
        # bottom_max = temp.groupby('new_row_number')[['bottom']].max().reset_index()
        # top_bottom = pd.merge(top_min, bottom_max, on='new_row_number')
        # top_bottom.sort_values('new_row_number', ascending=True, inplace=True)

        # top_values = list(top_bottom['top'])
        # bottom_values = list(top_bottom['bottom'])
        # row_numbers = list(top_bottom['new_row_number'])

        # if (len(row_numbers) > 1) & (len(row_numbers) == len(top_values)) & (len(top_values) == len(bottom_values)):
        #     top_values = top_values[1:]
        #     bottom_values = bottom_values[:-1]
        #     row_numbers = row_numbers[1:]

        #     row_gap = [x1 - x2 for (x1, x2) in zip(top_values, bottom_values)]

        #     # Take first row gap as reference
        #     reference_gap = row_gap[0]
        #     threshold_gap = reference_gap*3
        #     print(reference_gap)
        #     row_gap = list(zip(row_numbers, row_gap))
        #     print(row_gap)
        #     demarking_rows = [(x1, x2) for (x1, x2) in row_gap if x2 > threshold_gap]

        #     print(demarking_rows)
        #     temp['line_item_section'] = 1
        #     line_item_section = 2
        #     for v in demarking_rows:
        #         r = v[0]
        #         temp.loc[temp['new_row_number'] >= r, 'line_item_section'] = line_item_section
        #         line_item_section += 1

        #     ls_sec = temp.groupby('line_item_section')[['prediction_probability']].mean().reset_index()
        #     final_lineitem_section = ls_sec.sort_values('prediction_probability', ascending=False).iloc[0]['line_item_section']
        #     temp = temp.loc[temp['line_item_section'] == final_lineitem_section]

        #     del temp['line_item_section']
        # Code Ends: Fix to pick final lineitem from multiple sections of lineitems based on horizontal gap in them


        rows = list(temp['new_row_number'].unique())
        rows.sort()
        row_number = 1
        row_prediction = {}

        for r in rows:

            temp_row = temp.loc[temp['new_row_number'] == r]
            temp_row = temp_row.groupby(['new_row_number', 'col_num_extracted']).agg({
                'text': lambda x: "%s" % ' '.join(x),
                'prediction_probability': 'mean', 'conf': 'mean',
                'left': 'min',
                'right': 'max',
                'top': 'min',
                'bottom': 'max',
                'image_height': 'first',
                'image_widht': 'first'}).reset_index()

            temp_row["Odds"] = temp_row['prediction_probability'] / (1 - temp_row['prediction_probability'])
            temp_row['model_confidence'] = temp_row['prediction_probability'].apply(logistic_function)

            temp_row.replace({"col_num_extracted": col_labels}, inplace=True)
            extracted_fields = list(temp_row['col_num_extracted'])
            list_extracted_fields = []
            for ext in extracted_fields:
                ss = temp_row.loc[temp_row['col_num_extracted'] == ext]
                del ss['col_num_extracted']
                del ss['new_row_number']
                p = {ext: ss.iloc[0].to_dict()}
                list_extracted_fields.append(p)
            row_prediction[row_number] = list_extracted_fields
            row_number += 1
        page_prediction[key] = row_prediction

    # print(pd.pivot_table(temp, index='new_row_number', columns='col_num_extracted', values='text',aggfunc='first'))

    dict_df = []

    for page, val in page_prediction.items():
        for row_num, line_items in val.items():
            dict_row = {'page_num':page,
            'row_num': row_num}
            for x in line_items:
                col_name = list(x.keys())[0]
                col_value = list(x.values())[0]['text']
                dict_row[col_name] = col_value
            dict_df.append(dict_row)

    DF_LI = pd.DataFrame(dict_df)
    if DF_LI.empty:
        return {}
    count_prediceted_cols = len(DF_LI.columns)-2
    count_predicted_rows = DF_LI.shape[0]
    minimum_lineitem_cols = math.ceil(count_prediceted_cols*0.4)
    nan_columns_allowed = count_prediceted_cols - minimum_lineitem_cols

    dict_rows_to_remove = {}
    for index, rows in DF_LI.iterrows():
        page_num = rows['page_num']
        row_num = rows['row_num']
        count_nan = rows.isnull().sum()
        if count_nan > nan_columns_allowed:
            if page_num in dict_rows_to_remove:
                val = dict_rows_to_remove[page_num]
                val.append(row_num)
                dict_rows_to_remove[page_num] = val
            else:
                dict_rows_to_remove[page_num] = [row_num]

    for key, val in dict_rows_to_remove.items():
        DF_LI = DF_LI.loc[~((DF_LI['page_num'] == key) & (DF_LI['row_num'].isin(val)))]

    cols_to_remove = list(DF_LI.columns[DF_LI.isna().all()])
    cols_to_remove.extend(['page_num', 'row_num'])
    cols_to_keep = [x for x in list(DF_LI.columns) if x not in cols_to_remove]

    pages = list(DF_LI['page_num'].unique())
    new_row_mapping_page = {}
    for p in pages:
        old_rows = list(DF_LI.loc[DF_LI['page_num'] == p]['row_num'].unique())
        old_rows.sort()
        new_row_num = 1
        new_row_mapping = {}
        for r in old_rows:
            new_row_mapping[r] = new_row_num
            new_row_num += 1
        new_row_mapping_page[p] = new_row_mapping


    for page, val in page_prediction.items():
        if page in dict_rows_to_remove:
            rows_to_remove = dict_rows_to_remove[page]
            for k in rows_to_remove:
                del val[k]
            page_prediction[page] = val

    empty_pages = []
    for page, val in page_prediction.items():
        if len(val) == 0:
            empty_pages.append(page)

    for page in empty_pages:
        del page_prediction[page]

    for page, val in page_prediction.items():
        keys = new_row_mapping_page[page]
        page_prediction[page] = dict([(keys.get(k), v) for k, v in val.items()])

    for page, val in page_prediction.items():
        for row_num, line_item in val.items():
            new_line_item = []
            temp = copy.deepcopy(cols_to_keep)
            for items in line_item:
                for k, v in items.items():
                    if k in temp:
                        new_line_item.append(items)
                        temp.remove(k)
            if len(temp) > 0:
                for t in temp:
                    line_item_to_be_added = {t: {'text': '-', 'prediction_probability': 0.0, 'conf': 0.0, 'left': 0.0,
                    'right': 0.0, 'top': 0.0, 'bottom': 0.0, 'image_height': 0, 'image_widht': 0, 'Odds': 0.0,
                    'model_confidence': 0.0}}
                    new_line_item.append(line_item_to_be_added)
            val[row_num] = new_line_item
        page_prediction[page] = val

    for page, val in page_prediction.items():
        for row_num, line_item in val.items():
            line_item = sorted(line_item, key=lambda x:list(x.keys())[0])
            val[row_num] = line_item
        page_prediction[page] = val

    # Code added to rename UOM as per validations
    page_prediction = UOM_SWAP_CHECK(page_prediction, DF_LI)
    page_prediction = refine_lineitem_prediction(page_prediction)
    page_prediction = derive_lineitem_columns(page_prediction)
    print(page_prediction)

    return page_prediction
def check_amount_field_presence_in_invoice(filter_df,amount,tol):
    for idx,row in filter_df.iterrows():
        if math.isclose(float(row["extracted_amount"]),amount,abs_tol=tol):
            return True
    return False

def add_NA_to_cities(final_prediction):
    final_prediction_copy = copy.deepcopy(final_prediction)
    try:
        list_of_fields = ["poNumber","billingCity","shippingCity"]
        for item in list_of_fields:
            if final_prediction.get(item)!= None:
                if final_prediction[item]["text"] == "" or final_prediction[item].get("text") == None :
                    final_prediction[item]["text"] = "N/A"
        return final_prediction
    except:
        print(traceback.print_exc(),
            "add_NA_to_cities Exception")
        return final_prediction_copy
@putil.timing
def add_zero_to_empty_tax_slabs(final_prediction,docMetaData):
    final_prediction_copy = copy.deepcopy(final_prediction)
    try:
        if docMetaData.get("result")!= None and docMetaData.get("result").get("document") != None and docMetaData.get("result").get("document").get("docType") == "Invoice":
            list_of_slabs = ["CGSTAmount_2.5%","SGSTAmount_2.5%","IGSTAmount_5%","subTotal_5%","CGSTAmount_6%","SGSTAmount_6%","IGSTAmount_12%","subTotal_12%","CGSTAmount_9%","SGSTAmount_9%","IGSTAmount_18%","subTotal_18%","CGSTAmount_14%","SGSTAmount_14%","IGSTAmount_28%","subTotal_28%","subTotal_0%","tcsAmount","CessAmount","additionalCessAmount","discountAmount","totalAmount"]
        else:
            list_of_slabs = ["CGSTAmount_2.5%","SGSTAmount_2.5%","IGSTAmount_5%","subTotal_5%","CGSTAmount_6%","SGSTAmount_6%","IGSTAmount_12%","subTotal_12%","CGSTAmount_9%","SGSTAmount_9%","IGSTAmount_18%","subTotal_18%","CGSTAmount_14%","SGSTAmount_14%","IGSTAmount_28%","subTotal_28%","subTotal_0%","totalAmount"]
        # list_of_slabs = ["CGSTAmount_2.5%","SGSTAmount_2.5%","IGSTAmount_5%","subTotal_5%","CGSTAmount_6%","SGSTAmount_6%","IGSTAmount_12%","subTotal_12%","CGSTAmount_9%","SGSTAmount_9%","IGSTAmount_18%","subTotal_18%","CGSTAmount_14%","SGSTAmount_14%","IGSTAmount_28%","subTotal_28%","subTotal_0%","tcsAmount"]
        for item in list_of_slabs:
            if final_prediction[item]["text"] == "":
                #print(item)
                final_prediction[item]["text"] = 0
        return final_prediction
    except:
        print(traceback.print_exc(),
              "add_zero_to_empty_tax_slabs Exception")
        return final_prediction_copy
@putil.timing
def verify_total_amount_fields(final_prediction,DF,docMetaData):
    
    final_prediction_copy = copy.deepcopy(final_prediction)
    try:
        from calculateAmountFields import check_if_cgst_v1
        
       
        cgst_present = check_if_cgst_v1(DF, final_prediction, docMetaData)
        df_amt = DF[((DF["extracted_amount"] > 0) | (((DF["potential_amount"] > 0)) & (DF["val"] > 0)))]
        df_amt['new_amount'] = df_amt.apply(lambda row: row['extracted_amount'] if row['potential_amount'] == 0 else row['val'], axis=1)
        df_amt = df_amt.sort_values(by='new_amount', ascending=False)
        df_amt = df_amt.reset_index(drop=True)  # Reset index without adding a new column for the old index

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
                

        filtered_df = df_amt.drop(rows_to_drop)

        # Sort the DataFrame again based on 'new_amount' before saving to CSV
        filtered_df = filtered_df.sort_values(by='new_amount', ascending=False)
       
        # filtered_df.to_csv("test1.csv")
        filter_df = filtered_df.head(5)
        # filter_df.to_csv("test2.csv")
        # filter_df = DF[(DF["extracted_amount"] > 0)]
        if final_prediction["subTotal_0%"]["text"] !="":
            subtotal_zero = float(final_prediction["subTotal_0%"]["text"])
        else:
            subtotal_zero = 0
        if final_prediction["subTotal_5%"]["text"] !="":
            subtotal_5 = float(final_prediction["subTotal_5%"]["text"])
        else:
            subtotal_5 = 0
        if final_prediction["subTotal_12%"]["text"] != "":
            subtotal_12 = float(final_prediction["subTotal_12%"]["text"])
        else:
            subtotal_12 = 0
        if final_prediction["subTotal_18%"]["text"] != "":
            subtotal_18 = float(final_prediction["subTotal_18%"]["text"])
        else:
            subtotal_18 = 0
        if final_prediction["subTotal_28%"]["text"] != "":
            subtotal_28 = float(final_prediction["subTotal_28%"]["text"])
        else:
            subtotal_28 = 0
        calculated_subtotal = (subtotal_zero + subtotal_5 + subtotal_12 + subtotal_18 + subtotal_28)

        if final_prediction["CGSTAmount_2.5%"]["text"] != "":
            cgst_amount_25 = float(final_prediction["CGSTAmount_2.5%"]["text"])
        else:
            cgst_amount_25 = 0
        if final_prediction["CGSTAmount_6%"]["text"] != "":
            cgst_amount_6 = float(final_prediction["CGSTAmount_6%"]["text"])
        else:
            cgst_amount_6 = 0
        if final_prediction["CGSTAmount_9%"]["text"] != "":
            cgst_amount_9 = float(final_prediction["CGSTAmount_9%"]["text"])
        else:
            cgst_amount_9 = 0
        if final_prediction["CGSTAmount_14%"]["text"] != "":
            cgst_amount_14 = float(final_prediction["CGSTAmount_14%"]["text"])
        else:
            cgst_amount_14 = 0
        calculated_cgst_total = (cgst_amount_25 + cgst_amount_6 + cgst_amount_9 + cgst_amount_14)

        if final_prediction["SGSTAmount_2.5%"]["text"] != "":
            sgst_amount_25 = float(final_prediction["SGSTAmount_2.5%"]["text"])
        else:
            sgst_amount_25 = 0
        if final_prediction["SGSTAmount_6%"]["text"] != "":
            sgst_amount_6 = float(final_prediction["SGSTAmount_6%"]["text"])
        else:
            sgst_amount_6 = 0
        if final_prediction["SGSTAmount_9%"]["text"] != "":
            sgst_amount_9 = float(final_prediction["SGSTAmount_9%"]["text"])
        else:
            sgst_amount_9 = 0
        if final_prediction["SGSTAmount_14%"]["text"] != "":
            sgst_amount_14 = float(final_prediction["SGSTAmount_14%"]["text"])
        else:
            sgst_amount_14 = 0
        calculated_sgst_total = (sgst_amount_25 + sgst_amount_6 + sgst_amount_9 + sgst_amount_14)

        if final_prediction["IGSTAmount_5%"]["text"] != "":
            igst_amount_5 = float(final_prediction["IGSTAmount_5%"]["text"])
        else:
            igst_amount_5 = 0
        if final_prediction["IGSTAmount_12%"]["text"] != "":
            igst_amount_12 = float(final_prediction["IGSTAmount_12%"]["text"])
        else:
            igst_amount_12 = 0
        if final_prediction["IGSTAmount_18%"]["text"] != "":
            igst_amount_18 = float(final_prediction["IGSTAmount_18%"]["text"])
        else:
            igst_amount_18 = 0 
        if final_prediction["IGSTAmount_28%"]["text"] != "":
            igst_amount_28 = float(final_prediction["IGSTAmount_28%"]["text"])
        else:
            igst_amount_28 = 0
        calculated_igst_amount = (igst_amount_5 + igst_amount_12 + igst_amount_18 + igst_amount_28)
               
        if (final_prediction.get("tcsAmount")!= None) and (final_prediction["tcsAmount"]["text"] !=""):
            tcs_amount = float(final_prediction["tcsAmount"]["text"])
        else:
            tcs_amount = 0
        if final_prediction.get("additionalCessAmount") != None and (final_prediction["additionalCessAmount"]["text"] !=""):
            additional_cess_amount = float(final_prediction["additionalCessAmount"]["text"])
        else:
            additional_cess_amount = 0
        if final_prediction["totalAmount"]["text"] != "":
            total_amount = float(final_prediction["totalAmount"]["text"])
        else:
            total_amount = 0
        
        if final_prediction.get("CessAmount") != None and (final_prediction["CessAmount"]["text"] != ""):
            cess_amount = float(final_prediction["CessAmount"]["text"])
        else:
            cess_amount = 0
        
        calculated_total = calculated_subtotal + calculated_cgst_total + calculated_sgst_total + calculated_igst_amount + cess_amount + additional_cess_amount + tcs_amount

        
        if not math.isclose(total_amount,calculated_total,abs_tol=0.2):
            print("Correcting total_amount Amount Fields",calculated_total)
            found = check_amount_field_presence_in_invoice(filter_df,calculated_total,0.8)
            print("found value", found)
            if found == True and calculated_total != 0:
                print("Found In Invoice",calculated_total)
                final_prediction["totalAmount"]["text"] = round(calculated_total,2)
                final_prediction["totalAmount"]["final_confidence_score"] = 1
        
        return final_prediction
    except:
        print(traceback.print_exc(),
              "verify total amount field Exception")
        return final_prediction_copy
@putil.timing
def get_zero_percentage_subTotal(df,final_prediction):
    final_prediction["subTotal_0%"]["text"] = 0
    final_prediction_copy = copy.deepcopy(final_prediction)
    try:
        
        if final_prediction_copy["totalAmount"]["text"] !="":
            total_amount = float(final_prediction_copy["totalAmount"]["text"])
        else:
            #return final_prediction
            total_amount = 0
        if final_prediction_copy["subTotal_5%"]["text"] !="":
            subtotal_5 = float(final_prediction_copy["subTotal_5%"]["text"])
        else:
            subtotal_5 = 0
        if final_prediction_copy["subTotal_12%"]["text"] != "":
            subtotal_12 = float(final_prediction_copy["subTotal_12%"]["text"])
        else:
            subtotal_12 = 0
        if final_prediction_copy["subTotal_18%"]["text"] != "":
            subtotal_18 = float(final_prediction_copy["subTotal_18%"]["text"])
        else:
            subtotal_18 = 0
        if final_prediction_copy["subTotal_28%"]["text"] != "":
            subtotal_28 = float(final_prediction_copy["subTotal_28%"]["text"])
        else:
            subtotal_28 = 0
        if final_prediction_copy["CGSTAmount_2.5%"]["text"] != "":
            cgst_25 = float(final_prediction_copy["CGSTAmount_2.5%"]["text"])
        else:
            cgst_25 = 0
        if final_prediction_copy["SGSTAmount_2.5%"]["text"] != "":
            sgst_25 = float(final_prediction_copy["SGSTAmount_2.5%"]["text"])
        else:
            sgst_25 = 0
        if final_prediction_copy["IGSTAmount_5%"]["text"] != "":
            igst_5 = float(final_prediction_copy["IGSTAmount_5%"]["text"])
        else:
            igst_5 = 0
        if final_prediction_copy["CGSTAmount_6%"]["text"] != "":
            cgst_6 = float(final_prediction_copy["CGSTAmount_6%"]["text"])
        else:
            cgst_6 = 0
        if final_prediction_copy["SGSTAmount_6%"]["text"] != "":
            sgst_6 = float(final_prediction_copy["SGSTAmount_6%"]["text"])
        else:
            sgst_6 = 0
        if final_prediction_copy["IGSTAmount_12%"]["text"] != "":
            igst_12 = float(final_prediction_copy["IGSTAmount_12%"]["text"])
        else:
            igst_12 = 0
        if final_prediction_copy["CGSTAmount_9%"]["text"] != "":
            cgst_9 = float(final_prediction_copy["CGSTAmount_9%"]["text"])
        else:
            cgst_9 = 0
        if final_prediction_copy["SGSTAmount_9%"]["text"] != "":
            sgst_9 = float(final_prediction_copy["SGSTAmount_9%"]["text"])
        else:
            sgst_9 = 0
        if final_prediction_copy["IGSTAmount_18%"]["text"] != "":
            igst_18 = float(final_prediction_copy["IGSTAmount_18%"]["text"])
        else:
            igst_18 = 0
        if final_prediction_copy["CGSTAmount_14%"]["text"] != "":
            cgst_14 = float(final_prediction_copy["CGSTAmount_14%"]["text"])
        else:
            cgst_14 = 0
        if final_prediction_copy["SGSTAmount_14%"]["text"] != "":
            sgst_14 = float(final_prediction_copy["SGSTAmount_14%"]["text"])
        else:
            sgst_14 = 0
        if final_prediction_copy["IGSTAmount_28%"]["text"] != "":
            igst_28 = float(final_prediction_copy["IGSTAmount_28%"]["text"])
        else:
            igst_28 = 0
        if (final_prediction_copy.get("CessAmount")!= None) and (final_prediction_copy["CessAmount"]["text"] != ""):
            cess_amt = float(final_prediction_copy["CessAmount"]["text"])
        else:
            cess_amt = 0
        if (final_prediction_copy.get("additionalCessAmount")!= None) and (final_prediction_copy["additionalCessAmount"]["text"] != ""):
            add_cess_amt = float(final_prediction_copy["additionalCessAmount"]["text"])
        else:
            add_cess_amt = 0
        if (final_prediction_copy.get("tcsAmount")!= None) and final_prediction_copy["tcsAmount"]["text"] != "":
            tcs_amt  = float(final_prediction_copy["tcsAmount"]["text"])
        else:
            tcs_amt = 0
        def check_df_for_subtotal(x,metadata_subtotal):
            try:
                metadata_subtotal = float(metadata_subtotal)
                text_x = float(x)
                if math.isclose(metadata_subtotal, text_x, abs_tol=0.5):
                    return True
                return False
            except:
                return False
        
        def get_subtotal_token(df,metadata_subtotal:str):
            try:
                df["is_subtotal"] = df["extracted_amount"].apply(lambda x: check_df_for_subtotal(x, metadata_subtotal))
                fdf = df[df["is_subtotal"]==True]
                print(fdf.shape)
                if len(fdf) > 0:
                    fdf = fdf.reset_index(drop = True) 
                    #print("s",fdf["token_id"][0])
                    return fdf["token_id"][0]
                else:
                    return
            except:
                print("find po_pattern exception :",traceback.print_exc())
                return       
        
        if (total_amount !=0) and (subtotal_5 != 0 or subtotal_12 != 0 or subtotal_18 != 0 or subtotal_28!=0):
            subtracted_amount = round(total_amount - subtotal_5 - cgst_25 - sgst_25 - igst_5- subtotal_12 - cgst_6 - sgst_6 - igst_12- subtotal_18 - cgst_9 - sgst_9 - igst_18- subtotal_28 - cgst_14 - sgst_14 - igst_28,2)
            if cess_amt != 0 and cess_amt != "":
                subtracted_amount -= cess_amt   
            if add_cess_amt != 0 and add_cess_amt != "":
                subtracted_amount -= add_cess_amt
            if tcs_amt != 0 and tcs_amt != "":
                subtracted_amount -= tcs_amt
            if subtracted_amount > 1:
                # 28 March 2024 Added token for subtotal_0%
                token = get_subtotal_token(df,subtracted_amount)
                print("Token is:", token)
                if token:
                    token_df = df[df["token_id"] == token]
                    print("token_df :",token_df.shape)
                    for _, row in token_df.iterrows():                        
                        final_prediction_copy["subTotal_0%"]["text"] = round(subtracted_amount,2)
                # final_prediction_copy["subTotal_0%"]["prob_subTotal_5%"] = 1
                # final_prediction_copy["subTotal_0%"]["final_confidence_score"] = 1
            
        if (total_amount !=0) and (subtotal_5 == 0) and subtotal_12 ==0  and subtotal_18 == 0 and subtotal_28 ==0:
            subtracted_amount = total_amount
            if cess_amt != 0 and cess_amt != "":
                subtracted_amount -= cess_amt   
            if add_cess_amt != 0 and add_cess_amt != "":
                subtracted_amount -= add_cess_amt
            if tcs_amt != 0 and tcs_amt != "":
                subtracted_amount -= tcs_amt
            if subtracted_amount > 1:
                # 28 March 2024 Added token for subtotal_0%
                token = get_subtotal_token(df,subtracted_amount)
                print("Token is:", token)
                if token:
                    token_df = df[df["token_id"] == token]
                    print("token_df :",token_df.shape)
                    for _, row in token_df.iterrows():                
                        final_prediction_copy["subTotal_0%"]["text"] = round(subtracted_amount,2)
            
        return final_prediction_copy
    except Exception as e:
        print("Exception occured in get_zero_percentage_subTotal", e)
        return final_prediction

# 25 April 2023 calculate the total of slab amounts  
def calculate_tax(tax_list,cgst_present):
    try:
        calcTotalGST_slab = 0
        for item in tax_list:
            if cgst_present == 1:
                calcTotalGST_slab = calcTotalGST_slab + (2*float(item["cgst_amount"]))
            elif cgst_present == 0:
                calcTotalGST_slab = calcTotalGST_slab + float(item["igst_amount"])
        return calcTotalGST_slab
    except:
        print(traceback.print_exc(),
              "calculate_tax Exception")
        return 0
def get_totalGST_dict(tax_list,cgst_present):
    try:
        if cgst_present == 1:
            l = []
            tax_slab_dict={}
            
            for item in tax_list:
                if item["cgst_percentage"] in l:
                    #d[cnt] = d[cnt]
                    cnt = item["cgst_percentage"]
                    tax_slab_dict[cnt]["taxable"] = tax_slab_dict[cnt]["taxable"] + item["taxable"]
                    tax_slab_dict[cnt]["cgst_amount"] = tax_slab_dict[cnt]["cgst_amount"] + item["cgst_amount"]
                    tax_slab_dict[cnt]["sgst_amount"] = tax_slab_dict[cnt]["sgst_amount"] + item["sgst_amount"]
                else:
                    cnt = item["cgst_percentage"]
                    l.append(item["cgst_percentage"])
                    tax_slab_dict[cnt] = {}
                    tax_slab_dict[cnt]["taxable"] = item["taxable"]
                    tax_slab_dict[cnt]["cgst_percentage"] = item["cgst_percentage"]
                    tax_slab_dict[cnt]["sgst_percentage"] = item["sgst_percentage"]
                    tax_slab_dict[cnt]["cgst_amount"] = item["cgst_amount"]
                    tax_slab_dict[cnt]["sgst_amount"] = item["sgst_amount"]
        elif cgst_present == 0:
            l = []
            tax_slab_dict={}
            for item in tax_list:
                if item["igst_percentage"] in l:
                    #d[cnt] = d[cnt]
                    cnt = item["igst_percentage"]
                    tax_slab_dict[cnt]["taxable"] = tax_slab_dict[cnt]["taxable"] + item["taxable"]
                    tax_slab_dict[cnt]["igst_amount"] = tax_slab_dict[cnt]["igst_amount"] + item["igst_amount"]
                else:
                    cnt = item["igst_percentage"]
                    l.append(item["igst_percentage"])
                    tax_slab_dict[cnt] = {}
                    tax_slab_dict[cnt]["taxable"] = item["taxable"]
                    tax_slab_dict[cnt]["igst_percentage"] = item["igst_percentage"]
                    tax_slab_dict[cnt]["igst_amount"] = item["igst_amount"]
        if cgst_present != -1:
            print("Tax slabs sub total",tax_slab_dict)
        return tax_slab_dict
    except:
        print(traceback.print_exc(),
              "get_totalGST_dict Exception")
        return {}
def validation_for_tax_slab_v2(df,tax_list,cgst_present):
    try:
        from tax_slab_analysis import get_extract_amount_for_tax_slab
        amount_df = get_extract_amount_for_tax_slab(df)
        amount_df = amount_df[amount_df["extract_amount_for_tax_slab"]>0]
        #amount_df.to_csv(r"C:\Users\Admin\Desktop\filt.csv")
        result = 0
        if len(tax_list) == 0:
            result = -1
            return result
        if len(tax_list)>0:
            calcTotalGST_dict = get_totalGST_dict(tax_list,cgst_present)
        amt_list = []
        for key, value in calcTotalGST_dict.items():
            if cgst_present == 1:
                amt = value["cgst_amount"]
            else:
                amt = value["igst_amount"]
            amt_list.append(amt)
        amount_df_list = list(map(float,amount_df["extract_amount_for_tax_slab"].unique()))
        matched_item = []
        for item1 in amt_list:
            for item2 in amount_df_list:
                if math.isclose(item1,item2,abs_tol=0.2):
                    matched_item.append(item1)
                    break
        if len(matched_item) == len(amt_list):
            result = 1
        else:
            diff = abs(len(amt_list) - len(matched_item))
            result = 1 - (diff/10)
        print("Result",result)
        return result
    except:
        print(traceback.print_exc(),
              "validation_for_tax_slab_v2 Exception")
        return 0

# 25 April 2023 calculate the total of slab amounts  
def calculate_tax_v3(tax_list,cgst_present):
    try:
        calcTotalGST_slab = 0
        calcsubtotal_slab = 0
        for item in tax_list:
            if cgst_present == 1:
                calcTotalGST_slab = calcTotalGST_slab + (2*float(item["cgst_amount"]))
            elif cgst_present == 0:
                calcTotalGST_slab = calcTotalGST_slab + float(item["igst_amount"])
            calcsubtotal_slab+= item["taxable"]
        #print(f"Subtotal slab:{calcsubtotal_slab}, calculate_GST_slab {calcTotalGST_slab}")
        return calcTotalGST_slab,calcsubtotal_slab
    except:
        print(traceback.print_exc(),
              "calculate_tax_v3 Exception")
        return 0

def check_subtotal_in_invoice(df,conv_subtotal_0):
    try:
        filter_df = df[df["extracted_amount"]>0]
        txt_amounts = []
        for text in filter_df["text"]:
            try:
                text = str(text).replace(",","")
                amt = float(text)
                txt_amounts.append(amt)
            except:
                pass
        txt_amounts = list(set(txt_amounts))
        print(txt_amounts)
        subtotal_0_present = 0 
        for item in txt_amounts:
            if math.isclose(item,conv_subtotal_0,abs_tol = 1):
                print("Found subtotal_0 in Invoice")
                subtotal_0_present = 1
                break
        if math.isclose(conv_subtotal_0,0,abs_tol = 1):
            print("No amount for Sub-total 0 percent")
            subtotal_0_present = 1
        return subtotal_0_present
    except:
        print(traceback.print_exc(),
              "check_subtotal_in_invoice Exception")
        return 0

def reducing_confidence_additional_tax_present(final_prediction, docMetaData):
    final_prediction_copy = copy.deepcopy(final_prediction)
    try:
        
        if (final_prediction.get("tcsAmount")!= None) and (final_prediction["tcsAmount"]["text"] !="" and final_prediction["tcsAmount"]["text"] != None):
            conv_tcsAmount = float(final_prediction["tcsAmount"]["text"])
        else:
            conv_tcsAmount = 0
        if (final_prediction.get("CessAmount")!= None) and (final_prediction["CessAmount"]["text"] !="" and final_prediction["CessAmount"]["text"] != None):
            conv_CessAmount = float(final_prediction["CessAmount"]["text"])
        else:
            conv_CessAmount = 0
        if (final_prediction.get("additionalCessAmount")!= None) and (final_prediction["additionalCessAmount"]["text"] !="" and final_prediction["additionalCessAmount"]["text"] != None):
            conv_additionalCessAmount = float(final_prediction["additionalCessAmount"]["text"])
        else:
            conv_additionalCessAmount = 0
        if (final_prediction.get("discountAmount")!= None) and (final_prediction["discountAmount"]["text"] !="" and final_prediction["discountAmount"]["text"] != None):
            conv_discountAmount = float(final_prediction["discountAmount"]["text"])
        else:
            conv_discountAmount = 0
            
        if docMetaData.get("result")!= None and docMetaData.get("result").get("document") != None and docMetaData.get("result").get("document").get("docType") == "Invoice":
            list_of_slabs = ["CGSTAmount_2.5%","SGSTAmount_2.5%","IGSTAmount_5%","subTotal_5%","CGSTAmount_6%","SGSTAmount_6%","IGSTAmount_12%","subTotal_12%","CGSTAmount_9%","SGSTAmount_9%","IGSTAmount_18%","subTotal_18%","CGSTAmount_14%","SGSTAmount_14%","IGSTAmount_28%","subTotal_28%","subTotal_0%","tcsAmount","CessAmount","additionalCessAmount","discountAmount","totalAmount"]
            # calc_totalAmount = calc_subtotal_slab + calc_totalGST_slab - conv_discountAmount + conv_CessAmount + conv_additionalCessAmount + conv_tcsAmount
        elif docMetaData.get("result")!= None and docMetaData.get("result").get("document") != None and docMetaData.get("result").get("document").get("docType") == "Discrepancy Note":
            list_of_slabs = ["CGSTAmount_2.5%","SGSTAmount_2.5%","IGSTAmount_5%","subTotal_5%","CGSTAmount_6%","SGSTAmount_6%","IGSTAmount_12%","subTotal_12%","CGSTAmount_9%","SGSTAmount_9%","IGSTAmount_18%","subTotal_18%","CGSTAmount_14%","SGSTAmount_14%","IGSTAmount_28%","subTotal_28%","subTotal_0%","totalAmount","CessAmount","additionalCessAmount"]
            # calc_totalAmount = calc_subtotal_slab + calc_totalGST_slab
        # print("sahil1 reducing_confidence_additional_tax_present", conv_tcsAmount, conv_CessAmount, conv_additionalCessAmount, conv_discountAmount)
        for item in list_of_slabs:
            if conv_tcsAmount > 0 or conv_CessAmount > 0 or conv_additionalCessAmount > 0 or conv_discountAmount > 0:
                print("Additional tax is greater than 0. Keeping the confidence low.", item)
                final_prediction[item]["final_confidence_score"] = 0.45
        return final_prediction        
    except Exception as e:
        print("Exception occured in reducing_confidence_additional_tax_present", str(e))
        return final_prediction_copy
        
# 19 May 2023 Increasing confidence of Amounr fields for STP 
@putil.timing
def modifying_confidence_amount_fields(DF,final_prediction,docMetaData):
    final_prediction_copy = copy.deepcopy(final_prediction)
    try:
        # Amount fields extraction 
        conv_totalAmount = final_prediction.get("totalAmount").get("text",0)
        if conv_totalAmount =="" or conv_totalAmount== None:
            conv_totalAmount = 0
        else:
            conv_totalAmount = float(conv_totalAmount) 
        
        if (final_prediction.get("tcsAmount")!= None) and (final_prediction["tcsAmount"]["text"] !="" and final_prediction["tcsAmount"]["text"] != None):
            conv_tcsAmount = float(final_prediction["tcsAmount"]["text"])
        else:
            conv_tcsAmount = 0
        if (final_prediction.get("CessAmount")!= None) and (final_prediction["CessAmount"]["text"] !="" and final_prediction["CessAmount"]["text"] != None):
            conv_CessAmount = float(final_prediction["CessAmount"]["text"])
        else:
            conv_CessAmount = 0
        if (final_prediction.get("additionalCessAmount")!= None) and (final_prediction["additionalCessAmount"]["text"] !="" and final_prediction["additionalCessAmount"]["text"] != None):
            conv_additionalCessAmount = float(final_prediction["additionalCessAmount"]["text"])
        else:
            conv_additionalCessAmount = 0
        if (final_prediction.get("discountAmount")!= None) and (final_prediction["discountAmount"]["text"] !="" and final_prediction["discountAmount"]["text"] != None):
            conv_discountAmount = float(final_prediction["discountAmount"]["text"])
        else:
            conv_discountAmount = 0
        
        ## Subtotal fields
        conv_subtotal_5 = final_prediction.get("subTotal_5%").get("text",0)
        if conv_subtotal_5 =="" or conv_subtotal_5== None:
            conv_subtotal_5 = 0
        else:
            conv_subtotal_5 = float(conv_subtotal_5)
        conv_subtotal_12 = final_prediction.get("subTotal_12%").get("text",0)
        if conv_subtotal_12 =="" or conv_subtotal_12== None:
            conv_subtotal_12 = 0
        else:
            conv_subtotal_12 = float(conv_subtotal_12)
        conv_subtotal_18 = final_prediction.get("subTotal_18%").get("text",0)
        if conv_subtotal_18 =="" or conv_subtotal_18== None:
            conv_subtotal_18 = 0
        else:
            conv_subtotal_18 = float(conv_subtotal_18)
        conv_subtotal_28 = final_prediction.get("subTotal_28%").get("text",0)
        if conv_subtotal_28 =="" or conv_subtotal_28== None:
            conv_subtotal_28 = 0
        else:
            conv_subtotal_28 = float(conv_subtotal_28)
        conv_subtotal_0 = final_prediction.get("subTotal_0%").get("text",0)
        if conv_subtotal_0 =="" or conv_subtotal_0== None:
            conv_subtotal_0 = 0
        else:
            conv_subtotal_0 = float(conv_subtotal_0)
        
        calc_subtotal_slab = conv_subtotal_5 + conv_subtotal_12 + conv_subtotal_18 + conv_subtotal_28 + conv_subtotal_0
        
        ## CGST/SGST fields
        conv_CGSTAmount_25 = final_prediction.get("CGSTAmount_2.5%").get("text",0)
        if conv_CGSTAmount_25 =="" or conv_CGSTAmount_25== None:
            conv_CGSTAmount_25 = 0
        else:
            conv_CGSTAmount_25 = float(conv_CGSTAmount_25)
        conv_SGSTAmount_25 = final_prediction.get("SGSTAmount_2.5%").get("text",0)
        if conv_SGSTAmount_25 =="" or conv_SGSTAmount_25== None:
            conv_SGSTAmount_25 = 0
        else:
            conv_SGSTAmount_25 = float(conv_SGSTAmount_25)
        conv_CGSTAmount_6 = final_prediction.get("CGSTAmount_6%").get("text",0)
        if conv_CGSTAmount_6 =="" or conv_CGSTAmount_6== None:
            conv_CGSTAmount_6 = 0
        else:
            conv_CGSTAmount_6 = float(conv_CGSTAmount_6)
        conv_SGSTAmount_6 = final_prediction.get("SGSTAmount_6%").get("text",0)
        if conv_SGSTAmount_6 =="" or conv_SGSTAmount_6== None:
            conv_SGSTAmount_6 = 0
        else:
            conv_SGSTAmount_6 = float(conv_SGSTAmount_6)
        conv_CGSTAmount_9 = final_prediction.get("CGSTAmount_9%").get("text",0)
        if conv_CGSTAmount_9 =="" or conv_CGSTAmount_9== None:
            conv_CGSTAmount_9 = 0
        else:
            conv_CGSTAmount_9 = float(conv_CGSTAmount_9)
        conv_SGSTAmount_9 = final_prediction.get("SGSTAmount_9%").get("text",0)
        if conv_SGSTAmount_9 =="" or conv_SGSTAmount_9== None:
            conv_SGSTAmount_9 = 0
        else:
            conv_SGSTAmount_9 = float(conv_SGSTAmount_9)
        conv_CGSTAmount_14 = final_prediction.get("CGSTAmount_14%").get("text",0)
        if conv_CGSTAmount_14 =="" or conv_CGSTAmount_14== None:
            conv_CGSTAmount_14 = 0
        else:
            conv_CGSTAmount_14 = float(conv_CGSTAmount_14)
        conv_SGSTAmount_14 = final_prediction.get("SGSTAmount_14%").get("text",0)
        if conv_SGSTAmount_14 =="" or conv_SGSTAmount_14== None:
            conv_SGSTAmount_14 = 0
        else:
            conv_SGSTAmount_14 = float(conv_SGSTAmount_14)   
        
        
        calc_CGSTAmount = conv_CGSTAmount_25 + conv_SGSTAmount_25 + conv_CGSTAmount_6 + conv_SGSTAmount_6 + conv_CGSTAmount_9 + conv_SGSTAmount_9 + conv_CGSTAmount_14 + conv_SGSTAmount_14
        
        
        ## IGST fields
        conv_IGSTAmount_5 = final_prediction.get("IGSTAmount_5%").get("text",0)
        if conv_IGSTAmount_5 =="" or conv_IGSTAmount_5== None:
            conv_IGSTAmount_5 = 0
        else:
            conv_IGSTAmount_5 = float(conv_IGSTAmount_5)
        conv_IGSTAmount_12 = final_prediction.get("IGSTAmount_12%").get("text",0)
        if conv_IGSTAmount_12 =="" or conv_IGSTAmount_12== None:
            conv_IGSTAmount_12 = 0
        else:
            conv_IGSTAmount_12 = float(conv_IGSTAmount_12)
        conv_IGSTAmount_18 = final_prediction.get("IGSTAmount_18%").get("text",0)
        if conv_IGSTAmount_18 =="" or conv_IGSTAmount_18== None:
            conv_IGSTAmount_18 = 0
        else:
            conv_IGSTAmount_18 = float(conv_IGSTAmount_18)
        conv_IGSTAmount_28 = final_prediction.get("IGSTAmount_28%").get("text",0)
        if conv_IGSTAmount_28 =="" or conv_IGSTAmount_28== None:
            conv_IGSTAmount_28 = 0
        else:
            conv_IGSTAmount_28 = float(conv_IGSTAmount_28)

        calc_IGSTAmount = conv_IGSTAmount_5 + conv_IGSTAmount_12 + conv_IGSTAmount_18 + conv_IGSTAmount_28

        calc_totalGST_slab = calc_CGSTAmount + calc_IGSTAmount

        ## check subtotal 0 % in Invoice
        found_subtotal = check_subtotal_in_invoice(DF,conv_subtotal_0) 
        if found_subtotal == 0:
            return final_prediction
        
        ## Checking the amount field with GSTIN extracted
        from calculateAmountFields import check_if_cgst_v1
        cgst_present = check_if_cgst_v1(DF,final_prediction_copy,docMetaData)
        print("CGST Present",cgst_present)
        if cgst_present == -1:
            print("Vendor GSTIN or Shipping GSTIN Issues")
            if docMetaData.get("result") != None and docMetaData.get("result").get("document").get("subDocType") != None:
                if docMetaData.get("result") != None and docMetaData.get("result").get("document").get("subDocType") == "F&V type":
                    print("Found F&B. Moving to Increase slab confidence")
                else:
                    return final_prediction
            else:
                return final_prediction
        if cgst_present == 1 and calc_IGSTAmount != 0:
            print("Slabs and GSTIN does not match. Not changing the conidence.")
            return final_prediction
        if cgst_present == 0 and calc_CGSTAmount != 0:
            print("Slabs and GSTIN does not match. Not changing the conidence.")
            return final_prediction
        
        ## Logic for validation of amount fields
        # netAmount = calcsubtotal_slab + calcTotalGST_slab  - conv_discountAmount + conv_CessAmount + conv_additionalCessAmount + conv_tcsAmount
        calc_totalAmount = calc_subtotal_slab + calc_totalGST_slab - conv_discountAmount + conv_CessAmount + conv_additionalCessAmount + conv_tcsAmount    
        print("Extracted Total Amount",conv_totalAmount)
        print("Calculated Total Amount",calc_totalAmount)

        if docMetaData.get("result")!= None and docMetaData.get("result").get("document") != None and docMetaData.get("result").get("document").get("docType") == "Invoice":
            list_of_slabs = ["CGSTAmount_2.5%","SGSTAmount_2.5%","IGSTAmount_5%","subTotal_5%","CGSTAmount_6%","SGSTAmount_6%","IGSTAmount_12%","subTotal_12%","CGSTAmount_9%","SGSTAmount_9%","IGSTAmount_18%","subTotal_18%","CGSTAmount_14%","SGSTAmount_14%","IGSTAmount_28%","subTotal_28%","subTotal_0%","tcsAmount","CessAmount","additionalCessAmount","discountAmount","totalAmount"]
            # calc_totalAmount = calc_subtotal_slab + calc_totalGST_slab - conv_discountAmount + conv_CessAmount + conv_additionalCessAmount + conv_tcsAmount
        elif docMetaData.get("result")!= None and docMetaData.get("result").get("document") != None and docMetaData.get("result").get("document").get("docType") == "Discrepancy Note":
            list_of_slabs = ["CGSTAmount_2.5%","SGSTAmount_2.5%","IGSTAmount_5%","subTotal_5%","CGSTAmount_6%","SGSTAmount_6%","IGSTAmount_12%","subTotal_12%","CGSTAmount_9%","SGSTAmount_9%","IGSTAmount_18%","subTotal_18%","CGSTAmount_14%","SGSTAmount_14%","IGSTAmount_28%","subTotal_28%","subTotal_0%","totalAmount","CessAmount","additionalCessAmount"]
            # calc_totalAmount = calc_subtotal_slab + calc_totalGST_slab
        
        if math.isclose(conv_totalAmount,calc_totalAmount,abs_tol = 1) and conv_totalAmount != 0:
            print("Exact match for amount fields. Increasing confidence for all fields")
            #list_of_slabs = ["CGSTAmount_2.5%","SGSTAmount_2.5%","IGSTAmount_5%","subTotal_5%","CGSTAmount_6%","SGSTAmount_6%","IGSTAmount_12%","subTotal_12%","CGSTAmount_9%","SGSTAmount_9%","IGSTAmount_18%","subTotal_18%","CGSTAmount_14%","SGSTAmount_14%","IGSTAmount_28%","subTotal_28%","subTotal_0%","tcsAmount","CessAmount","additionalCessAmount","discountAmount","totalAmount"]
            for item in list_of_slabs:
                if docMetaData.get("result") != None and docMetaData.get("result").get("document").get("subDocType") != None:
                    if docMetaData.get("result") != None and docMetaData.get("result").get("document").get("subDocType") == "F&V type":
                        final_prediction[item]["final_confidence_score"] = 0.95
                    else:
                        if conv_subtotal_0 > 0:
                            print("Subtotal_0 is greater than 0. Keeping the confidence low.")
                            final_prediction[item]["final_confidence_score"] = 0.38
                        else:     
                            final_prediction[item]["final_confidence_score"] = 1
                else:
                    if conv_subtotal_0 > 0:
                        print("Subtotal_0 is greater than 0. Keeping the confidence low.")
                        final_prediction[item]["final_confidence_score"] = 0.38
                    else:   
                        final_prediction[item]["final_confidence_score"] = 1
        else:
            print("Amount fields does not match with slab amounts. Not changing the conidence.")
        return final_prediction
    except:
        print(traceback.print_exc(),
              "modifying_confidence_amount_fields Exception")
        return final_prediction_copy
    
# 5 May 2023 Applying validations based on total_amount
def validation_for_tax_slab_v3(df,tax_list,flds,cgst_present):
    try:
        result = 0
        if len(tax_list) == 0:
            return -1
        elif len(tax_list)>0:
            calcTotalGST_slab,calcsubtotal_slab = calculate_tax_v3(tax_list,cgst_present)

        if (flds.get("CessAmount")!= None) and (flds["CessAmount"]["text"] !="" and flds["CessAmount"]["text"] != None):
            conv_CessAmount = float(flds["CessAmount"]["text"])
        else:
            conv_CessAmount = 0
        if (flds.get("additionalCessAmount")!= None) and (flds["additionalCessAmount"]["text"] !="" and flds["additionalCessAmount"]["text"] != None):
            conv_additionalCessAmount = float(flds["additionalCessAmount"]["text"])
        else:
            conv_additionalCessAmount = 0
        if (flds.get("tcsAmount")!= None) and (flds["tcsAmount"]["text"] !="" and flds["tcsAmount"]["text"] != None):
            conv_tcsAmount = float(flds["tcsAmount"]["text"])
        else:
            conv_tcsAmount = 0
        if (flds.get("discountAmount")!= None) and (flds["discountAmount"]["text"] !="" and flds["discountAmount"]["text"] != None):
            conv_discountAmount = float(flds["discountAmount"]["text"])
        else:
            conv_discountAmount = 0

        conv_totalAmount = flds.get("totalAmount").get("text",0)
        if conv_totalAmount =="" or conv_totalAmount== None:
            conv_totalAmount = 0
        else:
            conv_totalAmount = float(conv_totalAmount)

        # netAmount - subTotal_slab + convDiscAmt - calcTotalGST_slab - calcAddlTax
        calc_subtotal_0 = round((conv_totalAmount  - calcsubtotal_slab - calcTotalGST_slab + conv_discountAmount - conv_CessAmount - conv_additionalCessAmount - conv_tcsAmount ),2)
        #subTotal_0%
        if calc_subtotal_0 < -1:
            print("sub total is less than 0")
            return -1
        else:
            calc_subtotal_0 = abs(calc_subtotal_0)

        netAmount = calcsubtotal_slab + calcTotalGST_slab + calc_subtotal_0  - conv_discountAmount + conv_CessAmount + conv_additionalCessAmount + conv_tcsAmount
        print("Subtotal zero percent is: ",calc_subtotal_0)
        print("Net Amount",netAmount)
        #total_gst_amount = flds.get("totalGSTAmount").get("text",0)
        #print("total GST Amount :",total_gst_amount)
        filter_df = df[df["extracted_amount"]>0]
        txt_amounts = []
        for text in filter_df["text"]:
            try:
                text = str(text).replace(",","")
                amt = float(text)
                txt_amounts.append(amt)
            except:
                pass
        txt_amounts = list(set(txt_amounts))
        print(txt_amounts)
        subtotal_0_present = 0 
        for item in txt_amounts:
            if math.isclose(item,calc_subtotal_0,abs_tol = 1):
                print("Found subtotal_0 in Invoice")
                subtotal_0_present = 1
                break
        if math.isclose(calc_subtotal_0,0,abs_tol = 1):
            subtotal_0_present = 1
        if subtotal_0_present == 1:    
            txt_amounts = sorted(txt_amounts, reverse=True)[:3]    
            for item in txt_amounts:
                if math.isclose(item,netAmount,abs_tol=1):
                    return 1
        return result
    except:
        print(traceback.print_exc(),
              "validation_for_tax_slab_v3 Exception")
        return -1


# 25 April 2023 Added validations for tax-slab calculation    
def validation_for_tax_slab(tax_list,flds,cgst_present):
    try:
        result = 0
        if len(tax_list) == 0:
            result = -1
            return result
        if len(tax_list)>0:
            calcTotalGST_slab = calculate_tax(tax_list,cgst_present)
        total_gst_amount = flds.get("totalGSTAmount").get("text",0)
        #print("total GST Amount :",total_gst_amount)
        if flds["totalGSTAmount"]["text"] !="":
            total_gst_amount = float(flds["totalGSTAmount"]["text"])
        else:
            total_gst_amount = 0
        if math.isclose(total_gst_amount,calcTotalGST_slab,abs_tol= 0.5):
            result = 1
        return result
    except:
        print(traceback.print_exc(),
              "validation_for_tax_slab Exception")
        return 0


# 25 April 2023 Added to check the closest tax-slab from 3 different functions and returning the closest one    
def compare_close_gst_slab(tax_list_from_gst_table,tax_list_before_LI,tax_list_after_LI,tax_list_second_approach,final_prediction,cgst_present):
    try:
        total_gst_amount = final_prediction.get("totalGSTAmount").get("text",0)
        if final_prediction["totalGSTAmount"]["text"] !="":
            total_gst_amount = float(final_prediction["totalGSTAmount"]["text"])
        else:
            total_gst_amount = 0
        #print(total_gst_amount)
        if len(tax_list_from_gst_table)>0:
            calcTotalGST_slab_gst_table = calculate_tax(tax_list_from_gst_table,cgst_present)
        else:
            calcTotalGST_slab_gst_table = 0

        if len(tax_list_before_LI)>0:
            calcTotalGST_slab_before = calculate_tax(tax_list_before_LI,cgst_present)
        else:
            calcTotalGST_slab_before = 0
        if len(tax_list_after_LI)>0:
            calcTotalGST_slab_after = calculate_tax(tax_list_after_LI,cgst_present)
        else:
            calcTotalGST_slab_after = 0
        if len(tax_list_second_approach)>0:
            calcTotalGST_slab_second = calculate_tax(tax_list_second_approach,cgst_present)
        else:
            calcTotalGST_slab_second = 0
        #print(calcTotalGST_slab_second)
        if calcTotalGST_slab_second == 0 and calcTotalGST_slab_gst_table == 0:
            a,b = abs(total_gst_amount-calcTotalGST_slab_before),abs(total_gst_amount-calcTotalGST_slab_after)
            smallest = min(a,b)
        elif calcTotalGST_slab_second == 0 and calcTotalGST_slab_gst_table!=0:
            a,b,d = abs(total_gst_amount-calcTotalGST_slab_before),abs(total_gst_amount-calcTotalGST_slab_after),abs(total_gst_amount - calcTotalGST_slab_gst_table)
            smallest = min(a,b,d)
        elif calcTotalGST_slab_second != 0 and calcTotalGST_slab_gst_table==0:
            a,b,c = abs(total_gst_amount-calcTotalGST_slab_before),abs(total_gst_amount-calcTotalGST_slab_after),abs(total_gst_amount-calcTotalGST_slab_second)
            smallest = min(a,b,c)
        else:
            a,b,c,d = abs(total_gst_amount-calcTotalGST_slab_before),abs(total_gst_amount-calcTotalGST_slab_after),abs(total_gst_amount-calcTotalGST_slab_second),abs(total_gst_amount - calcTotalGST_slab_gst_table)
            smallest = min(a,b,c,d)
        # if calcTotalGST_slab_gst_table!=0:
        #     a,b,d = abs(total_gst_amount-calcTotalGST_slab_before),abs(total_gst_amount-calcTotalGST_slab_after),abs(total_gst_amount - calcTotalGST_slab_gst_table)
        #     smallest = min(a,b,d)
        # elif calcTotalGST_slab_second!=0:    
        #     a,b,c,d = abs(total_gst_amount-calcTotalGST_slab_before),abs(total_gst_amount-calcTotalGST_slab_after),abs(total_gst_amount-calcTotalGST_slab_second),abs(total_gst_amount - calcTotalGST_slab_gst_table)
        #     smallest = min(a,b,c,d)
        #print(smallest)
        if smallest == a:
            return tax_list_before_LI
        elif smallest == b:
            return tax_list_after_LI
        elif smallest == d:
            return tax_list_from_gst_table
        elif smallest == c:
            return tax_list_second_approach
        
    except:
        print(traceback.print_exc(),
              "compare_close_gst_slab Exception")
        return []
@putil.timing
def update_final_prediction_for_tax_slab(final_prediction,DF,docMetaData):
    amount_fields = ["subTotal_5%","subTotal_12%","subTotal_18%","subTotal_28%","subTotal_0%","CGSTAmount_2.5%","SGSTAmount_2.5%","IGSTAmount_5%","CGSTAmount_6%","SGSTAmount_6%","IGSTAmount_12%","CGSTAmount_9%","SGSTAmount_9%","IGSTAmount_18%","CGSTAmount_14%","SGSTAmount_14%","IGSTAmount_28%"]
    for field in amount_fields:
        final_prediction.update(biz.add_empty_field(field,0))
    final_prediction_copy = copy.deepcopy(final_prediction)
    try:
        conf = 0
        from tax_slab_analysis import tax_slab_gst_table,tax_slab_before_LI,tax_slab_before_LI_discr,tax_slab_line_item,tax_slab_line_item_discr,tax_slab_line_item_discr_v2,tax_slab_before_LI_discr_combined,tax_slab_line_item_discr_combined,gstslablistdictionary
        from calculateAmountFields import check_if_cgst_v1
        tax_list = []
        cgst_present = check_if_cgst_v1(DF,final_prediction_copy,docMetaData)
        print("cgst/igst present :",cgst_present)
        if docMetaData.get("result") != None and docMetaData.get("result").get("document").get("subDocType") != None:
            if docMetaData.get("result") != None and docMetaData.get("result").get("document").get("subDocType") == "F&V type":
                print("Found F&B. Not calculating slabs")
                return final_prediction

        if docMetaData.get("result") != None and docMetaData.get("result").get("document").get("docType") != None:
            if docMetaData.get("result").get("document").get("docType").lower() == "invoice":
                print("Processing for invoice")
                tax_list_from_gst_table = tax_slab_gst_table(DF,cgst_present)
                print("From gst table tax slab",tax_list_from_gst_table)
                result_slab_identification_from_gst_table = validation_for_tax_slab_v3(DF,tax_list_from_gst_table,final_prediction_copy,cgst_present)
                if result_slab_identification_from_gst_table == 1:
                    tax_list = tax_list_from_gst_table
                    conf = 1
                if len(tax_list) == 0:
                    tax_list_before_LI = tax_slab_before_LI(DF,cgst_present)
                    print("before LI tax slab",tax_list_before_LI)
                    result_slab_identification_before = validation_for_tax_slab_v3(DF,tax_list_before_LI,final_prediction_copy,cgst_present)
                    ## Checking In LI if tax slab is present
                    if result_slab_identification_before == 1:
                        tax_list = tax_list_before_LI
                        conf = 1
                if len(tax_list) == 0:
                    tax_list_after_LI = tax_slab_line_item(DF,cgst_present)
                    print("after LI tax slab:",tax_list_after_LI)
                    result_slab_identification_after = validation_for_tax_slab_v3(DF,tax_list_after_LI,final_prediction_copy,cgst_present)
                    if result_slab_identification_after == 1:
                        tax_list = tax_list_after_LI
                        conf = 1

                if len(tax_list) == 0:
                    print("Reducing confidence since amts does not match")
                    tax_list = tax_slab_gst_table(DF,cgst_present)
                    print("From gst table tax slab",tax_list)
                    if len(tax_list) == 0:
                        tax_list = tax_slab_before_LI(DF,cgst_present)
                        print("before LI tax slab",tax_list)
                    if len(tax_list) == 0:
                        tax_list = tax_slab_line_item(DF,cgst_present)
                        print("after LI tax slab:",tax_list)
                    conf = 0.8
                if len(tax_list) == 0:
                    conf = 0.9
                
        if docMetaData.get("result") != None and docMetaData.get("result").get("document").get("docType") != None:
            if docMetaData.get("result").get("document").get("docType") == "Discrepancy Note":
                ## 27 sept 2023 code for getting total amount only for discr note
                # try:
                #     if final_prediction.get("totalAmount") != None and final_prediction.get("totalAmount").get("text") != None:
                #         total_amount_predicted = final_prediction.get("totalAmount").get("text")
                #         amount_present = None
                #         print("sahil2 Total Amount is zero")
                #         for index, row in DF.iterrows():
                #             # print(row["left_processed_ngbr"], type(row["left_processed_ngbr"]))
                #             if "total" in row["left_processed_ngbr"].lower() and "payable" in row["left_processed_ngbr"].lower() and "including" in row["left_processed_ngbr"].lower() and "tax" in row["left_processed_ngbr"].lower():
                #                 try:
                #                     amount_present = float(row["text"])
                #                     print("Found total", row["text"])
                #                 except:
                #                     pass
                                
                # except Exception as e:
                #     print("Exception in getting total amount for discr note", e)
                ## 20 March 2024 Modifying the LI based on footer keyword, only applicable for Discrepancy Note    
                # filter_df_footer = DF[(DF["line_row_new"]>0) & DF['line_text'].str.contains('number of items', case=False)]["token_id"] 
                # line_item_starts = DF[DF["line_row_new"]>0]["token_id"]
                # if filter_df_footer.min() > line_item_starts.min():
                #     DF.loc[DF['token_id'] >= filter_df_footer.min(), 'line_row_new'] = 0
                #     DF.loc[DF['token_id'] >= filter_df_footer.min(), 'tbl_col_hdr'] = "none"
                #     print("Modified Original df's LI based on foter keyword ")
                            
                print("Processing for discr note") 
                tax_list_before_LI = tax_slab_before_LI_discr(DF,cgst_present)
                print("before LI tax slab",tax_list_before_LI)
                result_slab_identification_before = validation_for_tax_slab_v3(DF,tax_list_before_LI,final_prediction_copy,cgst_present)
                if result_slab_identification_before == 1:
                    tax_list = tax_list_before_LI
                    conf = 1
                if len(tax_list) == 0:
                    tax_list_after_LI = tax_slab_line_item_discr(DF,cgst_present)
                    print("after LI tax slab:",tax_list_after_LI)
                    result_slab_identification_after = validation_for_tax_slab_v3(DF,tax_list_after_LI,final_prediction_copy,cgst_present)
                    if result_slab_identification_after == 1:
                        tax_list = tax_list_after_LI
                        conf = 1
                if len(tax_list) == 0:
                    tax_list_after_LI_v2 = tax_slab_line_item_discr_v2(DF,cgst_present)
                    print("after LI tax slab new template:",tax_list_after_LI_v2)
                    result_slab_identification_after = validation_for_tax_slab_v3(DF,tax_list_after_LI_v2,final_prediction_copy,cgst_present)
                    if result_slab_identification_after == 1:
                        tax_list = tax_list_after_LI_v2
                        conf = 1
                        
                # Checking for combined GSTs  uncommented on 15 March 2024
                if len(tax_list) == 0:
                    tax_list_after_LI_comb = tax_slab_line_item_discr_combined(DF,tax_list,cgst_present)
                    print("after LI tax slab for discr comb",tax_list_after_LI_comb)
                    result_slab_identification_after_comb = validation_for_tax_slab_v3(DF,tax_list_after_LI_comb,final_prediction_copy,cgst_present)
                    if result_slab_identification_after_comb == 1:
                        tax_list = tax_list_after_LI_comb    
                if len(tax_list) == 0:
                    tax_list_before_LI_comb = tax_slab_before_LI_discr_combined(DF,tax_list,cgst_present)
                    print("before LI tax slab for discr combined",tax_list_before_LI_comb)
                    result_slab_identification_before_comb = validation_for_tax_slab_v3(DF,tax_list_before_LI_comb,final_prediction_copy,cgst_present)
                    if result_slab_identification_before_comb == 1:
                        tax_list = tax_list_before_LI_comb
                        
                if len(tax_list) == 0:
                    print("Reducing confidence since amts does not match")
                    tax_list = tax_slab_before_LI_discr(DF,cgst_present)
                    print("before LI tax slab",tax_list)
                    if len(tax_list) == 0:
                        tax_list = tax_slab_line_item_discr_v2(DF,cgst_present)
                        print("after LI tax slab new template:",tax_list)
                    if len(tax_list)==0:
                        tax_list = tax_slab_line_item_discr(DF,cgst_present)
                        print("after LI tax slab:",tax_list)
                    # Checking for combined GST Amount uncommented on 15 March 2024
                    if len(tax_list) == 0:
                        tax_list = tax_slab_line_item_discr_combined(DF,tax_list,cgst_present)
                        print("after LI tax slab for discr comb",tax_list)
                    if len(tax_list) == 0:
                        tax_list = tax_slab_before_LI_discr_combined(DF,tax_list,cgst_present)
                        print("before LI tax slab for discr combined",tax_list)
                    conf = 0.8
                    if len(tax_list) == 0:
                        conf = 0.9
                        
        print("------------------")    
        print("Final tax slab",tax_list)
        print("------------------")
        if cgst_present == 1:
            l = []
            tax_slab_dict={}
            
            for item in tax_list:
                if item["cgst_percentage"] in l:
                    #d[cnt] = d[cnt]
                    cnt = item["cgst_percentage"]
                    tax_slab_dict[cnt]["taxable"] = tax_slab_dict[cnt]["taxable"] + item["taxable"]
                    tax_slab_dict[cnt]["cgst_amount"] = tax_slab_dict[cnt]["cgst_amount"] + item["cgst_amount"]
                    tax_slab_dict[cnt]["sgst_amount"] = tax_slab_dict[cnt]["sgst_amount"] + item["sgst_amount"]
                else:
                    cnt = item["cgst_percentage"]
                    l.append(item["cgst_percentage"])
                    tax_slab_dict[cnt] = {}
                    tax_slab_dict[cnt]["taxable"] = item["taxable"]
                    tax_slab_dict[cnt]["cgst_percentage"] = item["cgst_percentage"]
                    tax_slab_dict[cnt]["sgst_percentage"] = item["sgst_percentage"]
                    tax_slab_dict[cnt]["cgst_amount"] = item["cgst_amount"]
                    tax_slab_dict[cnt]["sgst_amount"] = item["sgst_amount"]
        elif cgst_present == 0:
            l = []
            tax_slab_dict={}
            for item in tax_list:
                if item["igst_percentage"] in l:
                    #d[cnt] = d[cnt]
                    cnt = item["igst_percentage"]
                    tax_slab_dict[cnt]["taxable"] = tax_slab_dict[cnt]["taxable"] + item["taxable"]
                    tax_slab_dict[cnt]["igst_amount"] = tax_slab_dict[cnt]["igst_amount"] + item["igst_amount"]
                else:
                    cnt = item["igst_percentage"]
                    l.append(item["igst_percentage"])
                    tax_slab_dict[cnt] = {}
                    tax_slab_dict[cnt]["taxable"] = item["taxable"]
                    tax_slab_dict[cnt]["igst_percentage"] = item["igst_percentage"]
                    tax_slab_dict[cnt]["igst_amount"] = item["igst_amount"]
        if cgst_present != -1:
            print("Tax slabs sub total",tax_slab_dict)
            # print("Tax slabs sub total v2",tax_slab_dict2)
        
        if cgst_present == 1:
            for key,value in tax_slab_dict.items():
                if value["cgst_percentage"] == 2.5:
                    final_prediction_copy["subTotal_5%"]["text"] = round(value["taxable"],2)
                    final_prediction_copy["CGSTAmount_2.5%"]["text"] = round(value["cgst_amount"],2)
                    final_prediction_copy["SGSTAmount_2.5%"]["text"] = round(value["sgst_amount"],2)
                    final_prediction_copy["IGSTAmount_5%"]["text"] = 0

                    final_prediction_copy["subTotal_5%"]["final_confidence_score"] = conf
                    final_prediction_copy["CGSTAmount_2.5%"]["final_confidence_score"] = conf
                    final_prediction_copy["SGSTAmount_2.5%"]["final_confidence_score"] = conf
                    final_prediction_copy["IGSTAmount_5%"]["final_confidence_score"] = conf
                    
                elif value["cgst_percentage"] == 6:
                    final_prediction_copy["subTotal_12%"]["text"] = round(value["taxable"],2)
                    final_prediction_copy["CGSTAmount_6%"]["text"] = round(value["cgst_amount"],2)
                    final_prediction_copy["SGSTAmount_6%"]["text"] = round(value["sgst_amount"],2)
                    final_prediction_copy["IGSTAmount_12%"]["text"] = 0

                    final_prediction_copy["subTotal_12%"]["final_confidence_score"] = conf
                    final_prediction_copy["CGSTAmount_6%"]["final_confidence_score"] = conf
                    final_prediction_copy["SGSTAmount_6%"]["final_confidence_score"] = conf
                    final_prediction_copy["IGSTAmount_12%"]["final_confidence_score"] = conf

                elif value["cgst_percentage"] == 9:
                    final_prediction_copy["subTotal_18%"]["text"] = round(value["taxable"],2)
                    final_prediction_copy["CGSTAmount_9%"]["text"] = round(value["cgst_amount"],2)
                    final_prediction_copy["SGSTAmount_9%"]["text"] = round(value["sgst_amount"],2)
                    final_prediction_copy["IGSTAmount_18%"]["text"] = 0

                    final_prediction_copy["subTotal_18%"]["final_confidence_score"] = conf
                    final_prediction_copy["CGSTAmount_9%"]["final_confidence_score"] = conf
                    final_prediction_copy["SGSTAmount_9%"]["final_confidence_score"] = conf
                    final_prediction_copy["IGSTAmount_18%"]["final_confidence_score"] = conf

                elif value["cgst_percentage"] == 14:
                    final_prediction_copy["subTotal_28%"]["text"] = round(value["taxable"],2)
                    final_prediction_copy["CGSTAmount_14%"]["text"] = round(value["cgst_amount"],2)
                    final_prediction_copy["SGSTAmount_14%"]["text"] = round(value["sgst_amount"],2)
                    final_prediction_copy["IGSTAmount_28%"]["text"] = 0

                    final_prediction_copy["subTotal_28%"]["final_confidence_score"] = conf
                    final_prediction_copy["CGSTAmount_14%"]["final_confidence_score"] = conf
                    final_prediction_copy["SGSTAmount_14%"]["final_confidence_score"] = conf
                    final_prediction_copy["IGSTAmount_28%"]["final_confidence_score"] = conf

        elif cgst_present == 0:
            for key,value in tax_slab_dict.items():
                if value["igst_percentage"] == 5:
                    final_prediction_copy["subTotal_5%"]["text"] = round(value["taxable"],2)
                    final_prediction_copy["IGSTAmount_5%"]["text"] = round(value["igst_amount"],2)
                    final_prediction_copy["CGSTAmount_2.5%"]["text"] = 0
                    final_prediction_copy["SGSTAmount_2.5%"]["text"] = 0

                    final_prediction_copy["subTotal_5%"]["final_confidence_score"] = conf
                    final_prediction_copy["IGSTAmount_5%"]["final_confidence_score"] = conf
                    final_prediction_copy["CGSTAmount_2.5%"]["final_confidence_score"] = conf
                    final_prediction_copy["SGSTAmount_2.5%"]["final_confidence_score"] = conf
                    
                elif value["igst_percentage"] == 12:
                    final_prediction_copy["subTotal_12%"]["text"] = round(value["taxable"],2)
                    final_prediction_copy["IGSTAmount_12%"]["text"] = round(value["igst_amount"],2)
                    final_prediction_copy["CGSTAmount_6%"]["text"] = 0
                    final_prediction_copy["SGSTAmount_6%"]["text"] = 0

                    final_prediction_copy["subTotal_12%"]["final_confidence_score"] = conf
                    final_prediction_copy["IGSTAmount_12%"]["final_confidence_score"] = conf
                    final_prediction_copy["CGSTAmount_6%"]["final_confidence_score"] = conf
                    final_prediction_copy["SGSTAmount_6%"]["final_confidence_score"] = conf
                    
                elif value["igst_percentage"] == 18:
                    final_prediction_copy["subTotal_18%"]["text"] = round(value["taxable"],2)
                    final_prediction_copy["IGSTAmount_18%"]["text"] = round(value["igst_amount"],2)
                    final_prediction_copy["CGSTAmount_9%"]["text"] = 0
                    final_prediction_copy["SGSTAmount_9%"]["text"] = 0

                    final_prediction_copy["subTotal_18%"]["final_confidence_score"] = conf
                    final_prediction_copy["IGSTAmount_18%"]["final_confidence_score"] = conf
                    final_prediction_copy["CGSTAmount_9%"]["final_confidence_score"] = conf
                    final_prediction_copy["SGSTAmount_9%"]["final_confidence_score"] = conf

                elif value["igst_percentage"] == 28:
                    final_prediction_copy["subTotal_28%"]["text"] = round(value["taxable"],2)
                    final_prediction_copy["IGSTAmount_28%"]["text"] = round(value["igst_amount"],2)
                    final_prediction_copy["CGSTAmount_14%"]["text"] = 0
                    final_prediction_copy["SGSTAmount_14%"]["text"] = 0

                    final_prediction_copy["subTotal_28%"]["final_confidence_score"] = conf
                    final_prediction_copy["IGSTAmount_28%"]["final_confidence_score"] = conf
                    final_prediction_copy["CGSTAmount_14%"]["final_confidence_score"] = conf
                    final_prediction_copy["SGSTAmount_14%"]["final_confidence_score"] = conf

        return final_prediction_copy
    except:
        print(traceback.print_exc(),
              "Tax Slab Exception")
        return final_prediction

def derive_lineitem_columns(pred):
    """
    Code added by Pramod to derive Quantity, ItemValue and UnitPrice
    when missing or wrongly extracted
    Derive itemQuantity, itemValue or unitPrice
    """

    def clean_amount(amount):
        """
        """
        text = amount
        chars_to_keep = '[^0123456789.]'
        updated_text = re.sub(chars_to_keep, '', text)
        split = updated_text.split('.')
        check_for_no_tokens = (len([i for i in split if i != ""])==0)
        if len(split) > 2:
            updated_text = ''.join(split[:-1])
            updated_text = updated_text + "."
            updated_text = updated_text + str(split[-1])
        if (updated_text == "") or (check_for_no_tokens):
            return None
        amount = float(updated_text)
        return amount

    print("Inside derive_lineitem_columns")
    changed_pred = {}
    for page, page_prediction in pred.items():
        changed_page_prediction = {}

        for row, line_prediction in page_prediction.items():
            dict_predicted_values = {}
            dict_derived_values = {}
            # Get prediction for one row
            for item in line_prediction:
                col_name = list(item.keys())[0]
                predicted_value = item[col_name]['text']
                if (col_name == "itemQuantity") | (col_name == "unitPrice") | (col_name == "itemValue"):
                    dict_predicted_values[col_name] = predicted_value
            col_list = list(dict_predicted_values.keys())


            # NOTE: Case 1 All three are available
            # Check if one of them is incorrect, by validating datatypes
            if {"itemQuantity","unitPrice","itemValue"}.issubset(set(col_list)):
                col_to_derive = []

                for key,value in dict_predicted_values.items():
                    if parse_price(value).amount_float is None:
                        col_to_derive.append(key)
                col_to_derive = list(set(col_to_derive))

                for key, value in dict_predicted_values.items():
                    for field in col_to_derive:
                        if field == "itemQuantity":
                            item_val = clean_amount(dict_predicted_values['itemValue'])
                            unit_price = clean_amount(dict_predicted_values['unitPrice'])
                            if (item_val is not None) and (unit_price is not None) and (unit_price != 0.0):
                                dict_derived_values['itemQuantity'] = item_val/unit_price
                        elif field == "unitPrice":
                            item_val = clean_amount(dict_predicted_values['itemValue'])
                            item_quan = clean_amount(dict_predicted_values['itemQuantity'])
                            if (item_val is not None) and (item_quan is not None) and (item_quan != 0.0):
                                dict_derived_values['unitPrice'] = item_val/item_quan
                        else:
                            unit_price = clean_amount(dict_predicted_values['unitPrice'])
                            item_quan = clean_amount(dict_predicted_values['itemQuantity'])
                            if (unit_price is not None) and (item_quan is not None):
                                dict_derived_values['itemValue'] = unit_price*item_quan



            # NOTE: Case 2 Only two are available; third must be calculated. Quantity/Unit Price/ Item Value
            # Check if one of the values is missing; Convert the other two to numeric data; calculate the missing value
            elif ("itemQuantity" not in col_list) and {"unitPrice","itemValue"}.issubset(set(col_list)):
                for key, value in dict_predicted_values.items():
                    item_val = clean_amount(dict_predicted_values['itemValue'])
                    unit_price = clean_amount(dict_predicted_values['unitPrice'])
                    if (item_val is not None) and (unit_price is not None) and (unit_price != 0.0):
                        dict_derived_values['itemQuantity'] = item_val/unit_price

            elif ("unitPrice" not in col_list) and {"itemQuantity","itemValue"}.issubset(set(col_list)):
                for key, value in dict_predicted_values.items():
                    item_val = clean_amount(dict_predicted_values['itemValue'])
                    item_quan = clean_amount(dict_predicted_values['itemQuantity'])
                    if (item_val is not None) and (item_quan is not None) and (item_quan != 0.0):
                        dict_derived_values['unitPrice'] = item_val/item_quan

            elif ("itemValue" not in col_list) and {"unitPrice","itemQuantity"}.issubset(set(col_list)):
                for key, value in dict_predicted_values.items():
                    unit_price = clean_amount(dict_predicted_values['unitPrice'])
                    item_quan = clean_amount(dict_predicted_values['itemQuantity'])
                    if (unit_price is not None) and (item_quan is not None):
                        dict_derived_values['itemValue'] = unit_price*item_quan

            changed_line_prediction = []
            # Get prediction for one row
            image_h = None
            image_w = None
            for item in line_prediction:
                col_name = list(item.keys())[0]
                predicted_value = item[col_name]
                image_h = predicted_value['image_height']
                image_w = predicted_value['image_widht']
                # Change this logi
                if col_name in dict_derived_values:
                    predicted_value['text'] = str(dict_derived_values[col_name])
                    predicted_value['left'] = 0
                    predicted_value['right'] = 0
                    predicted_value['top'] = 0
                    predicted_value['bottom'] = 0
                    predicted_value['model_confidence'] = 1
                    del dict_derived_values[col_name]
                changed_line_prediction.append({col_name: predicted_value})
            for derived_col, derived_val in dict_derived_values.items():
                derived_lineitem = {}
                derived_lineitem[derived_col] = {'text': str(derived_val),
                'prediction_probability': 1,
                'conf': 1,
                'left': 0,
                'right': 0,
                'top': 0,
                'bottom': 0,
                'image_height': image_h,
                'image_widht': image_w,
                'Odds': 1,
                'model_confidence': 1
                }
                changed_line_prediction.append(derived_lineitem)
            changed_page_prediction[row] = changed_line_prediction
        changed_pred[page] = changed_page_prediction

    return changed_pred



def refine_lineitem_prediction(pred):
    """
    """
    changed_pred = {}
    for page, page_prediction in pred.items():
        changed_page_prediction = {}
        for row, line_prediction in page_prediction.items():
            changed_line_prediction = []
            for item in line_prediction:
                col_name = list(item.keys())[0]
                predicted_value = item[col_name]
                if (col_name == "itemQuantity") | (col_name == "unitPrice") | (col_name == "itemValue") | (col_name == "HSNCode"):
                    # If the prediction has multiple values seprated by space
                    # Extract the first float value as prediction and discard other
                    values = str(predicted_value['text']).split(' ')
                    if len(values) > 1:
                        for v in values:
                            v = str(v).replace(",", "")
                            v = str(v).replace("$", "")
                            s = convert_float(v)
                            if s != '':
                                predicted_value['text'] = str(s)
                                break
                    changed_line_prediction.append({col_name: predicted_value})
                elif (col_name == "UOM"):
                    # If the prediction has multiple values seprated by space
                    # Extract the first string value as prediction and discard other
                    values = str(predicted_value['text']).split(' ')
                    if len(values) > 1:
                        for v in values:
                            v = str(v).replace(".", "")
                            if str(v).isalpha() == True:
                                predicted_value['text'] = str(v)
                                break
                    changed_line_prediction.append({col_name: predicted_value})
                else:
                    changed_line_prediction.append({col_name: predicted_value})
            changed_page_prediction[row] = changed_line_prediction
        changed_pred[page] = changed_page_prediction

    return changed_pred


def change_lineitem_column_JSON(pred, old_name, new_name):
    """
    """
    changed_pred = {}
    for page, page_prediction in pred.items():
        changed_page_prediction = {}
        for row, line_prediction in page_prediction.items():
            changed_line_prediction = []
            for item in line_prediction:
                col_name = list(item.keys())[0]
                predicted_value = item[col_name]
                if col_name == old_name:
                    changed_line_prediction.append({new_name: predicted_value})
                else:
                    changed_line_prediction.append({col_name: predicted_value})
            changed_page_prediction[row] = changed_line_prediction
        changed_pred[page] = changed_page_prediction

    return changed_pred
def calcAmountFields_validation(df,prediction):

    try:
        import re
        vendorGSTIN = prediction.get("vendorGSTIN")
        
        billingGSTIN = prediction.get("billingGSTIN")
        print(billingGSTIN,"RAJINI")
        cgst = -1
        if str(vendorGSTIN['text'])[:2]==str(billingGSTIN['text'])[:2]:
            cgst = 1
        else:
            cgst = 0
        #Get vendor and buyer gstin
        # vendorGstins = list(df[df["predict_label"] == 'vendorGstin']["text"])
        # vendorGstins = list(set(vendorGstins))
        # vendorGstin = vendorGstins[0]
        # billingGstins = list(df[df["predict_label"] == 'billingGstin']["text"])
        # billingGstins = list(set(billingGstins))
        # billingGstin = billingGstins[0]

        # unqGSTins = list(df[df["is_gstin_format"] == 1]["text"])
        # unqGSTins = list(set(unqGSTins))
        # GSTIN_PATTERN = r"\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}"
        # #Cgst or Igst
        # cgst = -1
        # if len(unqGSTins) > 1:
        #     firstGSTIN = unqGSTins[0]
        #     secondGSTIN = unqGSTins[1]
        #     first_span = re.search(GSTIN_PATTERN, firstGSTIN).span()
        #     firstGSTIN = firstGSTIN[first_span[0]:first_span[1]]
        #     second_span = re.search(GSTIN_PATTERN, secondGSTIN).span()
        #     secondGSTIN = secondGSTIN[second_span[0]:second_span[1]]

        #     # if vendorGstin[:2] == billingGstin[:2]:
        #     if firstGSTIN[:2] == secondGSTIN[:2]:
        #         cgst = 1
        #     else:
        #         cgst = 0

        #Get all extracted amounts
        max_line_row = max(list(df["line_row"]))
        # commented the calculate total amount limit 
        df_amt = df[(df["extracted_amount"] > 0) & # (df["extracted_amount"] < 1000000) & 
                    ((df["line_row"] == 0) | (df["line_row"] == max_line_row))]
        # commented the calculate total amount limit 
        ocr_amts = list(df_amt["extracted_amount"])
        unq_amts = list(set(ocr_amts))
        unq_amts = sorted(unq_amts,
                          reverse = True)

        #loop through top 5 highest amounts only
        max_range = min(3,len(unq_amts))
        print("Max Range",max_range)
        for i in range(0,max_range):
            max_amt = unq_amts[i]
            amts = unq_amts[i+1:]
            if len(amts) > 0:
                print("Calling Tax OnlyAmts",
                      max_amt,
                      cgst,
                      amts)
                result = taxOnlyAmts(max_amt,
                                     cgst,
                                     amts)
                # print("max",max_amt,result)
                if result == {}:
                    continue
                elif result is None:
                    continue
                else:
                    # print(result)
                    return result
            else:
                break
        # result = {"total":unq_amts[0],
        #           "subTotal":unq_amts[0],
        #           "cgst":0.0,
        #           "sgst":0.0,
        #           "igst":0.0,
        #           "totalGst":0.0}
        # print(result)
        # return result
        return None
    except:
        print(traceback.print_exc(),
              "Calc Amount Fields")
        return None



def UOM_SWAP_CHECK(pred, DF):
    """
    """
    print("Inside UOM_SWAP_CHECK")

    cols_extracted = list(DF.columns)

    # Convert UOM, UnitPrice, ItemValue and ItemQuantity to amount
    try:
        for col in set(['UOM', 'itemValue', 'unitPrice', 'itemQuantity']):
            if col in cols_extracted:
                DF[col] = DF[col].str.replace(',', '')
                DF[col] = DF[col].apply(lambda x: extractAmount(x)[1])

        if ("itemValue" in cols_extracted) & ("UOM" in cols_extracted):
            if ("unitPrice" in cols_extracted) & ("itemQuantity" in cols_extracted):
                return pred
            elif ("unitPrice" in cols_extracted):
                # Validate itemValue with unitPrice and UOM
                DF['Calculated_itemValue'] = DF['UOM'] * DF['unitPrice']
                DF['CHECK'] = np.isclose(DF['Calculated_itemValue'], DF['itemValue'])
                match_score = DF['CHECK'].sum()/DF.shape[0]
                if match_score > 0.5:
                    return change_lineitem_column_JSON(pred, "UOM", "itemQuantity")
            elif ("itemQuantity" in cols_extracted):
                # Validate itemValue with unitPrice and UOM
                DF['Calculated_itemValue'] = DF['UOM'] * DF['itemQuantity']
                DF['Calculated_itemValue'] = DF['UOM'] * DF['itemQuantity']
                DF['CHECK'] = np.isclose(DF['Calculated_itemValue'], DF['itemValue'])
                match_score = DF['CHECK'].sum()/DF.shape[0]
                if match_score > 0.5:
                    return change_lineitem_column_JSON(pred, "UOM", "unitPrice")
    except:
        return pred
    return pred
@putil.timing
def line_items_(DF, prediction):
    """
    New method added to extract line items from model output
    Row Number is taken from line_row
    """
    # Convert text to string dtype
    DF['text'] = DF['text'].astype("str")
    # Define Line Item Area for each page from Model Prediction and line_row
    TEMP = DF.loc[(DF['line_row'] > 0) & (DF["predict_label"].isin(FIELD['lineitem_value_labels']))]
    # Reassign row numbers
    pages = list(TEMP['page_num'].unique())
    TEMP['new_line_row'] = 0
    for p in pages:
        A = TEMP.loc[TEMP['page_num'] == p].groupby(['line_row'])[['line_top']].mean().reset_index().sort_values(['line_top'])
        A['new_line_row'] = [float(x+1) for x in range(0, A.shape[0])]
        print(A)
        dict_swap = dict(zip(A['line_row'], A['new_line_row']))
        for k,v in dict_swap.items():
            TEMP.loc[(TEMP['page_num'] == p) & (TEMP['line_row'] == k) , 'new_line_row'] = v


    # Extract final value for each page and row
    pages = list(TEMP['page_num'].unique())
    pages.sort()
    TEMP['model_confidence'] = TEMP['prediction_probability']
    line_item_prediction = {}
    print(TEMP[['page_num', 'new_line_row','text','predict_label']])
    for p in pages:
        page_prediction = {}
        PAGE = TEMP.loc[TEMP['page_num'] == p]
        PAGE.sort_values(['line_num', 'word_num'], ascending=[True, True], inplace=True) # NOTE: added 4th March, to be tested
        line_rows = list(PAGE['new_line_row'].unique())
        line_rows.sort()
        print(p, line_rows)
        for l in line_rows:
            row_prediction = []
            for f in FIELD['lineitem_value_labels']:
                FIELD_PREDICTION = PAGE.loc[(PAGE['new_line_row'] == l) & ((PAGE['predict_label'] == f))]
                if FIELD_PREDICTION.shape[0] == 0:
                    continue
                if f == 'LI_itemDescription':
                    pred = FIELD_PREDICTION.groupby(['new_line_row']).agg({'text': lambda x: "%s" % ' '.join(x),
                        'left': 'min',
                        'right': 'max',
                        'top': 'min',
                        'bottom': 'max',
                        'conf': 'mean',
                        'prediction_probability':'mean',
                        'model_confidence':'mean',
                        'image_height': 'first',
                        'image_widht': 'first'}).reset_index()
                    dict_row_ = pred.iloc[0].to_dict()
                    del dict_row_['new_line_row']
                    row_prediction.append({f[3:]: dict_row_})
                else:
                    FIELD_PREDICTION.sort_values('prediction_probability', ascending=False, inplace = True)
                    # FIELD_PREDICTION = FIELD_PREDICTION.groupby(['line_num']).agg({'text': lambda x: "%s" % ''.join(x),
                    #     'left': 'min',
                    #     'right': 'max',
                    #     'top': 'min',
                    #     'bottom': 'max',
                    #     'conf': 'mean',
                    #     'prediction_probability':'mean',
                    #     'model_confidence':'mean',
                    #     'image_height': 'first',
                    #     'image_widht': 'first'}).reset_index()
                    # FIELD_PREDICTION.sort_values('prediction_probability', ascending=False, inplace = True)
                    dict_row_ = FIELD_PREDICTION[['text', 'left', 'right', 'top', 'bottom',
                    'conf', 'prediction_probability', 'model_confidence', 'image_height',
                    'image_widht']].iloc[0].to_dict()
                    row_prediction.append({f[3:]: dict_row_})
            page_prediction[l] = row_prediction
        line_item_prediction[p] = page_prediction
    line_item_prediction = derive_lineitem_columns(line_item_prediction)
    # print("LineItem New Method:", line_item_prediction)
    return line_item_prediction

def procefinal_candidates_vender_specific_header_without_vendormasterdata(field, DF):
    if field in PREDICTION_THRESHOLD:
        TEMP = DF.loc[DF["prob_" + field] > PREDICTION_THRESHOLD[field]]
    else:
        TEMP = DF.loc[DF["predict_label"] == field] # Model Prediction
    if TEMP.shape[0] == 0:
        return {field: None}
    TEMP = TEMP.sort_values("prob_" + field, ascending=False).head(10)

    TEMP["Label_Present"] = False

    TEMP["text"] = TEMP["text"].astype(str)
    """
    final_candidates_ = TEMP.sort_values(['word_num'], ascending=True).groupby(['line_num']).agg(
        {"prob_" + field: 'mean',
         'text': lambda x: "%s" % ' '.join(x),
         'Label_Present': 'sum',
         'word_num': 'count',
         'left': 'min',
         'right': 'max',
         'top': 'min',
         'bottom': 'max',
         'conf': 'mean',
         'page_num': 'first',
         'height': 'first',
         'width': 'first',
         'image_height': 'first',
         'image_widht': 'first'}).reset_index()
    """
    final_candidates_ = TEMP.sort_values(['page_num','line_num','word_num'], ascending=[True,True,True]).groupby(['page_num','line_num']).agg(
        {"prob_" + field: 'mean',
         'text': lambda x: "%s" % ' '.join(x),
         'Label_Present': 'sum',
         'word_num': 'count',
         'left': 'min',
         'right': 'max',
         'top': 'min',
         'bottom': 'max',
         'conf': 'mean',
         'height': 'first',
         'width': 'first',
         'image_height': 'first',
         'image_widht': 'first'}).reset_index()

    final_candidates_["label_confidence"] = 0.0
    final_candidates_["wordshape"] = ""
    final_candidates_["wordshape_confidence"] = 0.0

    final_candidates_["Odds"] = final_candidates_['prob_' + field] / (1 - final_candidates_['prob_' + field])
    final_candidates_['model_confidence'] = final_candidates_['prob_'+ field].apply(logistic_function)

    final_candidates_['final_confidence_score'] = final_candidates_['model_confidence']
    final_candidates_['vendor_masterdata_present'] = False
    final_candidates_['extracted_from_masterdata'] = False

    final_candidates_.sort_values(['page_num','final_confidence_score'], ascending=[True,False], inplace=True)

    if final_candidates_.shape[0] > 0:
        return {field: final_candidates_.iloc[0].to_dict()}
    else:
        return {field: None}

def procefinal_candidates_vender_names_without_vendormasterdata(field, DF):
    if field in PREDICTION_THRESHOLD:
        TEMP = DF.loc[DF["prob_" + field] > PREDICTION_THRESHOLD[field]]
    else:
        TEMP = DF.loc[DF["predict_label"] == field] # Model Prediction
    if TEMP.shape[0] == 0:
        return {field: None}
    TEMP = TEMP.sort_values("prob_" + field, ascending=False).head(10)

    TEMP["Label_Present"] = False

    TEMP["line_text"] = TEMP["line_text"].astype(str)

    """
    final_candidates_ = TEMP.sort_values(['word_num'], ascending=True).groupby(['line_num']).agg(
        {"prob_" + field: 'mean',
         'text': lambda x: "%s" % ' '.join(x),
         'Label_Present': 'sum',
         'word_num': 'count',
         'left': 'min',
         'right': 'max',
         'top': 'min',
         'bottom': 'max',
         'conf': 'mean',
         'page_num': 'first',
         'height': 'first',
         'width': 'first',
         'image_height': 'first',
         'image_widht': 'first'}).reset_index()
    """
    final_candidates_ = TEMP.sort_values(['page_num','line_num','word_num'], ascending=[True,True,True]).groupby(['page_num','line_num']).agg(
        {"prob_" + field: 'mean',
         'line_text': 'first',#lambda x: "%s" % ' '.join(x),
         'Label_Present': 'sum',
         'word_num': 'count',
         'left': 'min',
         'right': 'max',
         'top': 'min',
         'bottom': 'max',
         'conf': 'mean',
         'height': 'first',
         'width': 'first',
         'image_height': 'first',
         'image_widht': 'first'}).reset_index()
    final_candidates_.rename(columns={"line_text":"text"},inplace = True)
    final_candidates_["label_confidence"] = 0.0
    final_candidates_["wordshape"] = ""
    final_candidates_["wordshape_confidence"] = 0.0

    final_candidates_["Odds"] = final_candidates_['prob_' + field] / (1 - final_candidates_['prob_' + field])
    final_candidates_['model_confidence'] = final_candidates_['prob_'+ field].apply(logistic_function)

    final_candidates_['final_confidence_score'] = final_candidates_['model_confidence']
    final_candidates_['vendor_masterdata_present'] = False
    final_candidates_['extracted_from_masterdata'] = False

    final_candidates_.sort_values(['page_num','final_confidence_score'], ascending=[True,False], inplace=True)

    if final_candidates_.shape[0] > 0:
        return {field: final_candidates_.iloc[0].to_dict()}
    else:
        return {field: None}

def procefinal_candidates_vender_specific_fields_with_vendormasterdata(field, DF, vendor_masterdata_score,
                                                                       vendor_masterdata):
    final_candidates_ = {}
    if pd.isna(vendor_masterdata[field]):
        return {}
    final_candidates_['line_num'] = None
    final_candidates_["prob_" + field] = None
    final_candidates_['text'] = vendor_masterdata[field]
    final_candidates_["Label_Present"] = None
    final_candidates_['word_num'] = None

    final_candidates_['left'] = 0
    final_candidates_['right'] = 1
    final_candidates_['conf'] = 1
    final_candidates_['top'] = 0
    final_candidates_['bottom'] = 1

    final_candidates_['page_num'] = 0
    final_candidates_['image_height'] = 1
    final_candidates_['image_widht'] = 1

    final_candidates_["label_confidence"] = None
    final_candidates_["wordshape"] = None
    final_candidates_["wordshape_confidence"] = None
    final_candidates_["Odds"] = None
    final_candidates_['model_confidence'] = None

    final_candidates_['final_confidence_score'] = 1
    final_candidates_['vendor_masterdata_present'] = True
    final_candidates_['extracted_from_masterdata'] = True

    return {field: final_candidates_}


def procefinal_candidates_vendor_addrefinal_candidates_without_vendormasterdata(field, DF_INV):
    """
    """
    prediction = {}

    candidate = []
    TEMP = DF_INV[DF_INV['predict_label'] == field]
    # TEMP = TEMP.loc[TEMP["prob_" + field] > CONST['model_threshold']]
    if TEMP.shape[0] == 0:
        prediction[field] = None
        return prediction

    TEMP["text"] = TEMP["text"].astype(str)
    LINE_CANDIDATES = TEMP.groupby(['FileName', 'page_num', 'line_num']).agg(
        {"prob_" + field: 'mean',
         'left': 'min',
         'text': lambda x: "%s" % ' '.join(x),

         }).reset_index()

    pages = set(LINE_CANDIDATES['page_num'])
    for p in pages:
        PAGE = LINE_CANDIDATES.loc[LINE_CANDIDATES['page_num'] == p]
        lines = list(PAGE['line_num'])
        lines.sort()
        indices = [i + 1 for (x, y, i) in zip(lines, lines[1:], range(len(lines))) if CONST['line_gap'] < abs(x - y)]
        result = [lines[start:end] for start, end in zip([0] + indices, indices + [len(lines)])]
        for res in result:
            min_line = min(res)
            max_line = max(res)
            WORD_CANDIDATES = DF_INV.loc[(DF_INV['page_num'] == p) & (DF_INV['line_num'] >= min_line)
                                         & (DF_INV['line_num'] <= max_line)][
                ['FileName', 'page_num', 'line_num', 'word_num',
                 'text', 'left', 'right', 'top', 'bottom', 'conf', 'image_widht', 'image_height', "prob_" + field]]
            line_lefts = list(WORD_CANDIDATES.groupby(['line_num'])[['left']].min().reset_index()['left'])
            line_lefts.sort()
            indices_line_lefts = [i + 1 for (x, y, i) in zip(line_lefts, line_lefts[1:], range(len(line_lefts)))
                                  if CONST['line_start_gap'] < abs(x - y)]
            result_line_lefts = [line_lefts[start:end] for start, end in
                                 zip([0] + indices_line_lefts, indices_line_lefts + [len(line_lefts)])]

            for r in result_line_lefts:
                final_lines = list(WORD_CANDIDATES.loc[WORD_CANDIDATES['left'].isin(r)]['line_num'])
                FINAL_WORDS = WORD_CANDIDATES.loc[WORD_CANDIDATES['line_num'].isin(final_lines)]
                FINAL_WORDS = FINAL_WORDS.sort_values(['line_num', 'word_num'], ascending=True)
                FINAL_WORDS['text'] = FINAL_WORDS['text'].astype(str)
                SS = FINAL_WORDS.groupby(['page_num', 'line_num']).agg({"prob_" + field: 'mean',
                                                                        'text': lambda x: "%s" % ' '.join(x),
                                                                        'left': 'min',
                                                                        'right': 'max',
                                                                        'top': 'min',
                                                                        'bottom': 'max',
                                                                        'conf': 'mean',
                                                                        'image_height': 'first',
                                                                        'image_widht': 'first'}).reset_index()
                SS = SS.groupby(['page_num']).agg({"prob_" + field: 'mean',
                                                   'text': lambda x: "%s" % '\n'.join(x),
                                                   'left': 'min',
                                                   'right': 'max',
                                                   'top': 'min',
                                                   'bottom': 'max',
                                                   'conf': 'mean',
                                                   'image_height': 'first',
                                                   'image_widht': 'first'}).reset_index()

                candidate.append(
                    [p, SS['text'].iloc[0], SS["prob_" + field].iloc[0], SS['left'].iloc[0], SS['right'].iloc[0],
                     SS['top'].iloc[0],
                     SS['bottom'].iloc[0], SS['conf'].iloc[0], SS['image_height'].iloc[0], SS['image_widht'].iloc[0]])

    final_candidate = {"text": "", "prob_" + field: 0.0, 'left': 0,
                       'right': 1,
                       'top': 0,
                       'bottom': 1,
                       'conf': 1,
                       'image_height': 1,
                       'image_widht': 1}

    for c in candidate:
        if c[2] > final_candidate["prob_" + field]:
            final_candidate['page_num'] = c[0]
            final_candidate["text"] = c[1]
            final_candidate["prob_" + field] = c[2]
            final_candidate['left'] = c[3]
            final_candidate['right'] = c[4]
            final_candidate['top'] = c[5]
            final_candidate['bottom'] = c[6]
            final_candidate['conf'] = c[7]
            final_candidate['image_height'] = c[8]
            final_candidate['image_widht'] = c[9]

    final_candidate["label_confidence"] = None
    final_candidate["wordshape"] = None
    final_candidate["wordshape_confidence"] = None
    final_candidate["Odds"] = final_candidate['prob_' + field] / (1 - final_candidate['prob_' + field])
    final_candidate["model_confidence"] = logistic_function(final_candidate['prob_' + field])
    final_candidate['final_confidence_score'] = final_candidate['model_confidence']
    final_candidate['vendor_masterdata_present'] = False
    final_candidate['extracted_from_masterdata'] = False
    prediction[field] = final_candidate
    return prediction
### START of changes for Issue number PBAIP-21
def exclude_rows_with_keywords_invnum(df):
    df_copy = df.copy(deep = True)
    try:
        keywords_to_include = ["invoice",'inv']
        keywords_to_exclude = ['tax','sales']
        #pattern to say if we are excluding keywords only if their is DATE keyword associated with iit
        pattern = "(?=.*(?:{}))".format("|".join(keywords_to_exclude))
        #to filter  rows containing the keywords
        df1 = df[(df['left_processed_ngbr'].str.contains('|'.join(keywords_to_include), na=False,case=False) | df['above_processed_ngbr'].str.contains('|'.join(keywords_to_include), na=False,case=False))]
        print(df1[['text','left_processed_ngbr','above_processed_ngbr']])
        if df1.empty:
            return ((df))
        # Use boolean indexing to filter out rows containing the keywords
        filtered_df = df1[~df1['left_processed_ngbr'].str.split(',').str.get(0).str.contains(pattern, na=False, case=False, flags=re.IGNORECASE) & ~df1['above_processed_ngbr'].str.split(',').str.get(0).str.contains(pattern, na=False, case=False, flags=re.IGNORECASE)]
        #filtered_df = df[~(df['left_processed_ngbr'].str.split(',').str.get(0).str.contains(pattern, na=False,case=False) | df['above_processed_ngbr'].str.split(',').str.get(0).str.contains(pattern, na=False,case=False))]
        #filtered_df = df[~(df['left_processed_ngbr'].str.split(',').str.get(0).str.contains('|'.join(keywords_to_exclude), na=False,case=False) | df['above_processed_ngbr'].str.split(',').str.get(0).str.contains('|'.join(keywords_to_exclude), na=False,case=False))]
        if filtered_df.empty:
            return ((df1))
        print(filtered_df.shape,'asd1')
        # if filtered_df.shape[0] == 0:
        #     filtered_df = check_unique_dates(df)
        print(filtered_df.shape,'asd')
        # return the filtered DataFrame
        return ((filtered_df))
    except:
        print("exception",traceback.print_exc())
        return df_copy
def pattern_to_regex(pattern):
    return "\\b" + ''.join('[A-Z]' if char == 'X' else '[a-z]' if char == 'x' else '\d' if char == 'd' else re.escape(char) for char in pattern) + "\\b"

def get_matched_text(input_string,shape):
    try:
        #input_string = "No & Date SI3682425-00246, 22-04-2024  SA3682425-00246"
        pattern = pattern_to_regex(shape)
        print(pattern)
        # Find all matching substrings
        matches = re.findall(pattern, input_string)
        if len(matches)>0:
            return matches[0]
        else:
            return input_string
    except:
        return input_string
def get_inv_num_token_dataframe(df,TEMP,vendor_pan):
    try:
        refdat=REFERENCE_DATA[REFERENCE_DATA["review_status"]==1]
        refdat=refdat[refdat["field_name"]=='invoiceNumber']
        filtered_refdat = refdat[refdat['vendor_id'].str[2:12] == vendor_pan]
        #word_shapes_to_filter = filtered_refdat['field_shape'].unique().tolist()
        word_shapes_to_filter = filtered_refdat['field_shape'].dropna().replace('', np.nan).dropna().unique().tolist()
        print(word_shapes_to_filter,"word_shapes_to_filter")
        #filtered_df = df[df['wordshape'].str.contains('|'.join(word_shapes_to_filter))]
        filtered_df = df[df['wordshape'].isin(word_shapes_to_filter)]
        if filtered_df.empty:
            filtered_df = df[df['wordshape'].apply(lambda x: any(substring in x for substring in word_shapes_to_filter))]
        filtered_df = exclude_rows_with_keywords_invnum(filtered_df)
        if filtered_df.empty:
            print("returning temp")
            return TEMP
        else:
            input_string = filtered_df['text'].iloc[0]
            for shape in word_shapes_to_filter:
                refined_text = get_matched_text(input_string,shape)
                if refined_text != input_string:
                    break
            filtered_df.at[0, 'text'] = refined_text
            return filtered_df.head(1)
    except:
        print("exception",traceback.print_exc())
        return TEMP
### END of changes for Issue number PBAIP-21
def get_inv_num_using_refdata(df,word_shapes_to_filter,prediction):
    try:
        print(word_shapes_to_filter,"word_shapes_to_filter")
        #filtered_df = df[df['wordshape'].str.contains('|'.join(word_shapes_to_filter))]
        filtered_df = df[df['wordshape'].isin(word_shapes_to_filter)]
        if filtered_df.empty:
            filtered_df = df[df['wordshape'].apply(lambda x: any(substring in x for substring in word_shapes_to_filter))]
        filtered_df = exclude_rows_with_keywords_invnum(filtered_df)
        if filtered_df.empty:
            print("returning temp")
            return prediction
        else:
            keys = ['text', 'left', 'right', 'top', 'bottom', 'image_height', 'image_widht']
            # Update the prediction dictionary for 'invoiceNumber' using dataframe
            for key in keys:
                prediction['invoiceNumber'][key] = filtered_df[key].iloc[0]
            return prediction
    except:
        print("exception",traceback.print_exc())
        return prediction
def procefinal_candidates_header_without_vendormasterdata(field, DF, vendor_pan):
    """
    """
    # Sort by Model Probability and pick top 10 values
    if field in PREDICTION_THRESHOLD:
        TEMP = DF.loc[DF["prob_" + field] > PREDICTION_THRESHOLD[field]]
        # print("getting df greater than threshould prob ",TEMP.shape,PREDICTION_THRESHOLD[field])
    else:
        TEMP = DF.loc[DF["predict_label"] == field] # Model Prediction
        # print("Picking with model/modified label prediction ",TEMP)
    print(TEMP.shape,'checking inv')
    if field == "invoiceNumber" and TEMP.shape[0] == 0:
        TEMP = get_inv_num_token_dataframe(DF,TEMP, vendor_pan)
        print(TEMP.shape,'narewd')
    if TEMP.shape[0] == 0:
        return {field: None}
    TEMP = TEMP.sort_values("prob_" + field, ascending=False).head(10)

    TEMP["Label_Present"] = False
    final_candidates_ = TEMP.sort_values(['page_num','line_num','word_num'], ascending=[True,True,True]).groupby(['page_num','line_num']).agg(
        {"prob_" + field: 'mean',
         'text': lambda x: "%s" % ' '.join(x),
         'Label_Present': 'sum',
         'word_num': 'count',
         'left': 'min',
         'right': 'max',
         'top': 'min',
         'bottom': 'max',
         'conf': 'mean',
         'height': 'first',
         'width': 'first',
         'image_height': 'first',
         'image_widht': 'first'}).reset_index()

    final_candidates_["label_confidence"] = 0.0
    final_candidates_["wordshape"] = final_candidates_['text'].apply(get_wordshape)
    final_candidates_["wordshape_confidence"] = 0.0

    final_candidates_["Odds"] = final_candidates_['prob_' + field] / (1 - final_candidates_['prob_' + field])
    final_candidates_['model_confidence'] = final_candidates_['prob_' + field].apply(logistic_function)

    final_candidates_['final_confidence_score'] = final_candidates_['model_confidence']
    final_candidates_['vendor_masterdata_present'] = False
    final_candidates_['extracted_from_masterdata'] = False

    final_candidates_.sort_values(['page_num','final_confidence_score'], ascending=[True,False], inplace=True)
    if final_candidates_.shape[0] > 0:
        return {field: final_candidates_.iloc[0].to_dict()}
    else:
        return {field: None}


def procefinal_candidates_amount_without_vendormasterdata(field, DF):
    # Sort by Model Probability and pick top 10 values
    if field in PREDICTION_THRESHOLD:
        TEMP = DF.loc[DF["prob_" + field] > PREDICTION_THRESHOLD[field]]
    else:
        TEMP = DF.loc[DF["predict_label"] == field] # Model Prediction

    if TEMP.shape[0] == 0:
        return {field: None}
    TEMP = TEMP.sort_values("prob_" + field, ascending=False).head(10)

    TEMP['text'] = TEMP['text'].astype(str)
    TEMP["Label_Present"] = False
    final_candidates_ = TEMP.sort_values(['page_num','line_num','word_num'], ascending=[True,True,True]).groupby(['page_num','line_num']).agg(
        {"prob_" + field: 'mean',
         'text': lambda x: "%s" % ' '.join(x),
         'Label_Present': 'sum',
         'word_num': 'count',
         'left': 'min',
         'right': 'max',
         'top': 'min',
         'bottom': 'max',
         'conf': 'mean',
         'height': 'first',
         'width': 'first',
         'image_height': 'first',
         'image_widht': 'first'}).reset_index()

    final_candidates_["label_confidence"] = 0.0
    final_candidates_["wordshape"] = ""
    final_candidates_["wordshape_confidence"] = 0.0

    final_candidates_["Odds"] = final_candidates_['prob_' + field] / (1 - final_candidates_['prob_' + field])
    final_candidates_['model_confidence'] = final_candidates_['prob_' + field].apply(logistic_function)

    final_candidates_['final_confidence_score'] = final_candidates_['model_confidence']
    final_candidates_['vendor_masterdata_present'] = False
    final_candidates_['extracted_from_masterdata'] = False

    final_candidates_.sort_values(['final_confidence_score'], ascending=False, inplace=True)

    if final_candidates_.shape[0] > 0:
        return {field: final_candidates_.iloc[0].to_dict()}
    else:
        return {field: None}


def procefinal_candidates_rate_without_vendormasterdata(field, DF):
    # Sort by Model Probability and pick top 10 values
    TEMP = DF.loc[DF["prob_" + field] > CONST['model_threshold']]
    # TEMP = DF.loc[DF["predict_label"] == field] # Model Prediction
    if TEMP.shape[0] == 0:
        return {field: None}
    TEMP = TEMP.sort_values("prob_" + field, ascending=False).head(10)

    TEMP['text'] = TEMP['text'].astype(str)
    TEMP["Label_Present"] = False
    final_candidates_ = TEMP.sort_values(['word_num'], ascending=True).groupby(['line_num']).agg(
        {"prob_" + field: 'mean',
         'text': lambda x: "%s" % ' '.join(x),
         'Label_Present': 'sum',
         'word_num': 'count',
         'left': 'min',
         'right': 'max',
         'top': 'min',
         'bottom': 'max',
         'conf': 'mean',
         'page_num': 'first',
         'height': 'first',
         'width': 'first',
         'image_height': 'first',
         'image_widht': 'first'}).reset_index()

    final_candidates_["label_confidence"] = 0.0
    final_candidates_["wordshape"] = ""
    final_candidates_["wordshape_confidence"] = 0.0

    final_candidates_["Odds"] = final_candidates_['prob_' + field] / (1 - final_candidates_['prob_' + field])
    final_candidates_['model_confidence'] = final_candidates_['prob_' + field].apply(logistic_function)

    final_candidates_['final_confidence_score'] = final_candidates_['model_confidence']
    final_candidates_['vendor_masterdata_present'] = False
    final_candidates_['extracted_from_masterdata'] = False

    final_candidates_.sort_values(['final_confidence_score'], ascending=False, inplace=True)

    if final_candidates_.shape[0] > 0:
        return {field: final_candidates_.iloc[0].to_dict()}
    else:
        return {field: None}
def check_unique_dates(df):
    try:
        dates = pd.to_datetime(df["text"], errors='coerce')
        print(dates)
        # Check if all dates are unique
        if dates.nunique() == 1:
            return df.head(1)
        else:
            return df.head(0)
    except:
        return df.head(0)
def exclude_rows_with_keywords(df):
    df_copy = df.copy(deep = True)
    try:
        keywords_to_include = ["invoice",'date','dated','inv','Dated']
        keywords_to_exclude = ['po', 'ack', 'reference','store','pickup','DELHIVERY','Printed on','order','Print','sign','sales','lr','p.o','fssai','so','approx','valid','due']
        additional_keyword = "date"
        #pattern to say if we are excluding keywords only if their is DATE keyword associated with iit
        pattern = "(?=.*(?:{}))(?=.*{})".format("|".join(keywords_to_exclude), additional_keyword)
        #to filter  rows containing the keywords
        df = df[(df['left_processed_ngbr'].str.contains('|'.join(keywords_to_include), na=False,case=False) | df['above_processed_ngbr'].str.contains('|'.join(keywords_to_include), na=False,case=False))]
        print(df[['text','left_processed_ngbr','above_processed_ngbr']])
        # Use boolean indexing to filter out rows containing the keywords
        filtered_df = df[~df['left_processed_ngbr'].str.split(',').str.get(0).str.contains(pattern, na=False, case=False, flags=re.IGNORECASE) & ~df['above_processed_ngbr'].str.split(',').str.get(0).str.contains(pattern, na=False, case=False, flags=re.IGNORECASE)]
        #filtered_df = df[~(df['left_processed_ngbr'].str.split(',').str.get(0).str.contains(pattern, na=False,case=False) | df['above_processed_ngbr'].str.split(',').str.get(0).str.contains(pattern, na=False,case=False))]
        #filtered_df = df[~(df['left_processed_ngbr'].str.split(',').str.get(0).str.contains('|'.join(keywords_to_exclude), na=False,case=False) | df['above_processed_ngbr'].str.split(',').str.get(0).str.contains('|'.join(keywords_to_exclude), na=False,case=False))]
        print(filtered_df.shape,'asd1')
        if filtered_df.shape[0] == 0:
            filtered_df = check_unique_dates(df)
        print(filtered_df.shape,'asd')
        # return the filtered DataFrame
        return ((filtered_df))
    except:
        print("exception",traceback.print_exc())
        return df_copy
def get_consecutive_numbers(unique_number, number_list):
    try:
        # Sort the list for easier consecutive number identification
        number_list.sort()

        # Find the index of the unique number in the sorted list
        unique_index = number_list.index(unique_number)

        # Initialize variables to track the start and end of the consecutive sequence
        start_index = unique_index
        end_index = unique_index

        # Find the start of the consecutive sequence
        while start_index > 0 and number_list[start_index - 1] == number_list[start_index] - 1:
            start_index -= 1

        # Find the end of the consecutive sequence
        while end_index < len(number_list) - 1 and number_list[end_index + 1] == number_list[end_index] + 1:
            end_index += 1

        # Extract the consecutive sequence
        consecutive_sequence = number_list[start_index:end_index + 1]

        return consecutive_sequence
    except:
        return [unique_number]
def get_consecutive_sequences_tokens(df, unique_number):
    try:
        # Filter DataFrame where is_date_1 is equal to 1
        filtered_df = df[df['is_date_1'] == 1]
        # Get the unique values of the token_id column
        token_ids = filtered_df['token_id'].unique().tolist()
        # Find consecutive sequences that include the unique token_id
        consecutive_sequence = []
        consecutive_sequence = get_consecutive_numbers(unique_number,token_ids)

        return consecutive_sequence
    except:
        return [unique_number]
# Function to calculate the distance between two rectangles
def distance_between_rectangles(rect1, rect2):
    # Calculate the x and y distances between the rectangles
    dx = max(0, max(rect1['left'], rect2['left']) - min(rect1['right'], rect2['right']))
    dy = max(0, max(rect1['top'], rect2['top']) - min(rect1['bottom'], rect2['bottom']))

    # Calculate the shortest distance
    return np.sqrt(dx**2 + dy**2)
def gettokenforinvdatebasedoninvnumber(df):
    try:
        coordinates_list=[]
        invdf = df.loc[df["prob_" + "invoiceNumber"]>0.24]
        if not invdf.empty:
            invdf_copy = invdf.copy()
            first_row_coords = invdf_copy.iloc[0]

            # Extract bounding box coordinates for the first row
            first_row_rect = {
                'left': first_row_coords['left'],
                'top': 1-first_row_coords['top'],
                'right': first_row_coords['right'],
                'bottom': 1-first_row_coords['bottom']
            }
        else:
            print("inv number not predicted.")
            return False
        invdatedf = df.loc[df["prob_" + "invoiceDate"]>0.24]
        invdatedf=invdatedf[invdatedf["is_date_1"]==1]
        if invdatedf.empty:
            invdatedf=df[df["is_date_1"]==1]
        print("befor",invdatedf["text"],invdatedf["above_processed_ngbr"],invdatedf["left_processed_ngbr"])
        invdatedf = exclude_rows_with_keywords(invdatedf)
        print("after",invdatedf["text"])
        if not invdatedf.empty:
            invdatedf_copy = invdatedf.copy()
            # Calculate the bounding box coordinates for each row in invdatedf
            invdatedf_copy['rect'] = invdatedf_copy.apply(lambda row: {'left': row['left'], 'top': 1-row['top'], 'right': row['right'], 'bottom': 1-row['bottom']}, axis=1)

            # Calculate the distance between the rectangles
            invdatedf_copy['distance'] = invdatedf_copy.apply(lambda row: distance_between_rectangles(first_row_rect, row['rect']), axis=1)
            print(" distance:", invdatedf_copy['distance'])
            # Find the row with the shortest distance
            nearest_neighbor_row = invdatedf_copy.loc[invdatedf_copy['distance'].idxmin()]
            return nearest_neighbor_row['token_id']
        return False
    except:
        print("exception",traceback.print_exc())
        return False
def filterdataframeforinvdatepred(DF,TEMP):
    try:
        invdatetoken = gettokenforinvdatebasedoninvnumber(DF)
        if  invdatetoken:
            token_list = get_consecutive_sequences_tokens(DF, invdatetoken)
            print("token_list",token_list)
            TEMP = DF[(DF["token_id"].isin(token_list))& (DF["is_date_1"] == 1)]
            print(TEMP.shape)
            return TEMP
        else:
            print(TEMP.shape)
            if TEMP.shape[0] == 0:
                TEMP = DF[(DF["is_date_1"] == 1)]
            TEMP = exclude_rows_with_keywords(TEMP)
            if not TEMP.empty:
                first_row_token_id = TEMP.iloc[0]['token_id']
                token_list = get_consecutive_sequences_tokens(TEMP, first_row_token_id)
                print("token_list",token_list)
                TEMP = TEMP[(TEMP["token_id"].isin(token_list))& (TEMP["is_date_1"] == 1)]
                print(TEMP.shape)
            return TEMP
    except:
        print("exception",traceback.print_exc())
        return TEMP
def procefinal_candidates_date_without_vendormasterdata(field, DF):
    # Sort by Model Probability and pick top 10 values
    if field in PREDICTION_THRESHOLD:
        TEMP = DF.loc[DF["prob_" + field] > PREDICTION_THRESHOLD[field]]
    else:
        TEMP = DF.loc[DF["predict_label"] == field] # Model Prediction
    print(TEMP)
    if field == "invoiceDate":
        TEMP = filterdataframeforinvdatepred(DF,TEMP)
    if TEMP.shape[0] == 0:
        print("TEMP shape is zero")
        return {field: None}
    TEMP = TEMP.sort_values("prob_" + field, ascending=False).head(10)

    TEMP['text'] = TEMP['text'].astype(str)
    TEMP["Label_Present"] = False
    final_candidates_ = TEMP.sort_values(['page_num','line_num','word_num'], ascending=[True,True,True]).groupby(['page_num']).agg(
        {"prob_" + field: 'mean',
         'text': lambda x: "%s" % ' '.join(x),
         'Label_Present': 'sum',
         'word_num': 'count',
         'left': 'min',
         'right': 'max',
         'top': 'min',
         'bottom': 'max',
         'conf': 'mean',
         'height': 'first',
         'width': 'first',
         'image_height': 'first',
         'image_widht': 'first'}).reset_index()
    final_candidates_["label_confidence"] = 0.0
    final_candidates_["wordshape"] = ""
    final_candidates_["wordshape_confidence"] = 0.0

    final_candidates_["Odds"] = final_candidates_['prob_' + field] / (1 - final_candidates_['prob_' + field])
    final_candidates_['model_confidence'] = final_candidates_['prob_' + field].apply(logistic_function)

    final_candidates_['final_confidence_score'] = final_candidates_['model_confidence']
    final_candidates_['vendor_masterdata_present'] = False
    final_candidates_['extracted_from_masterdata'] = False

    final_candidates_.sort_values(['page_num','final_confidence_score'], ascending=[True,False], inplace=True)

    if final_candidates_.shape[0] > 0:
        return {field: final_candidates_.iloc[0].to_dict()}
    else:
        return {field: None}


def procefinal_candidates_rate_with_vendormasterdata(field, DF, vendor_masterdata_score, vendor_masterdata):
    """
    """
    masterdata_upper = {k.upper(): v for k, v in vendor_masterdata.items()}
    label_masterdata = masterdata_upper["LBL" + field.upper()]
    position_label = masterdata_upper[field.upper() + "_POSITION"]
    wordshape = masterdata_upper[field.upper() + "_WORDSHAPE"]

    # Sort by Model Probability and pick top 10 values
    TEMP = DF.loc[DF["prob_" + field] > CONST['model_threshold']]
    # TEMP = DF.loc[DF["predict_label"] == field] # Model Prediction
    if TEMP.shape[0] == 0:
        return {field: None}
    TEMP = TEMP.sort_values("prob_" + field, ascending=False).head(10)

    TEMP["left_text"] = TEMP[["W5Lf", "W4Lf", "W3Lf", "W2Lf", "W1Lf"]].astype(str).agg(' '.join,
                                                                                       axis=1)
    TEMP["above_text"] = TEMP[["W1Ab", "W2Ab", "W3Ab", "W4Ab", "W5Ab"]].astype(str).agg(' '.join,
                                                                                        axis=1)

    TEMP['left_text'] = TEMP['left_text'].astype(str)
    TEMP['above_text'] = TEMP['above_text'].astype(str)

    TEMP['left_text'] = TEMP['left_text'].str.replace('nan', '')
    TEMP['above_text'] = TEMP['above_text'].str.replace('nan', '')

    # Check whether label from Vendor MasterData is present or not
    TEMP["Label_Present"] = False
    if position_label == "Left":
        text_column = TEMP['left_text'].str.replace('[^A-Za-z0-9]+', '').str.upper()
        label_masterdata_temp = re.sub('[^A-Za-z0-9]+', '', str(label_masterdata)).upper()
        TEMP["Label_Present"] = text_column.str.contains(label_masterdata_temp)
    elif position_label == "Above":
        text_column = TEMP['above_text'].str.replace('[^A-Za-z0-9]+', '').str.upper()
        label_masterdata_temp = re.sub('[^A-Za-z0-9]+', '', str(label_masterdata)).upper()
        TEMP["Label_Present"] = text_column.str.contains(label_masterdata_temp)

    TEMP['text'] = TEMP['text'].astype(str)
    final_candidates_ = TEMP.sort_values(['word_num'], ascending=True).groupby(['line_num']).agg(
        {"prob_" + field: 'mean',
         'text': lambda x: "%s" % ' '.join(x),
         'Label_Present': 'sum',
         'word_num': 'count',
         'left': 'min',
         'right': 'max',
         'top': 'min',
         'bottom': 'max',
         'conf': 'mean',
         'page_num': 'first',
         'height': 'first',
         'width': 'first',
         'image_height': 'first',
         'image_widht': 'first'}).reset_index()

    final_candidates_["label_confidence"] = final_candidates_["Label_Present"] / final_candidates_["word_num"]
    final_candidates_["wordshape"] = final_candidates_['text'].apply(get_wordshape)
    final_candidates_["wordshape_confidence"] = final_candidates_['wordshape'].apply(find_similarity_words, b=wordshape)

    final_candidates_["Odds"] = final_candidates_['prob_' + field] / (1 - final_candidates_['prob_' + field])
    final_candidates_['model_confidence'] = final_candidates_['prob_' + field].apply(logistic_function)

    final_candidates_['final_confidence_score'] = final_candidates_['model_confidence']
    final_candidates_['vendor_masterdata_present'] = True
    final_candidates_['extracted_from_masterdata'] = False
    for idx, row in final_candidates_.iterrows():
        if (row['label_confidence'] > 0.70) or (row['wordshape_confidence'] > 0.70):
            final_candidates_.at[idx, 'final_confidence_score']=logistic_function_(row['final_confidence_score'])


    final_candidates_.sort_values(['final_confidence_score'], ascending=False, inplace=True)

    if final_candidates_.shape[0] > 0:
        return {field: final_candidates_.iloc[0].to_dict()}
    else:
        return {field: None}

def candidates_header_without_vendormasterdata(field:str, DF):
    """
    """
    try:
        # Sort by Model Probability and pick top 10 values
        if field in PREDICTION_THRESHOLD:
            TEMP = DF.loc[DF["prob_" + field] > PREDICTION_THRESHOLD[field]]
            # print("getting df greater than threshould prob ",TEMP)
        else:
            TEMP = DF.loc[DF["predict_label"] == field] # Model Prediction
            # print("Picking with model/modified label prediction ",TEMP)
        if not(TEMP.shape[0]):
            return None
        TEMP = TEMP.sort_values("prob_" + field, ascending=False).head(10)

        TEMP["Label_Present"] = False
        final_candidates_ = TEMP.sort_values(['page_num','line_num','word_num'], ascending=[True,True,True]).groupby(['page_num','line_num']).agg(
            {"prob_" + field: 'mean',
            'text': lambda x: "%s" % ' '.join(x),
            'Label_Present': 'sum',
            'word_num': 'count',
            'left': 'min',
            'right': 'max',
            'top': 'min',
            'bottom': 'max',
            'conf': 'mean',
            'height': 'first',
            'width': 'first',
            'image_height': 'first',
            'image_widht': 'first'}).reset_index()

        final_candidates_["label_confidence"] = 0.0
        final_candidates_["wordshape"] = final_candidates_['text'].apply(get_wordshape)
        final_candidates_["wordshape_confidence"] = 0.0

        final_candidates_["Odds"] = final_candidates_['prob_' + field] / (1 - final_candidates_['prob_' + field])
        final_candidates_['model_confidence'] = final_candidates_['prob_' + field].apply(logistic_function)

        final_candidates_['final_confidence_score'] = final_candidates_['model_confidence']
        final_candidates_['vendor_masterdata_present'] = False
        final_candidates_['extracted_from_masterdata'] = False

        final_candidates_.sort_values(['page_num','final_confidence_score'], ascending=[True,False], inplace=True)
        # print("final_candidates_",final_candidates_.shape)
        return final_candidates_
    except:
        print(traceback.print_exc())
        return None
@putil.timing
def check_multiple_invoices(field:str,prediction:dict,DF)->dict:
    try:
        invoiceNumber = prediction.get("invoiceNumber")
        if not(invoiceNumber):
            return prediction
        if invoiceNumber.get("text") != '':
            invoiceNumber = invoiceNumber.get("text")
            wordShape =  prediction.get("invoiceNumber").get("wordshape")
            final_candidates = candidates_header_without_vendormasterdata(field,DF)
            if final_candidates is None:
                return prediction
            final_candidates = final_candidates[final_candidates["wordshape"] == wordShape]
            print("final_candidates sape:",final_candidates.shape)
            uniqueInvNum = list(set(final_candidates["text"]))
            print("uniqueInvNum :",len(uniqueInvNum),uniqueInvNum)
            if len(uniqueInvNum)>1:
                prediction["invoiceNumber"]["multi_invoices"] = 1
            #candidateWordShape = final_candidates
            print("invoiceNumber:",prediction.get("invoiceNumber"))
        print("Checking multiple invoice over!")
        return prediction
    except:
        print("Multiple invoice check exception",traceback.print_exc())
        return prediction 
        

def susp(str_):
    length = len(str(str_))
    string_ = ""
    for i in range(length):
        string_ += "0"
    if length == 0:
        return ""
    return string_


def line_itms(x):
    # print("line_items:", x)
    xx=[]
    for i,j in x.items():
        for q,y in j.items():
            d={}
            d['pageNumber']=str(int(i+1))
            d['rowNumber']=str(q)
            d['fieldset']=[]
            # print(q,y)
            for m in y:
                for it,va in m.items():
                    # print(it,va)
                    # if it in CHECK['amount']:
                    #     checkk=is_amount_(va['text']) or va['text'].isnumeric()
                    #     if checkk==False:
                    #         continue
                    # if it in CHECK['number']:
                    #     checkk1=va['text'].isnumeric()
                    #     if checkk1==False:
                    #         continue
                    # print("Adding Line Item JSON ")
                    f={}
                    f['fieldId']=it
                    if str(va['text']) == "nan":
                        f['fieldValue']=""
                    else:
                        f['fieldValue']=str(va['text'])
                    f['confidence']=round((va['model_confidence']*100),2)
                    f['suspiciousSymbol']=susp(f['fieldValue'])
                    f['boundingBox']={}
                    f['vendorMasterdata']=0
                    f['boundingBox']['left']=math.floor(va['left']*va['image_widht'])
                    f['boundingBox']['right']=math.ceil(va['right']*va['image_widht'])
                    f['boundingBox']['top']=math.floor(va['top']* va['image_height'])
                    f['boundingBox']['bottom']=math.ceil(va['bottom']* va['image_height'])
                    f['OCRConfidence']=round((va['conf']*100),2)
                    f['pageNumber']=str(int(i+1))
                    d['fieldset'].append(f)
            xx.append(d)
    print(xx)
    return xx
def countrybean_ocr_corr(prediction):
    try:
        #here we are correcting countrybean ocr mistakes
        print('kklpo')
        if prediction['vendorName']['text']=='COUNTRY BEAN PVT. LTD':
            if 'Number' in prediction['invoiceNumber']['text']:
                print('kk1')
                prediction['invoiceNumber']['text']=(prediction['invoiceNumber']['text']).split(':')[1]
            if 'Date' in prediction['invoiceDate']['text']:
                prediction['invoiceDate']['text']=(prediction['invoiceDate']['text']).split(':')[1]
                print('kk2')
        return prediction
    except:
        print('exemption')
        return prediction
def INNOVATIVE_foods_corr(prediction):
    try:
        if prediction['vendorGSTIN']['text']=='27AAACI5750N1Z8':
            if ':' in prediction['invoiceDate']['text']:
                prediction['invoiceDate']['text']=parse((prediction['invoiceDate']['text']).split(':')[1]).strftime('%d/%m/%Y')
                print('kk2')
        return prediction
    except:
        print('exeption')
        return prediction
def muddy_puddle_invnum(prediction,DF):
    try:
        if prediction['vendorGSTIN']['text'] in ['27AAKCM0484H1ZA','29AAKCM0484H1Z6']:
            a=''
            b=''
            for row in DF.itertuples():
                if ("tax" in row.left_processed_ngbr.lower()) & ("invoice" in row.left_processed_ngbr.lower()):
                    a=(row.text)
                if ("tax" in row.above_processed_ngbr.lower()) & ("invoice" in row.above_processed_ngbr.lower()):
                    if 'dd/dd' in row.wordshape:
                        b=(row.text)
                        break
            prediction['invoiceNumber']['text']=a+b
            return prediction
        else:
            return prediction
    except:
        print("exeption in muddy_puddle_invnum")
        return prediction
def wordshape(text):
    import re
    t1 = re.sub('[A-Z]', 'X',text)
    t2 = re.sub('[a-z]', 'x', t1)
    return re.sub('[0-9]', 'd', t2)
@putil.timing
def multi_token_invnum(prediction,df):
    prediction['invoiceNumber']['final_confidence_score']=validate_invnum(df)
    return prediction

from datetime import datetime
def search_date_pattern_with_english_keyword(date_invoice_original_text):
    try:
        
        date_invoice_original_text = date_invoice_original_text.upper()
        # Regex pattern to match any month name (abbreviated or full) and a four-digit year
        pattern_month = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

        result_month = [month for month in pattern_month if month in date_invoice_original_text]
        # print(result_month)
        if len(result_month)== 1:
            current_year = datetime.now().year
            current_year_last_two_digits = str(current_year)[-2:]
            if current_year_last_two_digits in date_invoice_original_text:
                print("Pattern Found")
                return True
            else:
                print("Pattern not matched")
                return False
        else:
            print("Month not found")
            return False
        
        # import re
        # # Regex pattern to match any month name (abbreviated or full) and a four-digit year
        # pattern = r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b.*\b\d{4}\b|\b\d{4}\b.*\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b'
        # # Search for the pattern in the text
        # if re.search(pattern, date_invoice_original_text, re.IGNORECASE):
        #     print("Pattern matched")
        #     return True
        # else:
        #     print("Pattern not matched")
        #     return False
    except Exception as e:
        print("Exception occured in search_date_pattern_with_english_keyword", e)
        return False
        
def dates_difference(prediction, grndate, date_format_flag):
    try:
        match_found = False
        from dateutil import parser
        date_invoice_extracted = prediction['invoiceDate']['text']
        date_invoice_original_text = prediction['invoiceDate'].get('original_text')
        if date_invoice_original_text != None:
            match_found = search_date_pattern_with_english_keyword(date_invoice_original_text)
        print("English Keyword Pattern found status", match_found)

        # print("sahil11",match_found, date_format_flag, date_invoice_extracted)
        if (not match_found) and (not date_format_flag):
            ## Check if date is less than 12, mark it as non-stp
            converted_text_1 = parser.parse(date_invoice_extracted, dayfirst=True).date()
            converted_text_2 = parser.parse(date_invoice_extracted, dayfirst=False).date()
            if converted_text_1 != converted_text_2:
                print("Date is less than or equal to 12.")
                return -100
        
        ## 20 August 2024 Removed GRN Date validation after implementation of second check for Invoice Date    
        # # Parse the input dates
        # parsed_date1 = datetime.strptime(grndate, "%Y-%m-%d")
        # parsed_date2 = datetime.strptime(date_invoice_extracted, "%d/%m/%Y")
        # # Calculate the difference
        # date_difference = parsed_date1 - parsed_date2
        return 1
    except Exception as e:
        print("Exception occured", str(e))
        return -5
    
def checking_grnDate_validation(prediction,grndat, date_format_flag):
    prediction_copy = prediction.copy()
    try:
        diff = dates_difference(prediction, grndat, date_format_flag)
        print("difference in days b/w grn days and invoice date is", diff)
        if diff == -100:
            print("Date is less than or equal to 12, so reducing the confidence" )
            prediction['invoiceDate']['final_confidence_score']=0.64
            return prediction
        elif diff == -5:
            print("Exception occured, so reducing the confidence" )
            prediction['invoiceDate']['final_confidence_score']=0.66
            return prediction
        ## 20 August 2024 Removed GRN Date validation after implementation of second check for Invoice Date
        # elif diff in [0,1]: 
        elif diff == 1:
            if prediction.get("invoiceDate").get("final_confidence_score") == 1:
                print("Increasing the confidence since all other conditions are satisfied")
                prediction['invoiceDate']['final_confidence_score']= 1.0
            return prediction
        else:
            print("Date difference is not b/w 0 and 1, so reducing the confidence")
            prediction['invoiceDate']['final_confidence_score']= 0.62
            return prediction
    except:
        print("checking_grnDate_validation exception",traceback.print_exc())
        return prediction_copy
def getInvoicenumber(df,rpainvnum,prediction):
    prediction_copy = prediction.copy()
    try:
        refdat=REFERENCE_DATA[REFERENCE_DATA["review_status"]==1]
        refdat=refdat[refdat["field_name"]=='invoiceNumber']
        invnum =prediction['invoiceNumber']['text']
        vendor_pan = prediction['vendorPAN']['text']
        filtered_refdat = refdat[refdat['vendor_id'].str[2:12] == vendor_pan]
        word_shapes_to_filter = filtered_refdat['field_shape'].dropna().replace('', np.nan).dropna().unique().tolist()
        print(word_shapes_to_filter,"word_shapes_to_filter")
        if wordshape(rpainvnum) in word_shapes_to_filter and  wordshape(invnum) not in word_shapes_to_filter:
            prediction['invoiceNumber']['text'] = rpainvnum
            return prediction
        else:
            if wordshape(invnum) in word_shapes_to_filter:
                print("extracted inv num shape is in ref data")
                return prediction
            else:
                print("taking invnum using ref data")
                prediction = get_inv_num_using_refdata(df,word_shapes_to_filter,prediction)
                return prediction
    except:
        print('there is a exeption in getInvoicenumber',traceback.print_exc())
        return prediction_copy
def modify_invNum_using_ref_rpa(prediction,docMetaData,df):
    prediction_copy = prediction.copy()
    try:
        print(prediction['invoiceNumber']['text'],"1234wdwq")
        if docMetaData.get("result")!= None and docMetaData.get("result").get("document") != None and docMetaData.get("result").get("document").get("docType") == "Discrepancy Note":
            print("Not Modifying Invoice Number for Discr Note")
            return prediction
        if prediction['invoiceNumber']['final_confidence_score']==0.3:
            rpainvnum = prediction['rpaInvoiceNumber']['text']
            #distance = editdistance.eval(invnum, rpainvnum)
            #if distance > 2:
            prediction= getInvoicenumber(df,rpainvnum,prediction)
            if len(prediction['invoiceNumber']['text']) == 0:
                prediction['invoiceNumber']['text']=rpainvnum
            print(prediction['invoiceNumber']['text'],"after")
            return prediction
        return prediction
    except:
        print('there is a exeption in modify_invNum_using_ref_rpa',traceback.print_exc())
        return prediction_copy
@putil.timing
def checking_stp_1(prediction,df,docMetaData,flag_invoice_number):
    try:
        if docMetaData.get("result")!= None and docMetaData.get("result").get("document") != None and docMetaData.get("result").get("document").get("docType") == "Discrepancy Note":
            print("Not checking reference data for Discr Note")
            if flag_invoice_number == True:
                print("Invoice Number matched in Discr note")
                prediction['invoiceNumber']['final_confidence_score']=1.0
            else:
                print("Invoice Number not matched in Discr note. Reducing the confidence")
                prediction['invoiceNumber']['final_confidence_score']=0.6
            return prediction
        print("checking Refence data")
        # df = pd.read_csv(swiggyvendorpath, encoding='unicode_escape')

        #checking stp for invoice number
        if flag_invoice_number == False:
            print("Rpa passed data is not matching, reducing the confidence")
            prediction['invoiceNumber']['final_confidence_score']=0.3
        else:
            print("Checking reference data for Invoice Number")
            a= math.floor(prediction['invoiceNumber']['left'] * prediction['invoiceNumber']['image_widht'])
            b=math.ceil(prediction['invoiceNumber']['right'] * prediction['invoiceNumber']['image_widht'])
            c=math.floor(prediction['invoiceNumber']['top'] * prediction['invoiceNumber']['image_height'])
            d=math.ceil(prediction['invoiceNumber']['bottom'] * prediction['invoiceNumber']['image_height'])
            # filt = df[df["vendor_id"] == prediction['vendorGSTIN']['text']]
            filt = df[df['vendor_id'].str[2:12] == prediction['vendorGSTIN']['text'][2:12]]
            # print(filt,"naswq")
            # filt = df[df["vendor_name"] == prediction['vendorName']['text']]
            filt = filt[filt['field_name']=='invoiceNumber']
            filt = filt[filt['status']==1]
            invnum=prediction['invoiceNumber']['text']
            i=0
            if (filt.shape[0])>0:
                for index,row in filt.iterrows():
                    # print(row,a,b,c,d)
                    # print( (a>=int(row['right'])) , (b<=int(row['left'])) , (c>=int(row['bottom'])),(d<=int(row['top'])) )
                    if ((d<=float(row['top'])) | (a>=float(row['right'])) | (b<=float(row['left'])) | (c>=float(row['bottom']))):
                        i=1+i
                        if (i>=filt.shape[0]):
                            print("reference data did n't matched")
                            prediction['invoiceNumber']['final_confidence_score']=0.4
                    elif wordshape(prediction['invoiceNumber']['text']) in list(filt["field_shape"]):
                        match=1
                        print("reference data matched")
                        prediction['invoiceNumber']['wordshape_confidence']=1
                        prediction['invoiceNumber']['final_confidence_score']=1
                        break
                    else:
                        print("wordshape did not matched")
                        prediction['invoiceNumber']['final_confidence_score']=0.75
                        break
            else:
                prediction['invoiceNumber']['final_confidence_score']=0.5
                print("there is no reference data")
        #invoice_date stp checking
        print("Checking reference data for Invoice Date")
        a= math.floor(prediction['invoiceDate']['left'] * prediction['invoiceDate']['image_widht'])
        b=math.ceil(prediction['invoiceDate']['right'] * prediction['invoiceDate']['image_widht'])
        c=math.floor(prediction['invoiceDate']['top'] * prediction['invoiceDate']['image_height'])
        d=math.ceil(prediction['invoiceDate']['bottom'] * prediction['invoiceDate']['image_height'])
        # filt = df[df["vendor_id"] == prediction['vendorGSTIN']['text']]
        filt = df[df['vendor_id'].str[2:12] == prediction['vendorGSTIN']['text'][2:12]]
        # filt = df[df["vendor_name"] == prediction['vendorName']['text']]
        # print(filt)
        filt = filt[filt['field_name']=='invoiceDate']
        filt = filt[filt['status']==1]
        # 20 Aug 2024 Added second step verification for Invoice Date
        filt = filt[filt["review_status"] == 1]
        i=0
        if (filt.shape[0])>0:
            for index,row in filt.iterrows():
                print("left:",a,"right:",b,"top:",c,"bottom",d)
                print("int(row['right']):",float(row['right']),"int(row['left'])",float(row['left']),"float(row['bottom'])",float(row['bottom']),"float(row['top']))",float(row['top']))
                print( (a>=int(row['right'])) , (b<=int(row['left'])) , (c>=int(row['bottom'])),(d<=int(row['top'])) )
                if ((a>=float(row['right'])) | (b<=float(row['left'])) | (c>=float(row['bottom'])) | (d<=float(row['top']))):
                    i=i+1
                    if (i>=filt.shape[0]):
                        # prediction['invoiceDate']['wordshape_confidence']=0.4
                        prediction['invoiceDate']['final_confidence_score']=0.4
                        print("Reduce dt confidence")
                elif wordshape(prediction['invoiceDate']['text']) == row['field_shape']:
                    prediction['invoiceDate']['wordshape_confidence']=1
                    prediction['invoiceDate']['final_confidence_score']=1
                    print('kko')
                    break
                else:
                    #prediction['invoiceDate']['wordshape_confidence']=0.75
                    print("bbox matching but word shape not matching :")
                    prediction['invoiceDate']['final_confidence_score']=0.65
                    break
        else:
            prediction['invoiceDate']['final_confidence_score']=0.5
            print("there is no reference data")
        return prediction
    except:
        print('there is a exeption',traceback.print_exc())
        return prediction
def combining_bounding_box_stp(prediction, original_df, reference_df,docMetaData,flag_invoice_number):
    try:
        if docMetaData.get("result")!= None and docMetaData.get("result").get("document") != None and docMetaData.get("result").get("document").get("docType") == "Discrepancy Note":
            print("Not checking reference data for Discr Note")
            if flag_invoice_number == True:
                print("Invoice Number matched in Discr note")
                prediction['invoiceNumber']['final_confidence_score']=1.0
            else:
                print("Invoice Number not matched in Discr note. Reducing the confidence")
                prediction['invoiceNumber']['final_confidence_score']=0.6
            return prediction
        print("checking Refence data")
        """if flag_invoice_number == False:
            print("Flag for invoice number is False")
            a= math.floor(prediction['invoiceNumber']['left'] * prediction['invoiceNumber']['image_widht'])
            b=math.ceil(prediction['invoiceNumber']['right'] * prediction['invoiceNumber']['image_widht'])
            c=math.floor(prediction['invoiceNumber']['top'] * prediction['invoiceNumber']['image_height'])
            d=math.ceil(prediction['invoiceNumber']['bottom'] * prediction['invoiceNumber']['image_height'])
            filt = df[df["vendor_id"] == prediction['vendorGSTIN']['text']]
            # filt = df[df["vendor_name"] == prediction['vendorName']['text']]
            filt = filt[filt['field_name']=='invoiceNumber']
            filt = filt[filt['status']==1]
            invnum=prediction['invoiceNumber']['text']
            i=0
            if (filt.shape[0])>0:
                for index,row in filt.iterrows():
                    #print(row,a,b,c,d)
                    #print( (a>=int(row['right'])) , (b<=int(row['left'])) , (c>=int(row['bottom'])),(d<=int(row['top'])) )
                    if ((d<=float(row['top'])) | (a>=float(row['right'])) | (b<=float(row['left'])) | (c>=float(row['bottom']))):
                        i=1+i
                        if (i>=filt.shape[0]):
                            if wordshape(invnum) != row['field_shape']:
                                prediction['invoiceNumber']['final_confidence_score']=0.4
                            elif wordshape(invnum) == row['field_shape']:
                                if (('X' in wordshape(invnum)) & ('d' in wordshape(invnum)) & (len(invnum)>=5)):
                                    prediction['invoiceNumber']['final_confidence_score']=0.8
                                else:
                                    prediction['invoiceNumber']['final_confidence_score']=0.6
                    elif wordshape(prediction['invoiceNumber']['text']) == row['field_shape']:
                        match=1
                        prediction['invoiceNumber']['wordshape_confidence']=1
                        prediction['invoiceNumber']['final_confidence_score']=1
                        break
                    elif ( (len(prediction['invoiceNumber']['text'])>=5)) :
                        if (('X' in wordshape(prediction['invoiceNumber']['text'])) & ('d' in wordshape(prediction['invoiceNumber']['text']))):
                            prediction['invoiceNumber']['final_confidence_score']=0.85
                        else:
                            prediction['invoiceNumber']['final_confidence_score']=0.75
                        break
            else:
                prediction['invoiceNumber']['final_confidence_score']=0.5
                print("there is no reference data")
        else:
            print("Flag for invoice number is True. Not checking reference data")
        """
        # print("sahil inside new function")
        print("Prediction", prediction["vendorGSTIN"]["text"])
        reference_filt = reference_df[reference_df["vendor_id"] == prediction['vendorGSTIN']['text']]
        reference_filt = reference_filt[reference_filt['field_name']=='invoiceDate']
        reference_filt = reference_filt[reference_filt['status']==1]
        if (reference_filt.shape[0])>0:
            group_column_values = reference_filt['field_shape'].unique()
            grouped_data = reference_filt.groupby('field_shape').agg({'bottom': 'max', 'left': 'min','right': 'max', 'top': 'min'})
            # print("sahil",grouped_data)
            for group_name, group_data in grouped_data.iterrows():
                agg_bottom = group_data["bottom"]
                agg_left = group_data["left"]
                agg_right = group_data["right"]
                agg_top = group_data["top"]
                o_a= math.floor(prediction['invoiceDate']['left'] * prediction['invoiceDate']['image_widht'])
                o_b=math.ceil(prediction['invoiceDate']['right'] * prediction['invoiceDate']['image_widht'])
                o_c=math.floor(prediction['invoiceDate']['top'] * prediction['invoiceDate']['image_height'])
                o_d=math.ceil(prediction['invoiceDate']['bottom'] * prediction['invoiceDate']['image_height'])
                print(f"pred file bottom {o_d} left {o_a} right {o_b} top {o_c}")
                print("Aggregated float(row['top'])", agg_top, "int(row['right']):", agg_right , "int(row['left'])",agg_left, "float(row['bottom'])",agg_bottom)
                print(((agg_top>=float(o_d)), (agg_right<=float(o_a)), (agg_left>=float(o_b)), agg_bottom<=float(o_c)))
                if ( (agg_top>=float(o_d)) | (agg_right<=float(o_a)) | (agg_left>=float(o_b)) | (agg_bottom<=float(o_c))):
                    print("Bounding box is not matcing")
                    prediction['invoiceDate']['final_confidence_score']=0.4
                elif wordshape(prediction['invoiceDate']['text']) == group_name:
                    print("Field shape is matching and also the bbox")
                    prediction['invoiceDate']['final_confidence_score']=1
                    break
                else:
                    print("bbox matching but word shape not matching :")
                    prediction['invoiceDate']['final_confidence_score']=0.65
        else:
            prediction['invoiceDate']['final_confidence_score']=0.45
            print(f"there is no reference data for GSTIN {prediction['vendorGSTIN']['text']}")
            print("---------------------------------------")  
        return prediction
    except:
        print('there is a exeption',traceback.print_exc())
        return prediction
@putil.timing
def removing_unnecessary_fields(final_prediction,doc_type):
    final_prediction_copy = final_prediction.copy()
    if doc_type.lower() == "invoice":
        doc_type = "DEFAULT"
    try:
        STP_CONFIGURATION = putil.getSTPConfiguration()
        STP_CONFIGURATION = STP_CONFIGURATION.get(doc_type)
        #list_stp_config = list(STP_CONFIGURATION.keys())
        list_stp_config = [key for key,val in STP_CONFIGURATION.items() if (val["display_flag"] == 1)]
        tmp_dict = final_prediction.copy()
        for item_, value_ in tmp_dict.items():
            if value_ is not None and item_ not in list_stp_config and item_ !="lineItemPrediction":
                final_prediction.pop(item_,"Key Not Found")
        return final_prediction
    except:
        print("removing_unnecessary_fields exception",traceback.print_exc())
        return final_prediction_copy 
@putil.timing
def build_final_json(prediction):
    print("Start build_final_json")
    json_obj = {}
    json_obj['documentInfo'] = []
    for item_, value_ in prediction.items():
        #print (item_)
        #print(value_)
        if item_ != 'lineItemPrediction':
            #print("Processing ", item_ ,"\t:",value_)
            if value_ is not None:
                dictiory = {}
                dictiory['fieldId'] = item_
                if str(value_['text']) == "nan" :
                    dictiory['fieldValue'] = ""
                else:
                    dictiory['fieldValue'] = str(value_['text'])
                dictiory['confidence'] = round((value_['final_confidence_score']*100),2)
                dictiory['suspiciousSymbol'] = susp(dictiory['fieldValue'])
                dictiory['boundingBox'] = {}
                dictiory['boundingBox']['left'] = math.floor(value_['left'] * value_['image_widht'])
                dictiory['boundingBox']['right'] = math.ceil(value_['right'] * value_['image_widht'])
                dictiory['boundingBox']['top'] = math.floor(value_['top'] * value_['image_height'])
                dictiory['boundingBox']['bottom'] = math.ceil(value_['bottom'] * value_['image_height'])
                dictiory['pageNumber'] = str(value_['page_num'] + 1)
                dictiory['OCRConfidence'] = round((value_['conf']*100),2)
                dictiory['mandatory'] = value_["mandatory"]
                if (value_['extracted_from_masterdata'] == True):
                    dictiory['vendorMasterdata'] = 1
                else:
                    dictiory['vendorMasterdata'] = 0
                if value_.get("extracted_from_entitydata"):
                    if (value_['extracted_from_entitydata'] == True):
                        dictiory['entityMasterdata'] = 1
                    else:
                        dictiory['entityMasterdata'] = 0
                else:
                    dictiory['entityMasterdata'] = 0            
                if value_.get("calculated_field"):
                    if (value_['calculated_field'] == True):
                        dictiory['calculated_field'] = 1
                    else:
                        dictiory['calculated_field'] = 0
                else:
                    dictiory['calculated_field'] = 0 
                if value_.get("multi_invoices"):
                    dictiory["multi_invoices"] = value_.get("multi_invoices")
                else:
                    dictiory["multi_invoices"] = 0
                if item_ in ["invoiceNumber","invoiceDate"]:
                   dictiory["isReferenceDataPresent"] = value_.get("isReferenceDataPresent")
                   dictiory["isNewLayout"] = value_.get("isNewLayout")
                if item_ == "vendorCode":
                    dictiory["dropDown"] = value_.get("dropDown")
                    dictiory["dropDownOptions"] = value_.get("dropDownOptions")
                json_obj['documentInfo'].append(dictiory)  
        elif item_ == 'lineItemPrediction':
            json_obj["documentLineItems"]=line_itms(value_)
    return json_obj

def score_value(prediction, vendor_masterdata,docMetaData):
    """
    """
    if docMetaData.get("result")!= None and docMetaData.get("result").get("document") != None and docMetaData.get("result").get("document").get("docType") != None:
        doc_type = docMetaData.get("result").get("document").get("docType")
    stp_config = get_stp_configuration(vendor_masterdata,doc_type)
    # print("STP_CONFIGURATION:", stp_config)
    stp_fields = {key:val for key,val in stp_config.items() if (val["display_flag"] == 1
        and val["minimum_confidence"] > 0)}
    # print("STP_FIELDS:", stp_fields)
    critical_score=0
    ovr_score = 0
    critical_score_count=0
    ovr_score_count=0
    for item_, value_ in prediction.items():
        if (item_ != "lineItemPrediction") & (value_ is not None):
            confidence= value_['final_confidence_score']
            if item_ in stp_fields.keys():
                critical_score += confidence
                critical_score_count += 1
            ovr_score += confidence
            ovr_score_count+=1
        elif (item_ == "lineItemPrediction") & (value_ is not None):
            for page, page_prediction in value_.items():
                for row, row_prediction in page_prediction.items():
                    for line_items in row_prediction:
                        LI_field = str(list(line_items.keys())[0])
                        extraction_confidence = line_items[LI_field]["model_confidence"]
                        if ("LI_" + LI_field) in stp_fields:
                            critical_score += extraction_confidence
                            critical_score_count += 1
                        ovr_score += extraction_confidence
                        ovr_score_count+=1

    avg_ovr_score = 0
    avg_critical_score = 0
    try:
        avg_ovr_score=round(ovr_score/ovr_score_count,4)
    except ZeroDivisionError as e:
        print("No header items in invoice predicted")
        pass
    try:
        avg_critical_score=round(critical_score/critical_score_count,4)
    except ZeroDivisionError as e:
        print("No critical fields in invoice predicted")
        pass

    print("Critical Score:", avg_critical_score)
    print("Overall Score:", avg_ovr_score)
    return avg_ovr_score,avg_critical_score

def update_stp_configuration(default_config, vendor_specific_config):
    """
    """
    for k, v in vendor_specific_config.items():
        if isinstance(v, collections.abc.Mapping):
            default_config[k] = update_stp_configuration(default_config.get(k, {}), v)
        else:
            default_config[k] = v
    return default_config
def get_stp_configuration(vendor_masterdata,doc_type = "DEFAULT"):
    """
    """
    if doc_type.lower() == "invoice":
        doc_type = "DEFAULT"
    
    STP_CONFIGURATION = putil.getSTPConfiguration()
    default_config = STP_CONFIGURATION[doc_type]
    if vendor_masterdata is not None:
        vendor_id = vendor_masterdata['VENDOR_ID']
        if vendor_id in STP_CONFIGURATION:
            vendor_specific_config = STP_CONFIGURATION[vendor_id]
            default_config = update_stp_configuration(default_config, vendor_specific_config)
    return default_config
@putil.timing
def check_subtotal_zero(stp, final_prediction):
    if stp:
        if final_prediction.get("subTotal_0%") != None and final_prediction.get("subTotal_0%").get("text")!= None:
            if final_prediction.get("subTotal_0%").get("text") > 0:
                print("Making STP as False since subtotal zero percentage is greater than zero")
                stp = False
    return stp
@putil.timing
def check_stp(prediction,vendor_masterdata,docMetaData):
    """
    Derive STP Flag
    """
    doc_type = "invoice"
    if docMetaData.get("result")!= None and docMetaData.get("result").get("document") != None and docMetaData.get("result").get("document").get("docType") != None:
        doc_type = docMetaData.get("result").get("document").get("docType")
    # if doc_type == "Discrepancy Note":
    #     print("Not setting STP for Discrepnacy Note:")
    #     return 0
    stp_config = get_stp_configuration(vendor_masterdata,doc_type)
    stp_fields = {key:val for key,val in stp_config.items() if (val["display_flag"] == 1
        and val["minimum_confidence"] > 0)}
    # Get the list of the extracted fields from prediction.items;
    # Compare with the list of fieds to be extracted(If len(list of fields expected)>len(list of fields extracted), stp=0)
    # Get Extracted fields
    header_extracted_fields = set()
    for item_, value_ in prediction.items():
        if (item_ != "lineItemPrediction") & (value_ is not None):
            header_extracted_fields.add(item_)

    print("Header Extracted Fields:", header_extracted_fields)

    LI_extracted_fields_rows = [] # All lineitems should have the required fields
    for item_, value_ in prediction.items():
        if (item_ == 'lineItemPrediction') & (value_ is not None):

            for page, page_prediction in value_.items():


                for row, row_prediction in page_prediction.items():
                    row_fields = set()
                    for line_items in row_prediction:
                        LI_field = str(list(line_items.keys())[0])
                        row_fields.add("LI_" + LI_field)
                    LI_extracted_fields_rows.append(row_fields)

    LI_extracted_fields = set()
    if len(LI_extracted_fields_rows) > 0:
        LI_extracted_fields = LI_extracted_fields_rows[0]


    for v in LI_extracted_fields_rows:
        LI_extracted_fields = LI_extracted_fields.intersection(v)
    print("LI Extracted Fields:", LI_extracted_fields)

    extracted_fields = set()
    extracted_fields.update(header_extracted_fields)
    extracted_fields.update(LI_extracted_fields)
    print("Extracted Fields:", extracted_fields)
    stp = 1
    mandatory_fields = set(stp_fields.keys())
    print("Mandatory Fields:", mandatory_fields)
    if len(mandatory_fields - extracted_fields) > 0:
        print("Missing Required STP Fields:", mandatory_fields - extracted_fields)
        stp = 0
        return 0
    else:
        print("All required STP Fields are extracted")
        print("Proceeding to chcek individual confidence")

    # Check confidence of individual fields
    for item_, value_ in prediction.items():
        if (item_ != "lineItemPrediction") & (value_ is not None):
            extraction_confidence= value_['final_confidence_score']
            if item_ in stp_fields.keys():
                if item_ in stp_fields:
                    minimum_confidence = stp_fields[item_]["minimum_confidence"]
                    print(item_, extraction_confidence, minimum_confidence)
                    if extraction_confidence < minimum_confidence:
                        print("No STP, as confidence of", item_, "is",
                            str(extraction_confidence), "which is less than minimum confidence of",
                            str(minimum_confidence))
                        stp = 0
                        return 0
        elif (item_ == "lineItemPrediction") & (value_ is not None):
            for page, page_prediction in value_.items():
                for row, row_prediction in page_prediction.items():
                    for line_items in row_prediction:
                        LI_field = str(list(line_items.keys())[0])
                        extraction_confidence = line_items[LI_field]["model_confidence"]
                        if ("LI_" + LI_field) in stp_fields:
                            minimum_confidence = stp_fields["LI_" + LI_field]["minimum_confidence"]
                            print(LI_field, extraction_confidence, minimum_confidence)
                            if extraction_confidence < minimum_confidence:
                                print("No STP, as confidence of", LI_field, "in lineItem is",
                                    str(extraction_confidence), "which is less than minimum confidence of",
                                    str(minimum_confidence))
                                stp = 0
                                return 0
    # if stp > 0:
    #     stp = True
    # else:
    #     stp = False
    # if stp and doc_type == "Invoice":
    #     print("In post-processor checking for stp")
    #     if stp == True and docMetaData.get("result")!= None and docMetaData.get("result").get("document") != None and docMetaData.get("result").get("document").get("documentId")!= None:
    #         documentId = docMetaData.get("result").get("document").get("documentId")
    #         from client_rules import bizRuleValidateForUi as biz_rl
    #         from client_rules import isMasterDataPresent
    #         from client_rules import check_field_stp_score,check_multi_invoices_stp
    #         callbackUrl = cfg.getUIServer()
    #         result = biz_rl(documentId, callbackUrl)
    #         print("Result after biz rule validations:",result)
    #         if result is not None:
    #             if len(result) > 0:
    #                 stp = False
    #             else:
    #                 isMasterDataPresent = isMasterDataPresent(documentId,
    #                                                       cfg.getUIServer())
    #                 stp = isMasterDataPresent
    #                 print("isMasterDataPresent stp flag :",isMasterDataPresent)
    #                 if stp:
    #                 # "invoiceDate","invoiceNumber","totalAmount" confidence check
    #                     invdt_stp = check_field_stp_score(documentId,
    #                                                         cfg.getUIServer())
    #                     print("date field stp Check :",invdt_stp)
    #                     stp = invdt_stp
    #                 if stp:
    #                     multi_invoices = check_multi_invoices_stp(documentId,
    #                                                         cfg.getUIServer())
    #                     print("Mult invoice stp check :",multi_invoices)
    #                     stp = multi_invoices
    #         else:
    #             print("result from biz rule validations were None.")
    # if stp == True:
    #     stp = 1
    # else:
    #     stp = 0
    return stp



def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try:
        parse(string, fuzzy=fuzzy)
        return True

    except Exception as e:
        return False


def is_amount_(text):
    """
    Checks whether passed string is valid amount or not
    Returns: 1 if amount, 0 otherwise
    """
    try:
        ptn1 = "[0-9]{1,3}[,]{1}[0-9]{3}[,]{1}[0-9]{3}[,]{1}[0-9]{3}[.]{1}[0-9]{1,4}"
        ptn2 = "[0-9]{1,3}[,]{1}[0-9]{3}[,]{1}[0-9]{3}[.]{1}[0-9]{1,4}"
        ptn3 = "[0-9]{1,3}[,]{1}[0-9]{3}[.]{1}[0-9]{1,4}"
        ptn4 = "[0-9]{1,3}[.]{1}[0-9]{1,4}"
        ptn5 = "[0-9]{1,3}[.]{1}[0-9]{3}[.]{1}[0-9]{3}[.]{1}[0-9]{3}[,]{1}[0-9]{1,4}"
        ptn6 = "[0-9]{1,3}[.]{1}[0-9]{3}[.]{1}[0-9]{3}[,]{1}[0-9]{1,4}"
        ptn7 = "[0-9]{1,3}[.]{1}[0-9]{3}[,]{1}[0-9]{1,4}"
        ptn8 = "[0-9]{1,3}[,]{1}[0-9]{1,4}"
        ptns = [ptn1,ptn2,ptn3,ptn4,ptn5,ptn6,ptn7,ptn8]
        for ptn in ptns:
            l = re.findall(ptn,text)
            l1 = [g for g in l if len(g) > 0]
            if len(l1) == 1:
                p = parse_price(text)
                if p.amount is not None:
                    return True
    except:
        return False
    return False

@putil.timing
def correct_duplicate_prediction(final_prediction):
    """
    Removes exact matching prediction based on bounding box
    """

    group_amount = ['totalAmount', 'taxAmount', 'subTotal','SGSTAmount', 'CGSTAmount', 'IGSTAmount','freightAmount','discountAmount']
    group_header_fields = ['invoiceNumber'] #, 'poNumber', 'paymentTerms'] # removed those fields which swiggy not required. which creating ambiguty in prediction
    group_date_fields = ['invoiceDate', 'dueDate']

    dict_group_amount = {}
    dict_group_header_fields = {}
    dict_group_date_fields = {}

    for key, val in final_prediction.items():
        if val is not None:
            if key in group_amount:
                bounding_box = (val['left'], val['right'], val['top'], val['bottom'])
                final_confidence = val['final_confidence_score']
                if bounding_box in dict_group_amount:
                    list_duplicate_fileds = dict_group_amount[bounding_box]
                    list_duplicate_fileds.append((key, final_confidence))
                    dict_group_amount[bounding_box] = list_duplicate_fileds
                else:
                    dict_group_amount[bounding_box] = [(key, final_confidence)]
            elif key in group_header_fields:
                bounding_box = (val['left'], val['right'], val['top'], val['bottom'])
                final_confidence = val['final_confidence_score']
                if bounding_box in dict_group_header_fields:
                    list_duplicate_fileds = dict_group_header_fields[bounding_box]
                    list_duplicate_fileds.append((key, final_confidence))
                    dict_group_header_fields[bounding_box] = list_duplicate_fileds
                else:
                    dict_group_header_fields[bounding_box] = [(key, final_confidence)]
            elif key in group_date_fields:
                bounding_box = (val['left'], val['right'], val['top'], val['bottom'])
                final_confidence = val['final_confidence_score']
                if bounding_box in dict_group_date_fields:
                    list_duplicate_fileds = dict_group_date_fields[bounding_box]
                    list_duplicate_fileds.append((key, final_confidence))
                    dict_group_date_fields[bounding_box] = list_duplicate_fileds
                else:
                    dict_group_date_fields[bounding_box] = [(key, final_confidence)]

    fields_to_remove = []
    for key, val in dict_group_amount.items():
        if len(val) > 1:
            field_with_max_confidence = max(val, key=lambda item:item[1])
            list_probable_duplicates = [v[0] for v in val]
            list_probable_duplicates.remove(field_with_max_confidence[0])
            fields_to_remove.extend(list_probable_duplicates)

    for key, val in dict_group_header_fields.items():
        if len(val) > 1:
            field_with_max_confidence = max(val, key=lambda item:item[1])
            list_probable_duplicates = [v[0] for v in val]
            list_probable_duplicates.remove(field_with_max_confidence[0])
            fields_to_remove.extend(list_probable_duplicates)

    for key, val in dict_group_date_fields.items():
        if len(val) > 1:
            field_with_max_confidence = max(val, key=lambda item:item[1])
            list_probable_duplicates = [v[0] for v in val]
            list_probable_duplicates.remove(field_with_max_confidence[0])
            fields_to_remove.extend(list_probable_duplicates)

    for k in fields_to_remove:
        final_prediction.pop(k, None)

    return final_prediction

def remove_special_charcters(s):
    """
    """
    a = re.sub(r"[^a-zA-Z ]+", '', s)
    if a == '':
        return s
    else:
        return a

def form_table(DF):
    """
    """
    # Convert text to string dtype
    DF['text'] = DF['text'].astype("str")
    # Define Line Item Area for each page from Model Prediction and line_row
    TEMP = DF.loc[(DF['ROW_NUM'] > 0)]


    # Extract final value for each page and row
    pages = list(TEMP['page_num'].unique())
    pages.sort()
    print(DF[["TABLE_COL_NUM", "TABLE_COL"]].drop_duplicates())
    dict_unique_cols = dict(zip(TEMP["TABLE_COL_NUM"],TEMP["TABLE_COL"]))
    col_order = sorted(dict_unique_cols.keys())
    print("dict_unique_cols:", dict_unique_cols)
    unique_cols = list(TEMP["TABLE_COL"].unique())
    extracted_table = {}
    print(TEMP[['page_num', 'ROW_NUM','text','TABLE_COL']])
    for p in pages:
        page_prediction = {}
        PAGE = TEMP.loc[TEMP['page_num'] == p]
        PAGE.sort_values(['line_num', 'word_num'], ascending=[True, True], inplace=True)
        line_rows = list(PAGE['ROW_NUM'].unique())
        line_rows.sort()
        print(p, line_rows)
        for l in line_rows:
            row_prediction = []
            for cc in col_order:
                f = dict_unique_cols[cc]
                FIELD_PREDICTION = PAGE.loc[(PAGE['ROW_NUM'] == l) & ((PAGE['TABLE_COL'] == f))]
                if FIELD_PREDICTION.shape[0] == 0:
                    continue

                pred = FIELD_PREDICTION.groupby(['ROW_NUM']).agg({'text': lambda x: "%s" % ' '.join(x),
                        'left': 'min',
                        'right': 'max',
                        'top': 'min',
                        'bottom': 'max',
                        'conf': 'mean',
                        'image_height': 'first',
                        'image_widht': 'first'}).reset_index()
                dict_row_ = pred.iloc[0].to_dict()
                del dict_row_['ROW_NUM']
                conf_var = np.random.uniform(0.7, 0.8)
                dict_row_["model_confidence"] = conf_var
                dict_row_["prediction_probability"] = conf_var
                row_prediction.append({f: dict_row_})

            page_prediction[l] = row_prediction
        extracted_table[p] = page_prediction

    print("Form Table:", extracted_table)
    return extracted_table

@putil.timing
def extract_table(DF, vendor_masterdata_score, vendor_masterdata):
    """
    """
    import traceback
    print("____________________")
    print('extract_table')
    ll = list(vendor_masterdata.keys())
    print(vendor_masterdata)

    vendor_id = vendor_masterdata['VENDOR_ID']
    global TABLE_CUSTOM_FIELD
    TABLE_CUSTOM_FIELD = pd.read_csv(tableFieldPath, encoding = 'unicode_escape')
    print(vendor_id)
    print(TABLE_CUSTOM_FIELD)
    TABLES = TABLE_CUSTOM_FIELD.loc[TABLE_CUSTOM_FIELD['VENDOR_ID'] == vendor_id]
    TABLES.dropna(inplace=True)

    #TABLES['FOOTER'] = TABLES['FOOTER'].astype(str).apply(remove_special_charcters).str.upper()
    TABLES['FOOTER'] = TABLES['FOOTER'].astype(str).str.upper()
    TABLES['FOOTER'] = TABLES['FOOTER'].str.replace(" ","")

    #TABLES['HEADER'] = TABLES['HEADER'].astype(str).apply(remove_special_charcters).str.upper()
    TABLES['HEADER'] = TABLES['HEADER'].astype(str).str.upper()
    TABLES['HEADER'] = TABLES['HEADER'].str.replace(" ","")

    # TABLES['COLUMN_HEADER'] = TABLES['COLUMN_HEADER'].astype(str).apply(remove_special_charcters).str.upper()
    # TABLES['COLUMN_HEADER'] = TABLES['COLUMN_HEADER'].str.replace(" ","")

    #DF['LINE_TEXT'] = DF['line_text'].astype(str).apply(remove_special_charcters).str.upper()
    DF['LINE_TEXT'] = DF['line_text'].astype(str).str.upper()
    DF['LINE_TEXT'] = DF['LINE_TEXT'].str.replace(" ","")


    # Populate Table Name and Column Names
    DF["TABLE_NAME"] = "NONE"
    DF["TABLE_COL"] = "NONE"
    DF["TABLE_COL_NUM"] = 0
    DF["ROW_NUM"] = 0

    tables_to_extract = list(TABLES["TABLE_NAME"].unique())
    for table in tables_to_extract:
        try:
            print("Extracting Table:", table)
            pages = list(DF["page_num"].unique())
            TABLE_INFO = TABLES.loc[TABLES["TABLE_NAME"] == table]
            print(TABLE_INFO[["FOOTER"]])
            FOOTER_INFO = TABLE_INFO.loc[TABLE_INFO["FOOTER"] != "NOTAPPLICABLE"]
            if FOOTER_INFO.shape[0] != 1:
                raise Exception('Multiple/No Footer Info rows found')
            footer = dict(FOOTER_INFO.iloc[0])["FOOTER"]

            # Header is optional
            HEADER_INFO = TABLE_INFO.loc[TABLE_INFO["HEADER"] != "NOTAPPLICABLE"]
            header = None
            if HEADER_INFO.shape[0] > 0:
                header = dict(HEADER_INFO.iloc[0])["HEADER"]

            for p in pages:
                print("************* PAGE NO:", p)
                DF_PAGE = DF.loc[DF["page_num"] == p]
                print("Table Footer:", footer)
                print("Table Header:", header)
                dict_footer = {}
                if isfloat(footer):
                    print("Pre-defined footer as:", footer)
                    dict_footer = {'page_num': p, 'LINE_TEXT': 'PRE_DEFINED',
                    'line_top': float(footer), 'FOOTER_SCORE': 1.0}
                elif footer == "PAGE_BOUNDARY":
                    print("End of the page as Footer!!!")
                    dict_footer = {'page_num': p, 'LINE_TEXT': 'PAGE_BOUNDARY',
                    'line_top': 1.0, 'FOOTER_SCORE': 1.0}
                else:
                    DF_PAGE["FOOTER_SCORE"] = DF_PAGE['LINE_TEXT'].apply(find_similarity_words,
                        b=footer)
                    DF_PAGE["FOOTER_SCORE_EXACT"] = 0
                    try:
                        DF_PAGE.loc[DF_PAGE['LINE_TEXT'].str.contains(footer),"FOOTER_SCORE_EXACT"] = 1
                    except:
                        pass

                    DF_PAGE["FOOTER_SCORE"] = DF_PAGE[["FOOTER_SCORE", "FOOTER_SCORE_EXACT"]].max(axis=1)
                    DF_FOOTER = DF_PAGE.loc[DF_PAGE["FOOTER_SCORE"] >= 0.6]
                    DF_FOOTER.sort_values(["FOOTER_SCORE"], ascending=[False], inplace=True)
                    print(DF_FOOTER[["LINE_TEXT", "FOOTER_SCORE"]])
                    if DF_FOOTER.shape[0] == 0:
                        continue
                    dict_footer = dict(DF_FOOTER.iloc[0][["page_num", "LINE_TEXT", "line_top",
                        "FOOTER_SCORE"]])

                dict_header = {}
                if header is not None:
                    DF_PAGE["HEADER_SCORE"] = DF_PAGE['LINE_TEXT'].apply(find_similarity_words,
                        b=header)

                    DF_PAGE["HEADER_SCORE_EXACT"] = 0
                    try:
                        DF_PAGE.loc[DF_PAGE['LINE_TEXT'].str.contains(header),"HEADER_SCORE_EXACT"] = 1
                    except:
                        pass

                    DF_PAGE["HEADER_SCORE"] = DF_PAGE[["HEADER_SCORE", "HEADER_SCORE_EXACT"]].max(axis=1)
                    DF_HEADER = DF_PAGE.loc[DF_PAGE["HEADER_SCORE"] >= 0.6]
                    DF_HEADER.sort_values(["HEADER_SCORE"], ascending=[False], inplace=True)
                    if DF_HEADER.shape[0] == 0:
                        continue
                    dict_header = dict(DF_HEADER.iloc[0][["page_num", "LINE_TEXT", "line_down",
                        "HEADER_SCORE"]])

                print("HEADER:", dict_header)
                print("FOOTER:", dict_footer)

                if len(dict_header) > 0:
                    DF_PAGE = DF_PAGE.loc[DF_PAGE["top"] >= dict_header["line_down"]*0.99]

                if len(dict_footer) > 0:
                    DF_PAGE = DF_PAGE.loc[DF_PAGE["bottom"] <= dict_footer["line_top"]*1.01]

                D_ = TABLE_INFO.loc[TABLE_INFO["ANCHOR_COLUMN_ALIGNMENT"] != "Not Applicable"]
                anchor_type = dict(D_.iloc[0])["ANCHOR_COLUMN_ALIGNMENT"]

                TABLE_COLUMNS = TABLE_INFO.loc[TABLE_INFO["COLUMN_HEADER"] != "Not Applicable"]
                list_cols_metadata = []
                print(list(DF_PAGE["LINE_TEXT"].unique()))
                for idx, row in TABLE_COLUMNS.iterrows():
                    DF_PAGE["COL_SCORE"] = 0
                    d_ = dict(row)
                    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
                    print("Extraction starts for:", d_)
                    col_name = d_["COLUMN_NAME"]
                    col_header = d_["COLUMN_HEADER"]
                    col_num = int(d_["COLUMN_NO"])
                    anchor_col = d_["ANCHOR_COLUMN"]
                    col_type = d_["COLUMN_TYPE"]
                    header_alignment = d_["TABLE_HEADER_ALIGNMENT"]
                    value_alignment = d_["TABLE_VALUE_ALIGNMENT"]
                    shift_units = d_["HEADER_SHIFT_UNITS"]
                    print(col_header)
                    len_ = len(str(col_header).split())

                    col_header = str(col_header).upper()

                    DF_PAGE["COL_SCORE_FUZZY"] = DF_PAGE['LINE_TEXT'].apply(find_similarity_words,
                        b=col_header)
                    DF_PAGE["COL_SCORE_EXACT"] = 0
                    try:
                        if len(col_header) >= 4:
                            DF_PAGE.loc[DF_PAGE['LINE_TEXT'].str.contains(col_header),"COL_SCORE_EXACT"] = 1
                    except:
                        pass

                    DF_PAGE["COL_SCORE"] = DF_PAGE[["COL_SCORE_FUZZY", "COL_SCORE_EXACT"]].max(axis=1)

                    DF_COL = DF_PAGE.loc[DF_PAGE["COL_SCORE"] >= 0.8]
                    DF_COL["LINE_LENGTH"] = DF_COL["LINE_TEXT"].str.len()
                    DF_COL.sort_values(["COL_SCORE", "LINE_LENGTH", "line_num"], ascending=[False, True, True], inplace=True)
                    print(DF_COL[["page_num", "line_num", "LINE_TEXT", "COL_SCORE"]])
                    if DF_COL.shape[0] > 0:
                        d_ = dict(DF_COL.iloc[0][["page_num", "line_num", "LINE_TEXT",
                            "line_top", "line_down", "line_left", "line_right",
                            "COL_SCORE"]])
                        print(d_, len_)
                        D_ = DF_PAGE.loc[(DF_PAGE["page_num"] == d_["page_num"]) &
                        (DF_PAGE["line_num"] == d_["line_num"])]
                        D_.sort_values(['word_num'], ascending=[True], inplace=True)
                        print(D_[["page_num", "line_num", "line_text", "text", "left", "right"]])
                        list_texts = list(D_["text"])
                        list_left = list(D_["left"])
                        list_right = list(D_["right"])
                        print("LIST TEXTS")
                        print(list_texts)
                        len_ = min(len_, len(list_texts))
                        list_text_search = [' '.join(list_texts[i: i + len_]) for i in range(len(list_texts)- len_ + 1)]
                        print(list_text_search)
                        list_text_search = [str(x).upper() for x in list_text_search]
                        list_score = [find_similarity_words(s, col_header) for s in list_text_search]
                        print(list_score)
                        max_score_index = list_score.index(max(list_score))

                        print("LEFT:", list_left)
                        print("RIGHT:", list_right)

                        list_left_search = [list_left[i: i + len_][0] for i in range(len(list_left)- len_ + 1)]
                        list_right_search = [list_right[i: i + len_][-1] for i in range(len(list_right)- len_ + 1)]

                        print("LEFT:", list_left_search)
                        print("RIGHT:", list_right_search)

                        left_b = list_left_search[max_score_index]
                        right_b = list_right_search[max_score_index]

                        d_["line_left"] = left_b
                        d_["line_right"] = right_b
                        d_["col_name"] = col_name
                        d_["col_header"] = col_header
                        d_["col_num_original"] = col_num
                        d_["anchor_col"] = anchor_col
                        d_["col_type"] = col_type
                        d_["col_type"] = col_type
                        d_["header_alignment"] = header_alignment
                        d_["value_alignment"] = value_alignment
                        d_["shift_units"] = shift_units
                        print("^^^^^^^^^^^^^^^^")
                        print("Adding:", d_)
                        # Check whether the line already identified as a column
                        # with higher score
                        add_column = True
                        list_delete = []
                        for l_ in list_cols_metadata:
                            if ((l_['page_num'] == d_['page_num']) & (l_['line_num'] == d_['line_num']) 
                                & (l_['LINE_TEXT'] == d_['LINE_TEXT'])):
                                if l_['COL_SCORE'] > d_['COL_SCORE']:
                                    # Column already identified with higher score
                                    # Don't add the new column
                                    add_column = False
                                    break
                                elif l_['COL_SCORE'] < d_['COL_SCORE']:
                                    # Older identified column has lower score
                                    # Delete the older identified column
                                    list_delete.append(l_)
                                elif l_['COL_SCORE'] == d_['COL_SCORE']:
                                    # Older identified column has same score
                                    # Keep the column whose header length is higher and delete the older one
                                    print("Inside equal score case!!!!")
                                    print(l_, d_)
                                    if len(str(d_["col_header"])) > len(str(l_["col_header"])):
                                        list_delete.append(l_)

                        list_cols_metadata = [a for a in list_cols_metadata if a not in list_delete]
                        if add_column:
                            list_cols_metadata.append(d_)

                col_bottoms = []
                cols_to_delete = []
                for col in list_cols_metadata:
                    print(col)
                    print("&&&&&&&&&&&&&&")
                    # Add code to remove identified cols which are not in sync with Footer
                    if col["page_num"] != dict_footer["page_num"]:
                        cols_to_delete.append(col["col_num_original"])
                    else:
                        col_bottoms.append(col["line_down"])

                for col in list_cols_metadata:
                    if col["col_num_original"] in cols_to_delete:
                        list_cols_metadata.remove(col)


                print(col_bottoms)
                col_bottoms.sort()
                print(col_bottoms)
                indices = [i + 1 for (x, y, i) in zip(col_bottoms, col_bottoms[1:],
                    range(len(col_bottoms))) if CONST['top_variance'] < abs(x - y)]
                # indices = [i + 1 for (x, y, i) in zip(col_bottoms, col_bottoms[1:],
                #     range(len(col_bottoms))) if 0.015 < abs(x - y)]
                result = [col_bottoms[start:end] for start, end in zip([0] + indices, indices + [len(col_bottoms)])]
                print(result)

                cols_to_keep = []
                for r in result:
                    print("XXXX:", cols_to_keep)
                    if len(cols_to_keep) < len(r):
                        cols_to_keep = r
                    print("XXXX Updated:", cols_to_keep)

                cols_to_delete = []
                for i in range(len(list_cols_metadata)):
                    col = list_cols_metadata[i]
                    to_delete = True
                    for v in cols_to_keep:
                        if math.isclose(v, col["line_down"]):
                            to_delete = False
                            break
                    if to_delete:
                        cols_to_delete.append(col["col_num_original"])

                print("Columns to Delete:", cols_to_delete)
                list_cols_metadata = [col for col in list_cols_metadata if not col["col_num_original"] in cols_to_delete]

                print("Shifting column headers:", list_cols_metadata)
                list_cols_metadata = sorted(list_cols_metadata,
                    key=operator.itemgetter('line_left'))

                col_num_ = 1
                for col in list_cols_metadata:
                    col["col_num"] = col_num_
                    col_num_ += 1

                print("Shifting column headers:", list_cols_metadata)

                # Code added for shift in column header and values
                for col in list_cols_metadata:
                    if ((col["header_alignment"] == "LEFT") & (col["value_alignment"] == "RIGHT")
                        | (col["header_alignment"] == "LEFT") & (col["value_alignment"] == "MIDDLE")):
                        # Expand Header till next column
                        print("AMIT:", col)
                        current_col_num = col["col_num"]
                        next_col_num = col["col_num"] + 1
                        new_right = 1.0
                        for c_ in list_cols_metadata:
                            if c_["col_num"] == next_col_num:
                                new_right = c_["line_left"] * 0.99
                                break
                        col["line_right"] = new_right
                        print("AMIT:", col)
                    elif ((col["header_alignment"] == "RIGHT") & (col["value_alignment"] == "LEFT")
                        | (col["header_alignment"] == "RIGHT") & (col["value_alignment"] == "MIDDLE")):
                        # Expand Header till previous column
                        print("AMIT:", col)
                        current_col_num = col["col_num"]
                        previous_col_num = col["col_num"] - 1
                        new_left = 0.0
                        for c_ in list_cols_metadata:
                            if c_["col_num"] == previous_col_num:
                                new_left = c_["line_right"] * 1.01
                                break
                        col["line_left"] = new_left
                        print("AMIT:", col)
                    elif ((col["header_alignment"] == "MIDDLE") & (col["value_alignment"] == "LEFT")
                        | (col["header_alignment"] == "MIDDLE") & (col["value_alignment"] == "RIGHT")):
                        # Add Left/Right Shift Parameter for Header
                        print("AMIT:", col)
                        shift = float(col["shift_units"])
                        if col["value_alignment"] == "LEFT":
                            col["line_left"] = col["line_left"] - (shift *
                                (col["line_right"] - col["line_left"]))
                        elif col["value_alignment"] == "RIGHT":
                            col["line_right"] = col["line_right"] + (shift *
                                (col["line_right"] - col["line_left"]))
                        print("AMIT:", col)


                table_start = 1.0
                table_end = dict_footer["line_top"]
                for col in list_cols_metadata:
                    table_start = min(table_start, col["line_down"])

                table_start = table_start * 0.99
                table_end = table_end * 1.01
                list_cols_metadata = sorted(list_cols_metadata, key=operator.itemgetter('col_num'))
                print("COLUMN METADATA:", list_cols_metadata, table_start, table_end, anchor_type)

                DF.loc[((DF['page_num'] == p) & (DF['top'] >= table_start)
                    & (DF['bottom'] <= table_end)), "TABLE_NAME"] = table

                for col in list_cols_metadata:
                    col_left = col["line_left"] * 0.99
                    col_right = col["line_right"] * 1.01
                    print(col["col_name"], col_left, col_right)
                    if col["col_type"] == "MULTI":
                        DF.loc[((DF['page_num'] == p) & (DF["TABLE_NAME"] == table) & (DF["TABLE_COL"] == "NONE")
                            & (col_left <= DF['line_right']) & (col_right >= DF['line_left'])),
                        "TABLE_COL"] = col["col_name"]
                        DF.loc[(DF['page_num'] == p) & (DF["TABLE_NAME"] == table) 
                        & (DF['TABLE_COL'] == col["col_name"]), "TABLE_COL_NUM"] = col["col_num_original"]
                    else:
                        DF.loc[((DF['page_num'] == p) & (DF["TABLE_NAME"] == table) & (DF["TABLE_COL"] == "NONE")
                            & (col_left <= DF['right']) & (col_right >= DF['left'])),
                        "TABLE_COL"] = col["col_name"]
                        DF.loc[(DF['page_num'] == p) & (DF["TABLE_NAME"] == table) 
                        & (DF['TABLE_COL'] == col["col_name"]), "TABLE_COL_NUM"] = col["col_num_original"]

                print(DF.loc[(DF['page_num'] == p)
                    & (DF["TABLE_NAME"] != "NONE")][["text", "TABLE_COL", "left",
                    "right"]])
                # Populate Row Num based on anchor column and type
                print("Identifying anchor column:", list_cols_metadata)
                anchor = None
                for col in list_cols_metadata:
                    if col["anchor_col"] == "YES":
                        if DF.loc[(DF['page_num'] == p) & (DF["TABLE_COL"] == col["col_name"])].shape[0] > 0:
                            anchor = col["col_name"]

                if anchor is None:
                    print("As no anchor column found on page, continue to next page!!!")
                    continue

                # Continue if very few columns are found
                if len(list_cols_metadata) <=1:
                    continue

                DF["vertical_center"] = DF[['top', 'bottom']].mean(axis=1)

                print("ANCHOR COLUMN and TYPE:", anchor, anchor_type)
                print(DF["TABLE_COL"].unique())
                print(DF.loc[(DF['page_num'] == p)
                    & (DF["TABLE_COL"] == anchor)][["text", "top", "bottom",
                    "vertical_center"]])

                row_centers = list(DF.loc[(DF['page_num'] == p)
                    & (DF["TABLE_COL"] == anchor)]["vertical_center"])
                row_tops = list(DF.loc[(DF['page_num'] == p)
                    & (DF["TABLE_COL"] == anchor)]["top"])
                row_bottoms = list(DF.loc[(DF['page_num'] == p)
                    & (DF["TABLE_COL"] == anchor)]["bottom"])
                if len(row_centers) == 0:
                    print("As no data in anchor column, continue to next page")
                    continue
                row_centers.sort()
                row_tops.sort()
                row_bottoms.sort()
                print(row_centers)
                indices = [i + 1 for (x, y, i) in zip(row_centers, row_centers[1:],
                    range(len(row_centers))) if 0.01 < abs(x - y)]
                result = [row_centers[start:end] for start, end in zip([0] + indices, indices + [len(row_centers)])]
                result_tops = [row_tops[start:end] for start, end in zip([0] + indices, indices + [len(row_tops)])]
                result_bottoms = [row_bottoms[start:end] for start, end in zip([0] + indices, indices + [len(row_bottoms)])]
                print(result)

                dict_row_centers = {}
                dict_row_tops = {}
                dict_row_bottoms = {}
                row_num = 1
                for idx,r in enumerate(result):
                    dict_row_centers[row_num] = sum(r) / len(r)
                    dict_row_tops[row_num] = sum(result_tops[idx])/len(result_tops[idx])
                    dict_row_bottoms[row_num] = sum(result_bottoms[idx])/len(result_bottoms[idx])
                    row_num += 1

                print("Row Center, Top & Bottom:")
                print(dict_row_centers)
                print(dict_row_tops)
                print(dict_row_bottoms)

                row_top = table_start
                for r_num, r_center in dict_row_centers.items():
                    print(r_num, r_center)
                    if anchor_type == "NONE":
                        # DF.loc[((DF['page_num'] == p) & (DF["TABLE_NAME"] == table)
                        # & (DF['top'] <= r_center) & (DF['bottom'] >= r_center)),
                        # "ROW_NUM"] = r_num
                        DF.loc[((DF['page_num'] == p) & (DF["TABLE_NAME"] == table)
                        & (DF['top'] <= dict_row_bottoms[r_num]) 
                        & (DF['bottom'] >= dict_row_tops[r_num])),
                        "ROW_NUM"] = r_num
                    elif anchor_type == "BOTTOM":
                        DF.loc[((DF['page_num'] == p) & (DF["TABLE_NAME"] == table)
                        & (DF['top'] >= row_top) & (DF['top'] <= r_center)),
                        "ROW_NUM"] = r_num
                        row_top = r_center
                    elif anchor_type == "TOP":
                        DF.loc[((DF['page_num'] == p) & (DF["TABLE_NAME"] == table)
                        & (DF['bottom'] >= r_center)),
                        "ROW_NUM"] = r_num
                    elif anchor_type == "MIDDLE":
                        previous_row_num = r_num-1
                        next_row_num = r_num+1
                        new_row_bottom = 1
                        new_row_top = 0
                        if (previous_row_num in dict_row_bottoms):
                            new_row_top = (dict_row_bottoms[previous_row_num] 
                                + dict_row_tops[r_num])/2
                        if (next_row_num in dict_row_tops):
                            new_row_bottom = (dict_row_bottoms[r_num] 
                                + dict_row_tops[next_row_num])/2

                        print("MIDDLE:", r_num, new_row_top, new_row_bottom)
                        DF.loc[((DF['page_num'] == p) & (DF["TABLE_NAME"] == table)
                        & (DF['top'] >= new_row_top) 
                        & (DF['bottom'] <= new_row_bottom)),
                        "ROW_NUM"] = r_num

                print("Table Info on Page:", p)
                print(DF.loc[(DF['page_num'] == p)
                    & (DF["TABLE_NAME"] != "NONE") & (DF["TABLE_COL"] == "ToIPlus")][["page_num",
                    "text", "TABLE_COL", "TABLE_COL_NUM", "ROW_NUM"]])

            print("Calling form_table:")
            print(DF.loc[(DF["TABLE_NAME"] != "NONE") & (DF["TABLE_COL"] != "NONE")][["page_num", "line_num", "ROW_NUM", "text", "TABLE_COL"]])
            return form_table(DF.loc[(DF["TABLE_NAME"] != "NONE") & (DF["TABLE_COL"] != "NONE")])

        except Exception as e:
            print(e)
            traceback.print_exc()
            pass
            return {}
    return {}

'''
def extract_vendor_specific_extra_fields(DF, vendor_masterdata_score, vendor_masterdata):
    """
    Write code to extract Vendor Specific Extra Fields
    """
    # Anjan: Get custom field and information from new sheet based on VENDOR_ID

    print("____________________")
    print('extract_vendor_specific_extra_fields')
    ll = list(vendor_masterdata.keys())
    print(vendor_masterdata)

    vendor_id = vendor_masterdata['VENDOR_ID']
    # Code added to test Template creation
    global VENDOR_SPECIFIC_FIELD
    VENDOR_SPECIFIC_FIELD = pd.read_csv(customFieldPath, encoding='unicode_escape')
    print(vendor_id)
    print(VENDOR_SPECIFIC_FIELD)
    TEMP = VENDOR_SPECIFIC_FIELD.loc[VENDOR_SPECIFIC_FIELD['VENDOR_ID'] == vendor_id]
    TEMP.dropna(inplace=True)

    print(TEMP)
    dict_fields_to_extract = {}
    if not TEMP.empty:
        for idx, row in TEMP.iterrows():
            field = row['Custom Field Name']
            label = row['Custom Field Label']
            pos = row['Custom Field Label Position']
            shape = row['Custom Field Shape']
            loc = row['Custom Field Location']
            default_val = row['Default Value']
            horizontal_anchor = row['Horizontal Anchor']
            vertical_anchor = row['Vertical Anchor']
            # Code added for location based extraction
            top = row['TOP']
            bottom = row['BOTTOM']
            left = row['LEFT']
            right = row['RIGHT']
            top_delimiter = row['Top Delimiter']
            bottom_delimiter = row['Bottom Delimiter']
            include_top = row['INCLUDE_TOP']
            include_bottom = row['INCLUDE_BOTTOM']
            page_identifier = row['PAGE_IDENTIFIER']
            dict_fields_to_extract[field] = (label, pos, shape, loc, default_val, horizontal_anchor,
                vertical_anchor, top, bottom, left, right, top_delimiter, bottom_delimiter,include_top,include_bottom,
                page_identifier)


    DF['Left_1'] = DF['W1Lf'].astype(str).str.upper().replace('NAN','')
    DF['Left_2'] = DF['W2Lf'].astype(str).str.upper().replace('NAN','')
    DF['Left_3'] = DF['W3Lf'].astype(str).str.upper().replace('NAN','')
    DF['Left_4'] = DF['W4Lf'].astype(str).str.upper().replace('NAN','')
    DF['Left_5'] = DF['W5Lf'].astype(str).str.upper().replace('NAN','')

    DF['Abv_1'] = DF['W1Ab'].astype(str).str.upper().replace('NAN','')
    DF['Abv_2'] = DF['W2Ab'].astype(str).str.upper().replace('NAN','')
    DF['Abv_3'] = DF['W3Ab'].astype(str).str.upper().replace('NAN','')
    DF['Abv_4'] = DF['W4Ab'].astype(str).str.upper().replace('NAN','')
    DF['Abv_5'] = DF['W5Ab'].astype(str).str.upper().replace('NAN','')

    DF['LINE_TEXT'] = DF['line_text'].astype(str).apply(remove_special_charcters).str.upper()
    DF['LINE_TEXT'] = DF['LINE_TEXT'].str.replace(" ","")

    #DF['TEXT_ABOVE'] = DF["Abv_5"] + DF["Abv_4"] + DF["Abv_3"] + DF["Abv_2"] + DF["Abv_1"]
    DF['TEXT_ABOVE'] = DF["Abv_1"] + DF["Abv_2"] + DF["Abv_3"] + DF["Abv_4"] + DF["Abv_5"]
    DF['TEXT_LEFT'] = DF["Left_4"] + DF["Left_3"] + DF["Left_2"] + DF["Left_1"]

    DF['TEXT_ABOVE'] = DF['TEXT_ABOVE'].astype(str).apply(remove_special_charcters).str.upper()
    DF['TEXT_LEFT'] = DF['TEXT_LEFT'].astype(str).apply(remove_special_charcters).str.upper()

    results = {}
    for f, v in dict_fields_to_extract.items():
        print("*********************************************************")
        print("extracting",f)
        label = v[0]
        pos = v[1]
        shape = v[2]
        loc = v[3]
        default_val = v[4]
        horizontal_anchor = v[5]
        vertical_anchor = v[6]
		# Code added for location based extraction
        top = v[7]
        bottom = v[8]
        left = v[9]
        right = v[10]
        top_delimiter = v[11]
        bottom_delimiter = v[12]
        include_top = v[13]
        include_bottom = v[14]
        page_identifier = v[15]

        label = label.upper()
        list_labels = label.split()
        list_labels = [remove_special_charcters(x) for x in list_labels]
        string_label = ''.join(list_labels)

        horizontal_anchor = horizontal_anchor.upper()
        list_horizontal_anchor = horizontal_anchor.split()
        list_horizontal_anchor = [remove_special_charcters(x) for x in list_horizontal_anchor]
        horizontal_anchor = ''.join(list_horizontal_anchor)

        vertical_anchor = vertical_anchor.upper()
        list_vertical_anchor = vertical_anchor.split()
        list_vertical_anchor = [remove_special_charcters(x) for x in list_vertical_anchor]
        vertical_anchor = ''.join(list_vertical_anchor)

        extracted_value = None

        # Add filtering based on location
        coordinates = LOCATION_COORDINATES[loc]
        TEMP = DF.loc[(DF['left'] >= coordinates[0]) & (DF['right'] <= coordinates[1]) &
        (DF['top'] >= coordinates[2]) & (DF['bottom'] <= coordinates[3])]

        # Add filtering based on delimiters
        # Search for delimiters
        top_delimiter = top_delimiter.upper()
        list_top_delimiter = top_delimiter.split()
        list_top_delimiter = [remove_special_charcters(x) for x in list_top_delimiter]
        top_delimiter = ' '.join(list_top_delimiter)

        bottom_delimiter = bottom_delimiter.upper()
        list_bottom_delimiter = bottom_delimiter.split()
        list_bottom_delimiter = [remove_special_charcters(x) for x in list_bottom_delimiter]
        bottom_delimiter = ' '.join(list_bottom_delimiter)

        if (top_delimiter != "NOT APPLICABLE") or (bottom_delimiter != "NOT APPLICABLE"):
            print("Inside Delimiter Case")
            print(top_delimiter, bottom_delimiter)
            top_delimit = 0.0
            bottom_delimit = 1.0
            page_num_min = 0
            page_num_max = TEMP['page_num'].max()

            print(top_delimit, bottom_delimit, page_num_min, page_num_max)

            TEMP_ = TEMP.copy()
            if top_delimiter != "NOT APPLICABLE":
                TEMP_["top_delimiter_score"] = TEMP_['LINE_TEXT'].apply(find_similarity_words,
                    b=top_delimiter)
                TEMP_ = TEMP_.loc[TEMP_['top_delimiter_score'] > 0.8]
                if TEMP_.shape[0] > 0:
                    TEMP_.sort_values(['top_delimiter_score'], ascending=[False],
                        inplace=True)
                    top_delimit = TEMP_.iloc[0]['line_down']
                    page_num_min = TEMP_.iloc[0]['page_num']

            TEMP_ = TEMP.copy()
            if bottom_delimiter != "NOT APPLICABLE":
                TEMP_["bottom_delimiter_score"] = TEMP_['LINE_TEXT'].apply(find_similarity_words,
                    b=bottom_delimiter)
                TEMP_ = TEMP_.loc[TEMP_['bottom_delimiter_score'] > 0.8]
                if TEMP_.shape[0] > 0:
                    TEMP_.sort_values(['bottom_delimiter_score'], ascending=[False],
                        inplace=True)
                    bottom_delimit = TEMP_.iloc[0]['line_top']
                    page_num_max = TEMP_.iloc[0]['page_num']

            print(top_delimit, bottom_delimit, page_num_min, page_num_max)
            TEMP = TEMP.loc[(TEMP['page_num'] >= page_num_min)
            & (TEMP['page_num'] <= page_num_max)
            & (TEMP['line_top'] >= top_delimit)
            & (TEMP['line_down'] <= bottom_delimit)]



        ### Find Score based on Left Value
        if pos == 'Left':
            TEMP["surrounding_text_score"] = TEMP['TEXT_LEFT'].apply(find_similarity_words,
             b=string_label)
            TEMP["wordshape_score"] = TEMP['wordshape'].apply(find_similarity_words, b=shape)
        elif pos == 'Above':
            TEMP["surrounding_text_score"] = TEMP['TEXT_ABOVE'].apply(find_similarity_words,
             b=string_label)
            TEMP["wordshape_score"] = TEMP['wordshape'].apply(find_similarity_words, b=shape)
        elif (pos == 'ANCHOR_LEFT_TOP') | (pos == "ANCHOR_LEFT_BOTTOM") | (pos == 'ANCHOR_RIGHT_TOP') | (pos == "ANCHOR_RIGHT_BOTTOM"):
            TEMP["surrounding_text_score"] = 0.0
            TEMP["wordshape_score"] = 0.0

            TEMP_ANCHOR = DF.copy()
            print("Inside ANCHOR Label CASE!!!!!!!", horizontal_anchor, vertical_anchor)
            TEMP_ANCHOR["horizontal_anchor_score"] = TEMP_ANCHOR['LINE_TEXT'].apply(find_similarity_words,
             b=horizontal_anchor)
            TEMP_ANCHOR["vertical_anchor_score"] = TEMP_ANCHOR['LINE_TEXT'].apply(find_similarity_words,
             b=vertical_anchor)

            # print(TEMP.sort_values(['left_anchor_score'], ascending=[False])[['LINE_TEXT', 'left_anchor_score', 'line_top', 'line_down','line_left', 'line_right']])
            # print(TEMP.sort_values(['top_anchor_score'], ascending=[False])[['LINE_TEXT', 'top_anchor_score', 'line_top', 'line_down','line_left', 'line_right']])

            TEMP_HORIZONTAL_ANCHOR = TEMP_ANCHOR.loc[(TEMP_ANCHOR['horizontal_anchor_score'] > 0.70)][['LINE_TEXT', 'horizontal_anchor_score', 'line_top', 'line_down','line_left', 'line_right']]
            TEMP_HORIZONTAL_ANCHOR.drop_duplicates(inplace=True)
            TEMP_VERTICAL_ANCHOR = TEMP_ANCHOR.loc[(TEMP_ANCHOR['vertical_anchor_score'] > 0.70)][['LINE_TEXT', 'vertical_anchor_score', 'line_left', 'line_right','line_top', 'line_down']]
            TEMP_VERTICAL_ANCHOR.drop_duplicates(inplace=True)
            print("CHecking Template")
            print(TEMP_VERTICAL_ANCHOR)
            print(TEMP_HORIZONTAL_ANCHOR)

            if (TEMP_HORIZONTAL_ANCHOR.shape[0] > 0) & (TEMP_VERTICAL_ANCHOR.shape[0] > 0):
                TEMP_HORIZONTAL_ANCHOR.sort_values(['horizontal_anchor_score'], ascending=[False], inplace=True)

                TEMP_VERTICAL_ANCHOR.sort_values(['vertical_anchor_score'], ascending=[False], inplace=True)
                print(TEMP_VERTICAL_ANCHOR)
                print(TEMP_HORIZONTAL_ANCHOR)

                candidate_found = False
                # TEMP = pd.DataFrame()
                print(TEMP.shape)
                TEMP_ = TEMP.copy()
                dummyDF = pd.DataFrame(columns = list(TEMP.columns))
                for idx_, row_ in TEMP_HORIZONTAL_ANCHOR.iterrows():
                    top_boundary = float(row_['line_top'])
                    bottom_boundary = float(row_['line_down'])
                    horizontal_cut = 0
                    horizontal_anchor_score = float(row_['horizontal_anchor_score'])
                    for idx__,row__ in TEMP_VERTICAL_ANCHOR.iterrows():
                        left_boundary = float(row__['line_left'])
                        right_boundary = float(row__['line_right'])
                        vertical_cut = 0
                        vertical_anchor_score = float(row__['vertical_anchor_score'])
                        TEMP = TEMP_.copy()
                        if pos == "ANCHOR_LEFT_TOP":
                            horizontal_cut = float(row_['line_right'])
                            vertical_cut = float(row__['line_down'])
                            TEMP = (TEMP.loc[(TEMP['top'] >= vertical_cut)
                                & (TEMP['left'] >= horizontal_cut)])
                        elif pos == "ANCHOR_LEFT_BOTTOM":
                            horizontal_cut = float(row_['line_right'])
                            vertical_cut = float(row__['line_top'])
                            TEMP = (TEMP.loc[(TEMP['bottom'] <= vertical_cut)
                                & (TEMP['left'] >= horizontal_cut)])
                        elif pos == "ANCHOR_RIGHT_TOP":
                            horizontal_cut = float(row_['line_left'])
                            vertical_cut = float(row__['line_down'])
                            TEMP = (TEMP.loc[(TEMP['top'] >= vertical_cut)
                                & (TEMP['right'] <= horizontal_cut)])
                        elif pos == "ANCHOR_RIGHT_BOTTOM":
                            horizontal_cut = float(row_['line_left'])
                            vertical_cut = float(row__['line_top'])
                            TEMP = (TEMP.loc[(TEMP['bottom'] <= vertical_cut)
                                & (TEMP['right'] <= horizontal_cut)])

                        TEMP = (TEMP.loc[(top_boundary <= TEMP['bottom'])
                            & (TEMP['top'] <= bottom_boundary)
                            & (left_boundary <= TEMP['right'])
                            & (TEMP['left'] <= right_boundary)])

                        if TEMP.shape[0] > 0:
                            TEMP['anchor_left'] = left_boundary
                            TEMP['anchor_right'] = right_boundary
                            TEMP['anchor_top'] = top_boundary
                            TEMP['anchor_bottom'] = bottom_boundary

                            TEMP['min_left'] = TEMP[['left','anchor_left']].max(axis=1)
                            TEMP['min_right'] = TEMP[['right','anchor_right']].min(axis=1)
                            TEMP['min_top'] = TEMP[['top','anchor_top']].max(axis=1)
                            TEMP['min_bottom'] = TEMP[['bottom','anchor_bottom']].min(axis=1)

                            TEMP['overlap_horizontal'] = TEMP['min_right'] - TEMP['min_left']
                            TEMP['overlap_vertical'] = TEMP['min_bottom'] - TEMP['min_top']

                            TEMP['overlap_score_horizontal'] = (TEMP['overlap_horizontal']
                                /TEMP['width'])
                            TEMP['overlap_score_vertical'] = (TEMP['overlap_vertical']
                                /TEMP['height'])

                            TEMP['overlap_score_vertical'] = TEMP['overlap_score_vertical'] * horizontal_anchor_score
                            TEMP['overlap_score_horizontal'] = TEMP['overlap_score_horizontal'] * vertical_anchor_score

                            TEMP["wordshape_score"] = TEMP['wordshape'].apply(find_similarity_words, b=shape)
                            TEMP = TEMP.loc[TEMP['wordshape_score'] > 0.60]
                            if TEMP.shape[0] > 0:
                                TEMP['surrounding_text_score'] = TEMP[['overlap_score_horizontal'
                                ,'overlap_score_vertical']].mean(axis=1)
                                dummyDF = pd.concat([dummyDF, TEMP], axis = 0)
                # print("CHECKING TEMPLAATE 2: \n", dummyDF.sort_values(['surrounding_text_score'], ascending=[False])[['text',
                                                            # 'surrounding_text_score','wordshape_score','overlap_score_horizontal'
                                                            # ,'overlap_score_vertical']])
                if dummyDF.shape[0]>0:
                    TEMP = dummyDF.sort_values(['surrounding_text_score'], ascending=[False])
                    TEMP = TEMP.reset_index()
                    #TEMP = TEMP.loc[[0]]
                    #TEMP['surrounding_text_score'] = 1.0
                else:
                    TEMP["surrounding_text_score"] = 0.0
                    TEMP["wordshape_score"] = 0.0
        elif pos == 'LOCATION':
            # Code added for Location based extraction
            print("Inside Location Based Extraction")
            try:
                top = float(top)*0.95
                bottom = float(bottom)*1.05
                left = float(left)*0.95
                right = float(right)*1.05

                TEMP = (TEMP.loc[(top <= TEMP['bottom'])
                    & (TEMP['top'] <= bottom)
                    & (left <= TEMP['right'])
                    & (TEMP['left'] <= right)])
                if TEMP.shape[0] > 0:
                    TEMP['top_boundary'] = top
                    TEMP['bottom_boundary'] = bottom
                    TEMP['left_boundary'] = left
                    TEMP['right_boundary'] = right

                    TEMP['min_top'] = TEMP[['top','top_boundary']].max(axis=1)
                    TEMP['min_bottom'] = TEMP[['bottom','bottom_boundary']].min(axis=1)
                    TEMP['min_left'] = TEMP[['left','left_boundary']].max(axis=1)
                    TEMP['min_right'] = TEMP[['right','right_boundary']].min(axis=1)

                    TEMP['overlap_area'] = ((TEMP['min_right'] - TEMP['min_left'])*
                        (TEMP['min_bottom'] - TEMP['min_top']))

                    TEMP['area'] = ((TEMP['right'] - TEMP['left'])*
                        (TEMP['bottom'] - TEMP['top']))

                    TEMP["wordshape_score"] = TEMP['wordshape'].apply(find_similarity_words, b=shape)
                    TEMP = TEMP.loc[TEMP['wordshape_score'] > 0.60]
                    if TEMP.shape[0] > 0:
                        TEMP['surrounding_text_score'] = TEMP['overlap_area']/TEMP['area']
                    else:
                        TEMP["surrounding_text_score"] = 0.0
                else:
                    TEMP["surrounding_text_score"] = 0.0
                    TEMP["wordshape_score"] = 0.0
            except:
                TEMP["surrounding_text_score"] = 0.0
                TEMP["wordshape_score"] = 0.0
        elif pos == 'MULTI_TOKEN':
            # Code added for Location based extraction
            print("Multi Token Extraction")
            try:
                proceed_for_extraction = True
                page_num = None
                top_b = 0
                bottom_b = 1
                left_b = 0
                right_b = 1

                page_identifier = str(page_identifier)
                page_identifier = page_identifier.strip()

                location_provided = False
                try:
                    page_num = int(page_identifier) - 1
                    top_b = float(str(top).strip())
                    bottom_b = float(str(bottom).strip())
                    left_b = float(str(left).strip())
                    right_b = float(str(right).strip())
                    location_provided = True
                except:
                    pass

                skip_top = False
                skip_bottom = False
                skip_left = False
                skip_right = False
                if isfloat(str(top).strip()):
                    top_b = float(str(top).strip())
                    skip_top = True

                if isfloat(str(bottom).strip()):
                    bottom_b = float(str(bottom).strip())
                    skip_bottom = True

                if isfloat(str(left).strip()):
                    left_b = float(str(left).strip())
                    skip_left = True

                if isfloat(str(right).strip()):
                    right_b = float(str(right).strip())
                    skip_right = True

                print(location_provided,"^^&&")
                page_identifier = page_identifier.upper()
                print("Page Identifier:", page_identifier)
                if (not location_provided) & (page_identifier != "NOT APPLICABLE"):
                    try:
                        page_num = int(page_identifier) - 1
                        TEMP = TEMP.loc[TEMP['page_num'] == page_num]
                    except:
                        page_identifier_list = re.split(r'[^\w]', str(page_identifier))
                        print("Page Identifier List:", page_identifier_list)
                        unique_pages = list(TEMP['page_num'].unique())
                        print("Unique Pages:", unique_pages)
                        page_identifier_match = {}
                        for p in unique_pages:
                            l = list(TEMP.loc[TEMP['page_num'] == p]['text'])
                            l = [str(x) for x in l]
                            s = " ".join(l)
                            s = s.upper()
                            matches = [x for x in page_identifier_list if x in s]
                            page_identifier_match[p] = len(matches) / len(page_identifier_list)
                        print("Page Identifier Match Scores:", page_identifier_match)
                        predicted_page = max(page_identifier_match.items(), key=operator.itemgetter(1))[0]
                        max_score = page_identifier_match[predicted_page]
                        if max_score < 0.6:
                            proceed_for_extraction = False
                        else:
                            TEMP = TEMP.loc[TEMP['page_num'] == predicted_page]
                            page_num = predicted_page
                        
                print(top, bottom, left, right)
                top_marker = remove_special_charcters(str(top)).upper()
                bottom_marker = remove_special_charcters(str(bottom)).upper()
                left_marker = remove_special_charcters(str(left)).upper()
                right_marker = remove_special_charcters(str(right)).upper()

                top_marker = top_marker.replace(" ","")
                bottom_marker = bottom_marker.replace(" ","")
                left_marker = left_marker.replace(" ","")
                right_marker = right_marker.replace(" ","")
                print(top_marker, bottom_marker, left_marker, right_marker)
                

                if (not location_provided) & (not skip_top) & (top_marker != "PAGEBOUNDARY"):
                    TEMP["top_marker_score"] = TEMP['LINE_TEXT'].apply(find_similarity_words,
                        b=top_marker)
                    TEMP["top_marker_score_exact"] = 0
                    TEMP.loc[TEMP['LINE_TEXT'].str.contains(top_marker),"top_marker_score_exact"] = 1
                    TEMP["top_marker_score"] = TEMP[["top_marker_score", "top_marker_score_exact"]].max(axis=1)

                    TEMP.sort_values(['top_marker_score', 'page_num'], ascending=[False, True], inplace=True)
                    print("top_marker:",top_marker)
                    print(TEMP[["page_num", "LINE_TEXT", "top_marker_score", "line_down"]])
                    d_ = dict(TEMP.iloc[0])
                    if d_["top_marker_score"] > 0.6:
                        if include_top == "YES":
                            top_b = d_["line_top"] * 0.99
                        else:
                            top_b = d_["line_down"] * 0.99
                        page_num = d_["page_num"]
                    else:
                        proceed_for_extraction = False
                if (not location_provided) & (not skip_bottom) & (bottom_marker != "PAGEBOUNDARY"):
                    TEMP["bottom_marker_score"] = TEMP['LINE_TEXT'].apply(find_similarity_words,
                        b=bottom_marker)
                    TEMP["bottom_marker_score_exact"] = 0
                    TEMP.loc[TEMP['LINE_TEXT'].str.contains(bottom_marker),"bottom_marker_score_exact"] = 1
                    TEMP["bottom_marker_score"] = TEMP[["bottom_marker_score", "bottom_marker_score_exact"]].max(axis=1)
                    TEMP.sort_values(['bottom_marker_score', 'page_num'], ascending=[False, True], inplace=True)
                    print("bottom_marker:",bottom_marker)
                    print(TEMP[["page_num", "LINE_TEXT", "bottom_marker_score", "line_top"]])
                    d_ = dict(TEMP.iloc[0])
                    if (d_["bottom_marker_score"] > 0.6) & ((page_num is None) | ((page_num is not None) & (page_num == d_["page_num"]))):
                        if include_bottom == "YES":
                            bottom_b = d_["line_down"] * 1.01
                        else:
                            bottom_b = d_["line_top"] * 1.01
                        page_num = d_["page_num"]
                    else:
                        proceed_for_extraction = False
                if (not location_provided) & (not skip_left) & (left_marker != "PAGEBOUNDARY"):
                    TEMP["left_marker_score"] = TEMP['LINE_TEXT'].apply(find_similarity_words,
                        b=left_marker)
                    TEMP["left_marker_score_exact"] = 0
                    TEMP.loc[TEMP['LINE_TEXT'].str.contains(left_marker),"left_marker_score_exact"] = 1
                    TEMP["left_marker_score"] = TEMP[["left_marker_score", "left_marker_score_exact"]].max(axis=1)
                    print("left_marker:",left_marker)
                    #print(TEMP[["page_num", "LINE_TEXT", "left_marker_score", 'left_marker_score_exact',"line_right"]].drop_duplicates().head(30),"left***")


                    TEMP.sort_values(['left_marker_score', 'page_num'], ascending=[False, True], inplace=True)
                    print(TEMP[["page_num", "LINE_TEXT", "left_marker_score", "line_right"]])
                    d_ = dict(TEMP.iloc[0])
                    if (d_["left_marker_score"] > 0.6) & (page_num == d_["page_num"]):
                        len_ = len(str(left).split())
                        TEMP_ = TEMP.loc[(DF["page_num"] == d_["page_num"]) &
                        (DF["line_num"] == d_["line_num"])]
                        TEMP_.sort_values(["page_num", "line_num", "word_num"], inplace=True)

                        list_texts = list(TEMP_["text"])
                        list_boundary = list(TEMP_["right"])

                        list_text_search = [''.join(list_texts[i: i + len_]) for i in range(len(list_texts)- len_ + 1)]
                        list_text_search = [remove_special_charcters(x).upper() for x in list_text_search]
                        list_score = [find_similarity_words(s, left_marker) for s in list_text_search]
                        max_score_index = list_score.index(max(list_score))

                        list_boundary_search = [list_boundary[i: i + len_][-1] for i in range(len(list_boundary)- len_ + 1)]
                        # print(list_text_search)
                        # print(list_boundary_search)
                        # print(list_score)
                        # print(TEMP_[["page_num", "line_num", "word_num", "text", "right"]])
                        # print(len(str(left).split()))
                        # left_b = d_["line_right"] * 1.01
                        left_b = list_boundary_search[max_score_index] * 0.99
                    else:
                        proceed_for_extraction = False
                if (not location_provided) & (not skip_right) & (right_marker != "PAGEBOUNDARY"):
                    TEMP["right_marker_score"] = TEMP['LINE_TEXT'].apply(find_similarity_words,
                        b=right_marker)
                    TEMP["right_marker_score_exact"] = 0
                    TEMP.loc[TEMP['LINE_TEXT'].str.contains(right_marker),"right_marker_score_exact"] = 1
                    TEMP["right_marker_score"] = TEMP[["right_marker_score", "right_marker_score_exact"]].max(axis=1)
                    print("right_marker:",right_marker)
                    TEMP.sort_values(['right_marker_score', 'page_num'], ascending=[False, True], inplace=True)
                    print(TEMP[["page_num", "LINE_TEXT", "right_marker_score", "line_left"]])
                    d_ = dict(TEMP.iloc[0])
                    if (d_["right_marker_score"] > 0.6) & ((page_num is None) | ((page_num is not None) & (page_num == d_["page_num"]))):
                        print("yes")
                        print(right,"rrr")
                        len_ = len(str(right).split())
                        print(len_,"LR")
                        TEMP_ = TEMP.loc[(DF["page_num"] == d_["page_num"]) &
                        (DF["line_num"] == d_["line_num"])]
                        TEMP_.sort_values(["page_num", "line_num", "word_num"], inplace=True)

                        list_texts = list(TEMP_["text"])
                        list_boundary = list(TEMP_["left"])

                        list_text_search = [''.join(list_texts[i: i + len_]) for i in range(len(list_texts)- len_ + 1)]
                        list_text_search = [remove_special_charcters(x).upper() for x in list_text_search]
                        list_score = [find_similarity_words(s, right_marker) for s in list_text_search]
                        

                        max_score_index = list_score.index(max(list_score))

                        list_boundary_search = [list_boundary[i: i + len_][0] for i in range(len(list_boundary)- len_ + 1)]

                        # right_b = d_["line_left"] * 0.99
                        right_b = list_boundary_search[max_score_index] * 1.01
                        page_num = d_["page_num"]
                    else:
                        proceed_for_extraction = False

                if proceed_for_extraction:
                    print(top_b, bottom_b, left_b, right_b)
                    print("pageno",page_num)
                    print("topb",top_b)
                    print("bob",bottom_b)
                    print("lb",left_b)
                    print("rb",right_b)

                    # TEMP = (TEMP.loc[(TEMP["page_num"] == page_num)
                    #     & (top_b < TEMP['bottom'])
                    #     & (TEMP['top'] < bottom_b)
                    #     & (left_b < TEMP['right'])
                    #     & (TEMP['left'] < right_b)])

                    TEMP = (TEMP.loc[(TEMP["page_num"] == page_num)
                        & (top_b < TEMP['top'])
                        & (TEMP['bottom'] < bottom_b)
                        & (left_b < TEMP['left'])
                        & (TEMP['right'] < right_b)])
                    print("temp",TEMP[["line_text"]])


                    TEMP['isalphanum'] = TEMP['text'].apply(only_spclchar)
                    TEMP = TEMP.loc[TEMP['isalphanum']]
                    if TEMP.shape[0] > 0:
                        TEMP.sort_values(["page_num", "line_num", "word_num"],
                            ascending=[True, True, True], inplace=True)

                        TEMP = TEMP.groupby(['page_num', 'line_num']).agg({'text': lambda x: "%s" % ' '.join(x),
                            'word_num': 'count',
                            'left': 'min',
                            'right': 'max',
                            'top': 'min',
                            'bottom': 'max',
                            'image_height': 'first',
                            'image_widht': 'first'}).reset_index()

                        TEMP = TEMP.groupby(['page_num']).agg({'text': lambda x: "%s" % '\n'.join(x),
                            'line_num': 'count',
                            'word_num': 'count',
                            'left': 'min',
                            'right': 'max',
                            'top': 'min',
                            'bottom': 'max',
                            'image_height': 'first',
                            'image_widht': 'first'}).reset_index()

                        TEMP["surrounding_text_score"] = 1.0
                        TEMP["wordshape_score"] = 1.0
                        print(TEMP)
                    else:
                        TEMP["surrounding_text_score"] = 0.0
                        TEMP["wordshape_score"] = 0.0
                else:
                    TEMP["surrounding_text_score"] = 0.0
                    TEMP["wordshape_score"] = 0.0
            except Exception as e:
                print(e)
                TEMP["surrounding_text_score"] = 0.0
                TEMP["wordshape_score"] = 0.0
        else:
            TEMP["surrounding_text_score"] = 0.0
            TEMP["wordshape_score"] = 0.0
        TEMP = TEMP.loc[TEMP["wordshape_score"]>=0.5]
        TEMP['field_score'] = (TEMP['surrounding_text_score']*0.7 + TEMP['wordshape_score']*0.3)
        print(TEMP.sort_values(['field_score'], ascending=[False], inplace=False)[['text',
         'wordshape_score', 'surrounding_text_score','field_score']])

        TEMP = TEMP.loc[TEMP['field_score'] > 0.55]
        TEMP.sort_values(['field_score'], ascending=[False], inplace=True)
        print(TEMP[['text',
         'wordshape_score', 'surrounding_text_score','field_score']])
        print(TEMP[['text',
         'wordshape_score', 'surrounding_text_score', 'field_score']])

        final_candidates_ = {}
        if not TEMP.empty:
            conf_var =  np.random.uniform(0.6, 0.8)
            extracted_value = TEMP.iloc[0]['text']
            final_candidates_['line_num'] = TEMP.iloc[0]['line_num']
            final_candidates_["prob_" + f] = conf_var
            final_candidates_['text'] = extracted_value
            final_candidates_["Label_Present"] = True
            final_candidates_['word_num'] = TEMP.iloc[0]['word_num']

            final_candidates_['left'] = TEMP.iloc[0]['left']
            final_candidates_['right'] = TEMP.iloc[0]['right']
            final_candidates_['conf'] = conf_var
            final_candidates_['top'] = TEMP.iloc[0]['top']
            final_candidates_['bottom'] = TEMP.iloc[0]['bottom']

            final_candidates_['page_num'] = TEMP.iloc[0]['page_num']
            final_candidates_['image_height'] = TEMP.iloc[0]['image_height']
            final_candidates_['image_widht'] = TEMP.iloc[0]['image_widht']

            final_candidates_["label_confidence"] = None
            final_candidates_["wordshape"] = None
            final_candidates_["wordshape_confidence"] = None
            final_candidates_["Odds"] = None
            final_candidates_['model_confidence'] = 1

            final_candidates_['final_confidence_score'] = conf_var
            final_candidates_['vendor_masterdata_present'] = True
            final_candidates_['extracted_from_masterdata'] = False
        else:
        	# Default Value is getting populated
            print("Inside Not Extracted:", default_val)
            extracted_value = default_val
            final_candidates_['line_num'] = 0
            final_candidates_["prob_" + f] = 1
            final_candidates_['text'] = extracted_value
            final_candidates_["Label_Present"] = True
            final_candidates_['word_num'] = 0

            final_candidates_['left'] = 0
            final_candidates_['right'] = 1
            final_candidates_['conf'] = 1
            final_candidates_['top'] = 0
            final_candidates_['bottom'] = 1

            final_candidates_['page_num'] = 0
            final_candidates_['image_height'] = 1
            final_candidates_['image_widht'] = 1

            final_candidates_["label_confidence"] = None
            final_candidates_["wordshape"] = None
            final_candidates_["wordshape_confidence"] = None
            final_candidates_["Odds"] = None
            final_candidates_['model_confidence'] = 1

            final_candidates_['final_confidence_score'] = 1
            final_candidates_['vendor_masterdata_present'] = True
            final_candidates_['extracted_from_masterdata'] = False
        results[f] = final_candidates_
    return results
'''


def extract_vendor_specific_extra_fields_old(DF, vendor_masterdata_score, vendor_masterdata):
    """
    Write code to extract Vendor Specific Extra Fields
    """
    # Anjan: Get custom field and information from new sheet based on VENDOR_ID

    print("____________________")
    print('extract_vendor_specific_extra_fields')
    ll = list(vendor_masterdata.keys())
    print(vendor_masterdata)

    vendor_id = vendor_masterdata['VENDOR_ID']
    # Code added to test Template creation
    global VENDOR_SPECIFIC_FIELD
    VENDOR_SPECIFIC_FIELD = pd.read_csv(customFieldPath, encoding='unicode_escape')
    print(vendor_id)
    print(VENDOR_SPECIFIC_FIELD)
    TEMP = VENDOR_SPECIFIC_FIELD.loc[VENDOR_SPECIFIC_FIELD['VENDOR_ID'] == vendor_id]
    TEMP.dropna(inplace=True)

    print(TEMP)
    dict_fields_to_extract = {}
    if not TEMP.empty:
        for idx, row in TEMP.iterrows():
            field = row['Custom Field Name']
            label = row['Custom Field Label']
            pos = row['Custom Field Label Position']
            shape = row['Custom Field Shape']
            loc = row['Custom Field Location']
            default_val = row['Default Value']
            horizontal_anchor = row['Horizontal Anchor']
            vertical_anchor = row['Vertical Anchor']
            # Code added for location based extraction
            top = row['TOP']
            bottom = row['BOTTOM']
            left = row['LEFT']
            right = row['RIGHT']
            top_delimiter = row['Top Delimiter']
            bottom_delimiter = row['Bottom Delimiter']
            dict_fields_to_extract[field] = (label, pos, shape, loc, default_val, horizontal_anchor,
                vertical_anchor, top, bottom, left, right, top_delimiter, bottom_delimiter)


    DF['Left_1'] = DF['W1Lf'].astype(str).str.upper().replace('NAN','')
    DF['Left_2'] = DF['W2Lf'].astype(str).str.upper().replace('NAN','')
    DF['Left_3'] = DF['W3Lf'].astype(str).str.upper().replace('NAN','')
    DF['Left_4'] = DF['W4Lf'].astype(str).str.upper().replace('NAN','')
    DF['Left_5'] = DF['W5Lf'].astype(str).str.upper().replace('NAN','')

    DF['Abv_1'] = DF['W1Ab'].astype(str).str.upper().replace('NAN','')
    DF['Abv_2'] = DF['W2Ab'].astype(str).str.upper().replace('NAN','')
    DF['Abv_3'] = DF['W3Ab'].astype(str).str.upper().replace('NAN','')
    DF['Abv_4'] = DF['W4Ab'].astype(str).str.upper().replace('NAN','')
    DF['Abv_5'] = DF['W5Ab'].astype(str).str.upper().replace('NAN','')

    DF['LINE_TEXT'] = DF['line_text'].astype(str).apply(remove_special_charcters).str.upper()
    DF['LINE_TEXT'] = DF['LINE_TEXT'].str.replace(" ","")

    #DF['TEXT_ABOVE'] = DF["Abv_5"] + DF["Abv_4"] + DF["Abv_3"] + DF["Abv_2"] + DF["Abv_1"]
    DF['TEXT_ABOVE'] = DF["Abv_1"] + DF["Abv_2"] + DF["Abv_3"] + DF["Abv_4"] + DF["Abv_5"]
    DF['TEXT_LEFT'] = DF["Left_4"] + DF["Left_3"] + DF["Left_2"] + DF["Left_1"]

    DF['TEXT_ABOVE'] = DF['TEXT_ABOVE'].astype(str).apply(remove_special_charcters).str.upper()
    DF['TEXT_LEFT'] = DF['TEXT_LEFT'].astype(str).apply(remove_special_charcters).str.upper()

    results = {}
    for f, v in dict_fields_to_extract.items():
        print("*********************************************************")
        label = v[0]
        pos = v[1]
        shape = v[2]
        loc = v[3]
        default_val = v[4]
        horizontal_anchor = v[5]
        vertical_anchor = v[6]
		# Code added for location based extraction
        top = v[7]
        bottom = v[8]
        left = v[9]
        right = v[10]
        top_delimiter = v[11]
        bottom_delimiter = v[12]

        label = label.upper()
        list_labels = label.split()
        list_labels = [remove_special_charcters(x) for x in list_labels]
        string_label = ''.join(list_labels)

        horizontal_anchor = horizontal_anchor.upper()
        list_horizontal_anchor = horizontal_anchor.split()
        list_horizontal_anchor = [remove_special_charcters(x) for x in list_horizontal_anchor]
        horizontal_anchor = ''.join(list_horizontal_anchor)

        vertical_anchor = vertical_anchor.upper()
        list_vertical_anchor = vertical_anchor.split()
        list_vertical_anchor = [remove_special_charcters(x) for x in list_vertical_anchor]
        vertical_anchor = ''.join(list_vertical_anchor)

        extracted_value = None

        # Add filtering based on location
        coordinates = LOCATION_COORDINATES[loc]
        TEMP = DF.loc[(DF['left'] >= coordinates[0]) & (DF['right'] <= coordinates[1]) &
        (DF['top'] >= coordinates[2]) & (DF['bottom'] <= coordinates[3])]

        # Add filtering based on delimiters
        # Search for delimiters
        top_delimiter = top_delimiter.upper()
        list_top_delimiter = top_delimiter.split()
        list_top_delimiter = [remove_special_charcters(x) for x in list_top_delimiter]
        top_delimiter = ' '.join(list_top_delimiter)

        bottom_delimiter = bottom_delimiter.upper()
        list_bottom_delimiter = bottom_delimiter.split()
        list_bottom_delimiter = [remove_special_charcters(x) for x in list_bottom_delimiter]
        bottom_delimiter = ' '.join(list_bottom_delimiter)

        if (top_delimiter != "NOT APPLICABLE") or (bottom_delimiter != "NOT APPLICABLE"):
            print("Inside Delimiter Case")
            print(top_delimiter, bottom_delimiter)
            top_delimit = 0.0
            bottom_delimit = 1.0
            page_num_min = 0
            page_num_max = TEMP['page_num'].max()

            print(top_delimit, bottom_delimit, page_num_min, page_num_max)

            TEMP_ = TEMP.copy()
            if top_delimiter != "NOT APPLICABLE":
                TEMP_["top_delimiter_score"] = TEMP_['LINE_TEXT'].apply(find_similarity_words,
                    b=top_delimiter)
                TEMP_ = TEMP_.loc[TEMP_['top_delimiter_score'] > 0.8]
                if TEMP_.shape[0] > 0:
                    TEMP_.sort_values(['top_delimiter_score'], ascending=[False],
                        inplace=True)
                    top_delimit = TEMP_.iloc[0]['line_down']
                    page_num_min = TEMP_.iloc[0]['page_num']

            TEMP_ = TEMP.copy()
            if bottom_delimiter != "NOT APPLICABLE":
                TEMP_["bottom_delimiter_score"] = TEMP_['LINE_TEXT'].apply(find_similarity_words,
                    b=bottom_delimiter)
                TEMP_ = TEMP_.loc[TEMP_['bottom_delimiter_score'] > 0.8]
                if TEMP_.shape[0] > 0:
                    TEMP_.sort_values(['bottom_delimiter_score'], ascending=[False],
                        inplace=True)
                    bottom_delimit = TEMP_.iloc[0]['line_top']
                    page_num_max = TEMP_.iloc[0]['page_num']

            print(top_delimit, bottom_delimit, page_num_min, page_num_max)
            TEMP = TEMP.loc[(TEMP['page_num'] >= page_num_min)
            & (TEMP['page_num'] <= page_num_max)
            & (TEMP['line_top'] >= top_delimit)
            & (TEMP['line_down'] <= bottom_delimit)]


        ### Find Score based on Left Value
        if pos == 'Left':
            TEMP["surrounding_text_score"] = TEMP['TEXT_LEFT'].apply(find_similarity_words,
             b=string_label)

            TEMP["wordshape_score"] = TEMP['wordshape'].apply(find_similarity_words, b=shape)
        elif pos == 'Above':
            TEMP["surrounding_text_score"] = TEMP['TEXT_ABOVE'].apply(find_similarity_words,
             b=string_label)
            print(string_label,"strin*****")
            TEMP["wordshape_score"] = TEMP['wordshape'].apply(find_similarity_words, b=shape)
        elif (pos == 'ANCHOR_LEFT_TOP') | (pos == "ANCHOR_LEFT_BOTTOM") | (pos == 'ANCHOR_RIGHT_TOP') | (pos == "ANCHOR_RIGHT_BOTTOM"):
            TEMP["surrounding_text_score"] = 0.0
            TEMP["wordshape_score"] = 0.0

            TEMP_ANCHOR = DF.copy()
            print("Inside ANCHOR Label CASE!!!!!!!", horizontal_anchor, vertical_anchor)
            TEMP_ANCHOR["horizontal_anchor_score"] = TEMP_ANCHOR['LINE_TEXT'].apply(find_similarity_words,
             b=horizontal_anchor)
            TEMP_ANCHOR["vertical_anchor_score"] = TEMP_ANCHOR['LINE_TEXT'].apply(find_similarity_words,
             b=vertical_anchor)

            # print(TEMP.sort_values(['left_anchor_score'], ascending=[False])[['LINE_TEXT', 'left_anchor_score', 'line_top', 'line_down','line_left', 'line_right']])
            # print(TEMP.sort_values(['top_anchor_score'], ascending=[False])[['LINE_TEXT', 'top_anchor_score', 'line_top', 'line_down','line_left', 'line_right']])

            TEMP_HORIZONTAL_ANCHOR = TEMP_ANCHOR.loc[(TEMP_ANCHOR['horizontal_anchor_score'] > 0.70)][['LINE_TEXT', 'horizontal_anchor_score', 'line_top', 'line_down','line_left', 'line_right']]
            TEMP_HORIZONTAL_ANCHOR.drop_duplicates(inplace=True)
            TEMP_VERTICAL_ANCHOR = TEMP_ANCHOR.loc[(TEMP_ANCHOR['vertical_anchor_score'] > 0.70)][['LINE_TEXT', 'vertical_anchor_score', 'line_left', 'line_right','line_top', 'line_down']]
            TEMP_VERTICAL_ANCHOR.drop_duplicates(inplace=True)
            print("CHecking Template")
            print(TEMP_VERTICAL_ANCHOR)
            print(TEMP_HORIZONTAL_ANCHOR)

            if (TEMP_HORIZONTAL_ANCHOR.shape[0] > 0) & (TEMP_VERTICAL_ANCHOR.shape[0] > 0):
                TEMP_HORIZONTAL_ANCHOR.sort_values(['horizontal_anchor_score'], ascending=[False], inplace=True)

                TEMP_VERTICAL_ANCHOR.sort_values(['vertical_anchor_score'], ascending=[False], inplace=True)
                print(TEMP_VERTICAL_ANCHOR)
                print(TEMP_HORIZONTAL_ANCHOR)

                candidate_found = False
                # TEMP = pd.DataFrame()
                print(TEMP.shape)
                TEMP_ = TEMP.copy()
                dummyDF = pd.DataFrame(columns = list(TEMP.columns))
                for idx_, row_ in TEMP_HORIZONTAL_ANCHOR.iterrows():
                    top_boundary = float(row_['line_top'])
                    bottom_boundary = float(row_['line_down'])
                    horizontal_cut = 0
                    horizontal_anchor_score = float(row_['horizontal_anchor_score'])
                    for idx__,row__ in TEMP_VERTICAL_ANCHOR.iterrows():
                        left_boundary = float(row__['line_left'])
                        right_boundary = float(row__['line_right'])
                        vertical_cut = 0
                        vertical_anchor_score = float(row__['vertical_anchor_score'])
                        TEMP = TEMP_.copy()
                        if pos == "ANCHOR_LEFT_TOP":
                            horizontal_cut = float(row_['line_right'])
                            vertical_cut = float(row__['line_down'])
                            TEMP = (TEMP.loc[(TEMP['top'] >= vertical_cut)
                                & (TEMP['left'] >= horizontal_cut)])
                        elif pos == "ANCHOR_LEFT_BOTTOM":
                            horizontal_cut = float(row_['line_right'])
                            vertical_cut = float(row__['line_top'])
                            TEMP = (TEMP.loc[(TEMP['bottom'] <= vertical_cut)
                                & (TEMP['left'] >= horizontal_cut)])
                        elif pos == "ANCHOR_RIGHT_TOP":
                            horizontal_cut = float(row_['line_left'])
                            vertical_cut = float(row__['line_down'])
                            TEMP = (TEMP.loc[(TEMP['top'] >= vertical_cut)
                                & (TEMP['right'] <= horizontal_cut)])
                        elif pos == "ANCHOR_RIGHT_BOTTOM":
                            horizontal_cut = float(row_['line_left'])
                            vertical_cut = float(row__['line_top'])
                            TEMP = (TEMP.loc[(TEMP['bottom'] <= vertical_cut)
                                & (TEMP['right'] <= horizontal_cut)])

                        TEMP = (TEMP.loc[(top_boundary <= TEMP['bottom'])
                            & (TEMP['top'] <= bottom_boundary)
                            & (left_boundary <= TEMP['right'])
                            & (TEMP['left'] <= right_boundary)])

                        if TEMP.shape[0] > 0:
                            TEMP['anchor_left'] = left_boundary
                            TEMP['anchor_right'] = right_boundary
                            TEMP['anchor_top'] = top_boundary
                            TEMP['anchor_bottom'] = bottom_boundary

                            TEMP['min_left'] = TEMP[['left','anchor_left']].max(axis=1)
                            TEMP['min_right'] = TEMP[['right','anchor_right']].min(axis=1)
                            TEMP['min_top'] = TEMP[['top','anchor_top']].max(axis=1)
                            TEMP['min_bottom'] = TEMP[['bottom','anchor_bottom']].min(axis=1)

                            TEMP['overlap_horizontal'] = TEMP['min_right'] - TEMP['min_left']
                            TEMP['overlap_vertical'] = TEMP['min_bottom'] - TEMP['min_top']

                            TEMP['overlap_score_horizontal'] = (TEMP['overlap_horizontal']
                                /TEMP['width'])
                            TEMP['overlap_score_vertical'] = (TEMP['overlap_vertical']
                                /TEMP['height'])

                            TEMP['overlap_score_vertical'] = TEMP['overlap_score_vertical'] * horizontal_anchor_score
                            TEMP['overlap_score_horizontal'] = TEMP['overlap_score_horizontal'] * vertical_anchor_score

                            TEMP["wordshape_score"] = TEMP['wordshape'].apply(find_similarity_words, b=shape)
                            TEMP = TEMP.loc[TEMP['wordshape_score'] > 0.60]
                            if TEMP.shape[0] > 0:
                                TEMP['surrounding_text_score'] = TEMP[['overlap_score_horizontal'
                                ,'overlap_score_vertical']].mean(axis=1)
                                dummyDF = pd.concat([dummyDF, TEMP], axis = 0)
                # print("CHECKING TEMPLAATE 2: \n", dummyDF.sort_values(['surrounding_text_score'], ascending=[False])[['text',
                                                            # 'surrounding_text_score','wordshape_score','overlap_score_horizontal'
                                                            # ,'overlap_score_vertical']])
                if dummyDF.shape[0]>0:
                    TEMP = dummyDF.sort_values(['surrounding_text_score'], ascending=[False])
                    TEMP = TEMP.reset_index()
                    #TEMP = TEMP.loc[[0]]
                    #TEMP['surrounding_text_score'] = 1.0
                else:
                    TEMP["surrounding_text_score"] = 0.0
                    TEMP["wordshape_score"] = 0.0
        elif pos == 'LOCATION':
            # Code added for Location based extraction
            print("Inside Location Based Extraction")
            try:
                top = float(top)*0.95
                bottom = float(bottom)*1.05
                left = float(left)*0.95
                right = float(right)*1.05

                TEMP = (TEMP.loc[(top <= TEMP['bottom'])
                    & (TEMP['top'] <= bottom)
                    & (left <= TEMP['right'])
                    & (TEMP['left'] <= right)])
                if TEMP.shape[0] > 0:
                    TEMP['top_boundary'] = top
                    TEMP['bottom_boundary'] = bottom
                    TEMP['left_boundary'] = left
                    TEMP['right_boundary'] = right

                    TEMP['min_top'] = TEMP[['top','top_boundary']].max(axis=1)
                    TEMP['min_bottom'] = TEMP[['bottom','bottom_boundary']].min(axis=1)
                    TEMP['min_left'] = TEMP[['left','left_boundary']].max(axis=1)
                    TEMP['min_right'] = TEMP[['right','right_boundary']].min(axis=1)

                    TEMP['overlap_area'] = ((TEMP['min_right'] - TEMP['min_left'])*
                        (TEMP['min_bottom'] - TEMP['min_top']))

                    TEMP['area'] = ((TEMP['right'] - TEMP['left'])*
                        (TEMP['bottom'] - TEMP['top']))

                    TEMP["wordshape_score"] = TEMP['wordshape'].apply(find_similarity_words, b=shape)
                    TEMP = TEMP.loc[TEMP['wordshape_score'] > 0.60]
                    if TEMP.shape[0] > 0:
                        TEMP['surrounding_text_score'] = TEMP['overlap_area']/TEMP['area']
                    else:
                        TEMP["surrounding_text_score"] = 0.0
                else:
                    TEMP["surrounding_text_score"] = 0.0
                    TEMP["wordshape_score"] = 0.0
            except:
                TEMP["surrounding_text_score"] = 0.0
                TEMP["wordshape_score"] = 0.0
        else:
            TEMP["surrounding_text_score"] = 0.0
            TEMP["wordshape_score"] = 0.0
        TEMP = TEMP.loc[TEMP["wordshape_score"]>=0.5]
        TEMP['field_score'] = (TEMP['surrounding_text_score']*0.7 + TEMP['wordshape_score']*0.3)
        print(TEMP.sort_values(['field_score'], ascending=[False], inplace=False)[['text', 'TEXT_LEFT','TEXT_ABOVE',
         'wordshape_score', 'surrounding_text_score','field_score']])
        TEMP = TEMP.loc[TEMP['field_score'] > 0.55]
        TEMP.sort_values(['field_score'], ascending=[False], inplace=True)
        print(TEMP[['text', 'TEXT_LEFT',
         'wordshape_score', 'surrounding_text_score','field_score']])
        print(TEMP[['text', 'TEXT_ABOVE',
         'wordshape_score', 'surrounding_text_score', 'field_score']])

        final_candidates_ = {}
        if not TEMP.empty:
            extracted_value = TEMP.iloc[0]['text']
            final_candidates_['line_num'] = TEMP.iloc[0]['line_num']
            final_candidates_["prob_" + f] = 1
            final_candidates_['text'] = extracted_value
            final_candidates_["Label_Present"] = True
            final_candidates_['word_num'] = TEMP.iloc[0]['word_num']

            final_candidates_['left'] = TEMP.iloc[0]['left']
            final_candidates_['right'] = TEMP.iloc[0]['right']
            final_candidates_['conf'] = 1
            final_candidates_['top'] = TEMP.iloc[0]['top']
            final_candidates_['bottom'] = TEMP.iloc[0]['bottom']

            final_candidates_['page_num'] = TEMP.iloc[0]['page_num']
            final_candidates_['image_height'] = TEMP.iloc[0]['image_height']
            final_candidates_['image_widht'] = TEMP.iloc[0]['image_widht']

            final_candidates_["label_confidence"] = None
            final_candidates_["wordshape"] = None
            final_candidates_["wordshape_confidence"] = None
            final_candidates_["Odds"] = None
            final_candidates_['model_confidence'] = 1

            final_candidates_['final_confidence_score'] = 1
            final_candidates_['vendor_masterdata_present'] = True
            final_candidates_['extracted_from_masterdata'] = False
        else:
        	# Default Value is getting populated
            print("Inside Not Extracted:", default_val)
            extracted_value = default_val
            final_candidates_['line_num'] = 0
            final_candidates_["prob_" + f] = 1
            final_candidates_['text'] = extracted_value
            final_candidates_["Label_Present"] = True
            final_candidates_['word_num'] = 0

            final_candidates_['left'] = 0
            final_candidates_['right'] = 1
            final_candidates_['conf'] = 1
            final_candidates_['top'] = 0
            final_candidates_['bottom'] = 1

            final_candidates_['page_num'] = 0
            final_candidates_['image_height'] = 1
            final_candidates_['image_widht'] = 1

            final_candidates_["label_confidence"] = None
            final_candidates_["wordshape"] = None
            final_candidates_["wordshape_confidence"] = None
            final_candidates_["Odds"] = None
            final_candidates_['model_confidence'] = 1

            final_candidates_['final_confidence_score'] = 1
            final_candidates_['vendor_masterdata_present'] = True
            final_candidates_['extracted_from_masterdata'] = False
        results[f] = final_candidates_
    return results

def isfloat(value):
    try:
        float(value)
        return True
    except:
        return False
    
    
def only_spclchar(s):
    if not re.match(r'^[_\W]+$', s):
        return True
    else:
        return False
def taxes_validate(DF,prediction):
    try:
        amounts = (calcAmountFields_validation(DF, prediction))
        print("taxes_validate amout:",amounts)
        if prediction.get('vendorGSTIN')['text'] not in vendorGSTIN_list:
            if abs(float(amounts.get('cgst'))- float(prediction.get('CGSTAmount')['text']))>0.9:
                prediction['CGSTAmount']['text']=round(amounts.get('cgst'),2)
            if abs(float(amounts.get('sgst'))- float(prediction.get('SGSTAmount')['text']))>0.9:
                prediction['SGSTAmount']['text']=round(amounts.get('sgst'),2)
            if abs(float(amounts.get('igst'))- float(prediction.get('IGSTAmount')['text']))>0.9:
                prediction['IGSTAmount']['text']=round(amounts.get('igst'),2)
            if abs(float(amounts.get('totalGst'))- float(prediction.get('totalGSTAmount')['text']))>0.9:
                prediction['totalGSTAmount']['text']=round(amounts.get('totalGst'),2)
            return prediction
        else:
            return prediction
    except:
        print("excemption in taxes_validate")
        return prediction
def add_rpainvnum_prediction(rpainvnum,final_prediction):
    final_prediction_copy =copy.deepcopy(final_prediction)
    try:
        print("add_rpainvnum_prediction",final_prediction)
        if 'rpaInvoiceNumber' in final_prediction:
            # print("acds")
            final_prediction['rpaInvoiceNumber']['text'] = rpainvnum.strip()
            final_prediction['rpaInvoiceNumber']['final_confidence_score'] = random.randint(900, 950)/1000
            final_prediction['rpaInvoiceNumber']['extracted_from_masterdata'] = True
        return final_prediction
    except:
        print("add_rpainvnum_prediction",traceback.print_exc())
        return final_prediction_copy
@putil.timing    
def post_process(DF, docMetaData=None):
    """
    Post Processing Code
    Step 1: Identify Vendor from Vendor MasterData
    There can be two cases: Vendor is present or absent
    """
    
    # Reading Master Data
    ADDRESS_MASTERDATA = pd.read_csv(addressFilePath, encoding='unicode_escape')
    ADDRESS_MASTERDATA = ADDRESS_MASTERDATA.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    VENDOR_MASTERDATA = pd.read_csv(masterFilePath, encoding='unicode_escape')
    VENDOR_MASTERDATA = VENDOR_MASTERDATA.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    REFERENCE_DATA = pd.read_csv(REFERENCE_MASTER_DATA_PATH, encoding='unicode_escape')
    print("The REFERENCE_DATA size: ",REFERENCE_DATA.shape)
    print("VENDOR_MASTERDATA: ",VENDOR_MASTERDATA.shape)
    print("ADDRESS_MASTERDATA: ",ADDRESS_MASTERDATA.shape)

    # Get docType and orgType from input
    if docMetaData:
        try:
            print("Checking input metadata: {}".format(docMetaData))
            result = docMetaData.get('result')
            document = result.get('document')
            preprocess_vendor = document.get('Vendor_Name')
            org_type = document.get('orgType')
            doc_type = document.get('docType')
            doc_poNumber = document.get('poNumber')
            doc_invNumber = document.get('invNumber')
            doc_vendorGSTIN = document.get('vendorGSTIN')
            grn_Date = document.get('grnDate')
            if not preprocess_vendor:
                preprocess_vendor = "UNKNOWN"
        except Exception as e:
            print(e)
            preprocess_vendor = "UNKNOWN"
    else:
        preprocess_vendor = "UNKNOWN"
    print("Calling g:", preprocess_vendor)
    format_, vendor_masterdata_score, vendor_masterdata = GetVendor(DF,VENDOR_MASTERDATA) #get_vendor(DF)
    print("format_",format_,
          "\tvendor_masterdata_score",vendor_masterdata_score,
          "\tVendor Master data dict :",vendor_masterdata)
    if vendor_masterdata is not None:
        vendor_pan = vendor_masterdata["VENDOR_GSTIN"][2:12]
    else:
        vendor_pan = ""
    
    # Modify prediction dataframe
    DF = modify_prediction(DF,vendor_masterdata)
    
    final_prediction = {}

    #header fields
    for f in FIELD['header_fields']:
        header_prediction = procefinal_candidates_header_without_vendormasterdata(f, DF , vendor_pan)
        print("header field prediction :",header_prediction)
        final_prediction = {**final_prediction, **header_prediction}

    # date fields
    for f in FIELD['date_fields']:
        date_prediction = procefinal_candidates_date_without_vendormasterdata(f, DF)
        print(date_prediction)
        final_prediction = {**final_prediction, **date_prediction}

    # amountfields
    for f in FIELD['amount_fields']:
        amount_prediction = procefinal_candidates_amount_without_vendormasterdata(f, DF)
        final_prediction = {**final_prediction, **amount_prediction}
   
    # vendorspecific fields
    for f in FIELD['vendor_specific_addrefinal_candidates_fields']:
        addrefinal_candidates_prediction = procefinal_candidates_vendor_addrefinal_candidates_without_vendormasterdata(
            f, DF)
        final_prediction = {**final_prediction, **addrefinal_candidates_prediction}

    for f in FIELD['vendor_specific_header_fields']:
        vendor_header_prediction = procefinal_candidates_vender_specific_header_without_vendormasterdata(f, DF)
        final_prediction = {**final_prediction, **vendor_header_prediction}

    for f in FIELD['vendor_names_fields']:
        vendor_header_prediction = procefinal_candidates_vender_names_without_vendormasterdata(f, DF)
        final_prediction = {**final_prediction, **vendor_header_prediction}

    # Amit: Code added for vendorGSTIN extraction May15
    if not "vendorGSTIN" in final_prediction:
        vendorGSTIN_prediction = procefinal_candidates_vender_specific_header_without_vendormasterdata("vendorGSTIN", DF)
        final_prediction = {**final_prediction, **vendorGSTIN_prediction}

    # Amit: Code added to extract billing and shipping Address from Model Prediction
    for f in FIELD['address_fields']:
        address_prediction = procefinal_candidates_vendor_addrefinal_candidates_without_vendormasterdata(f, DF)
        final_prediction = {**final_prediction, **address_prediction}

    line_item_prediction = {}
    if len(list(DF['line_row'].unique())) == 1:
        print("Calling Line Item Old Method: ", list(DF['line_row'].unique()))
        line_item_prediction = extract_line_items_new(DF, final_prediction)
    else:
        print("Calling Line Item New Method: ", list(DF['line_row'].unique()))
        line_item_prediction = line_items_(DF, final_prediction)
    line={}
    line['lineItemPrediction']=line_item_prediction
    final_prediction = {**final_prediction, **line}

    # print("Post Processor:", final_prediction)
    # Remove unnecessary fields from prediction
    final_prediction.pop("customerId", None)
    final_prediction.pop("customerAddress", None)
    final_prediction.pop("customerName", None)
    final_prediction.pop("taxAmount", None)
    final_prediction.pop("taxRate", None)

    # Code to correct duplicate prediction
    final_prediction = correct_duplicate_prediction(final_prediction)
    
    # final_prediction.pop('freightAmount', None)
    # final_prediction.pop('discountAmount', None)
    # Code to extract Vendpor Specific Extra Fields
    if format_ is not None:
        vendor_specific_extra_field_prediction = extract_vendor_specific_extra_fields(DF, vendor_masterdata)
        final_prediction = {**final_prediction, **vendor_specific_extra_field_prediction}
        
        extracted_table = extract_table(DF, vendor_masterdata_score, vendor_masterdata)
        print("Extracted Table: \n ", extracted_table)
        
        if extracted_table:
            final_prediction.pop('lineItemPrediction',None)
            line={}
            line['lineItemPrediction'] = extracted_table
            final_prediction = {**final_prediction, **line}


    # Code to extract billing and shipping address from MasterData
    address = get_addresses_from_masterdata(DF, final_prediction)
    final_prediction = {**final_prediction, **address}
    #Method to reduce final confidence if OCR Confidence is low
    final_prediction = check_ocr_confidence(DF, final_prediction)
    # print("Final prediction after check ocr confidence:", final_prediction)
    # # Apply business rules
    final_prediction = apply_business_rules(DF, final_prediction, format_,ADDRESS_MASTERDATA,VENDOR_MASTERDATA)
    print("Final prediction after apply business rules:", final_prediction)

    # #Apply client rules
    final_prediction = apply_client_rules(DF, final_prediction, docMetaData, ADDRESS_MASTERDATA,VENDOR_MASTERDATA,doc_vendorGSTIN)
    print("Final prediction after client rules:", final_prediction)
    # 15 May 2023 Apply templates for OCR Issues GSTIN
    # if format_ is None:
    format_, vendor_masterdata_score, vendor_masterdata = GetVendorByPred(final_prediction,VENDOR_MASTERDATA) #get_vendor(DF)
    print("Vendor Id After prediction:")
    print("format_",format_,
        "\tvendor_masterdata_score",vendor_masterdata_score,
        "\tVendor Master data dict :",vendor_masterdata)
    if format_ is not None:
        vendor_specific_extra_field_prediction = extract_vendor_specific_extra_fields(DF, vendor_masterdata)
        final_prediction = {**final_prediction, **vendor_specific_extra_field_prediction}
    ## 9-Oct-2023 Added template for total_amount of discr note
    if doc_type == "Discrepancy Note":
        print("Using template for discrepancy note")
        vendor_specific_extra_field_prediction = extract_vendor_specific_extra_fields(DF, vendor_masterdata, doc_type)
        final_prediction = {**final_prediction, **vendor_specific_extra_field_prediction}    
    ## 26 sept 2023 Added to convert dates to dd/mm/yyyy format Blinkit Requirenments
    if cfg.get_blinkit_date_format():
        final_prediction, date_format_flag = cl_rules.convert_dates(final_prediction, REFERENCE_DATA)
        final_prediction = cl_rules.convert_total_amount(final_prediction)  
        # extracted_table = extract_table(DF, vendor_masterdata_score, vendor_masterdata)
        # print("Extracted Table: \n ", extracted_table)
        
        # if extracted_table:
        #     final_prediction.pop('lineItemPrediction',None)
        #     line={}
        #     line['lineItemPrediction'] = extracted_table
        #     final_prediction = {**final_prediction, **line}
    final_prediction=sanitize_invoice_date_text(final_prediction)
    final_prediction=countrybean_ocr_corr(final_prediction)
    final_prediction=INNOVATIVE_foods_corr(final_prediction)
    final_prediction=muddy_puddle_invnum(final_prediction, DF)
    # ## This changes was added for KYC Documents that we r not processing now so commenting this changes: March 17, 2023
    # if cfg.getAppName() == "PAIGES":
    #     final_prediction = present_doc_output(final_prediction, doc_type, org_type)
    #     print("Final prediction after present doc_output done")
    #     print("Org Type",org_type)
    #     # if (org_type.upper()!="ACC PAYABLE") or (org_type.upper()!="ORG 001") or (org_type.upper()!="ORG_001") or (org_type.upper()!="ORG_001"):
    #     #     final_prediction={'lineItemPrediction':{}}
    #     #     extracted_data = insurance_validation(DF, final_prediction)
    #     #     final_prediction = {**final_prediction, **extracted_data}
    #     # final_prediction = present_doc_output(final_prediction, doc_type, org_type)
    # ## This changes was added for KYC Documents that we r not processing now so commenting this changes: March 17, 2023

    # #code added to extract poNumber from masterdata
    # def validate_poNumber(final_prediction,doc_poNumber,df):
    #     final_prediction_copy = final_prediction.copy()
    #     try:

    #         if final_prediction['poNumber']['text'] == doc_poNumber:
    #             return final_prediction

        
    #         match_list=predictPoNumber_from_metadata(df)
                
    #         print(match_list,"Matched")
    #         if len(match_list)>0:
    #             for match in match_list:
    #                 if doc_poNumber is not None and doc_poNumber==match:
    #                     final_prediction['poNumber']['text']=doc_poNumber
    #                     final_prediction['poNumber']['final_confidence_score']= 1.0
    #                     break
    #                 else:
    #                     print("Taking from extraction as metadata poNumber is None")

    #             return final_prediction     
                    
    #         else:
    #             # 
    #             print("No match found so,predictpoNumber is running")
    #             return final_prediction
            
    #     except:
    #         print('exception in po Number', traceback.print_exc())
    #         return final_prediction_copy


    # Add rule based validation
    if format_ is not None:
        final_prediction = validate_final_output(final_prediction, vendor_masterdata['VENDOR_ID'])
    else:
        final_prediction = validate_final_output(final_prediction, "DEFAULT")
    final_prediction = scale_bounding_boxes(DF, final_prediction)

    ovr,crit = score_value(final_prediction, vendor_masterdata,docMetaData)
    print("Overall and crit scores: ",ovr, crit)
    #stp=check_stp(final_prediction,vendor_masterdata)
# #     ## commentig this changes only added for demo purpose
#     # CHANGED FOR DEMO MACHINE; STP TO BE MADE 100% FOR NON INVOICE FIELDS
#     if cfg.getAppName() == "PAIGES":
# # <<<<<<< HEAD
#         # if (doc_type.upper()!="INVOICE") and (org_type.upper()!="ACC PAYABLE"):
# # =======
#         if org_type.upper()!="ACC PAYABLE":
# # >>>>>>> dbde80ef059e1e46ab6a01743f24b5a0a1730bc6
#             stp=1
#             crit = ovr
#     ## commentig this changes only added for demo purpose

    # Keep required fileds for the client in the prediction
    final_prediction = client_required_fields(final_prediction, vendor_masterdata,docMetaData)
    final_prediction = add_rpainvnum_prediction(doc_invNumber,final_prediction)
    final_prediction = makeDefaultPredValToNA(final_prediction)
    final_prediction = vendor_name_validation(final_prediction,VENDOR_MASTERDATA)
    # final_prediction = validating_amount_fields_increasing_confidence(DF,final_prediction)
    print("final prediction after adding required fields :",final_prediction.keys())
    final_prediction =  custom_sorting_prediction(final_prediction)
    print("Prediction after sorting by order :",final_prediction.keys())
    
    final_prediction = multi_token_invnum(final_prediction, DF)
    #final_prediction = checking_stp_1(final_prediction,REFERENCE_DATA)
    # final_prediction = cl_rules.matchReferenceData(final_prediction)
    final_prediction = cl_rules.add_reference_data_flag(final_prediction,REFERENCE_DATA)
    print("calling check_multiple_invoices")
    #final_prediction = taxes_validate(DF,final_prediction)
    final_prediction = check_multiple_invoices("invoiceNumber",final_prediction,DF)
    # # Added rounding off total amount to 2 decimal 30 Nov
    # final_prediction = biz.roundOffAmount(final_prediction)
    # # Added rounding off total amount to 2 decimal 30 Nov

    # stp=0
    final_prediction = update_final_prediction_for_tax_slab(final_prediction,DF,docMetaData)
    #final_prediction = calculateandassignslab(DF,final_prediction)
    
    final_prediction = get_zero_percentage_subTotal(DF,final_prediction)
    # verifying Amount Fields
    final_prediction = verify_total_amount_fields(final_prediction,DF,docMetaData)
    final_prediction = add_zero_to_empty_tax_slabs(final_prediction,docMetaData)
    # final_prediction = cl_rules.validate_poNumber(DF,final_prediction,doc_poNumber,docMetaData)
    # final_prediction = cl_rules.checklenpoNumner(final_prediction)
    final_prediction,flag_invoice_number = cl_rules.validate_invNumber(DF,final_prediction,doc_invNumber,docMetaData)
    final_prediction = add_NA_to_cities(final_prediction)
    final_prediction,flag_invoice_number = cl_rules.extract_vendor_specific_fields(final_prediction, DF, flag_invoice_number)
    final_prediction = cl_rules.get_vendor_code(final_prediction, docMetaData, VENDOR_MASTERDATA)
    final_prediction = adding_mandatory_fieldFlag(final_prediction,doc_type)
    final_prediction = removing_unnecessary_fields(final_prediction,doc_type)
    final_prediction = modifying_confidence_amount_fields(DF,final_prediction,docMetaData)
    final_prediction = reducing_confidence_additional_tax_present(final_prediction, docMetaData)
    final_prediction = checking_stp_1(final_prediction,REFERENCE_DATA,docMetaData,flag_invoice_number)
    final_prediction = cl_rules.check_if_future_Date(final_prediction)
    final_prediction = checking_grnDate_validation(final_prediction,grn_Date, date_format_flag)
    final_prediction = modify_invNum_using_ref_rpa(final_prediction,docMetaData,DF)
	# final_prediction = combining_bounding_box_stp(final_prediction,DF,REFERENCE_DATA,docMetaData,flag_invoice_number)
    stp=check_stp(final_prediction,vendor_masterdata,docMetaData)
    final_prediction = cl_rules.create_bounding_box_for_amounts(DF, final_prediction)
	# Removing STP if subtotal zero percentage is greater than zero
    stp = check_subtotal_zero(stp,final_prediction)
    print("Final stp score:",stp)
    if format_ is None:
        format_ = "UNKNOWN"
    print("VENDOR ID: ", format_)
    
    # return final_prediction,stp,ovr,format_  # NOTE: To return Overall Score
    return final_prediction,stp,crit,format_ # NOTE: To return Critical Score

def convert(o):
    if isinstance(o, numpy.int64): return int(o)

#DF = pd.read_csv(r"D:\Invoice Data Extraction\TAPP 3.0\Demos\SGI Demo\SMS_INDUSTRIES_LLC-2\UI\doc_1600406574403_9bb988b8998_pred.csv")
#
#results = post_process(DF)
def check_ocr_confidence(DF, prediction):
    """
    """

    dict_ocr_threshold = {'date_fields': 0.61, 'header_fields': 0.58, 'amount_fields' : 0.43}

    pred = {}
    threshold_ocr_confidence = 0
    for key, val in prediction.items():
        if key == "lineItemPrediction":
            pred[key] = val
        else:
            if key in FIELD['date_fields']:
                threshold_ocr_confidence = dict_ocr_threshold['date_fields']
            elif key in FIELD['header_fields']:
                threshold_ocr_confidence = dict_ocr_threshold['header_fields']
            elif key in FIELD['amount_fields']:
                threshold_ocr_confidence = dict_ocr_threshold['amount_fields']

            if val is not None:
                ocr_conf = val['conf']
                if ocr_conf < threshold_ocr_confidence:
                    print("key :",key,"val__: ",val['conf'],"\tthreshold_ocr_confidence :",threshold_ocr_confidence)
                    new_conf = (val['conf']+ val['model_confidence'])/2
                    print("Updating new confidence :",new_conf)
                    val['final_confidence_score'] = new_conf
            pred[key] = val

    return pred

#def helper_main():
#    """
#    Read entire OCR output and separate into different files
#    Extract FileName first
#    """
#    DF = pd.read_csv('SAMPLE_TRAIN_UPDATED.csv', index_col=0)
#
#    for idx, rows in DF.iterrows():
#        predicted_label = rows['predict_label']
#        prediction_probability = rows["prob_" + predicted_label]
#        DF.at[idx, 'prediction_probability'] = prediction_probability
#
#    DF[['a', 'b', 'c']] = DF['OriginalFile'].str.split('_', expand=True)
#    DF[['Page Num', 'Extension']] = DF['c'].str.split('.', expand=True)
#    DF['FileName'] = DF['a'] + '_' + DF['b'] + '.' + DF['Extension']
#
#    DF.drop(['a', 'b', 'c', 'Page Num', 'Extension'], axis=1, inplace=True)
#    print("FileName extracted!!!!")
#
#    temp = DF[['Client', 'Format', 'FileName']].drop_duplicates()
#
#    FINAL_PREDICTION = pd.DataFrame(columns=['Client', 'Format', 'FileName', 'Field',
#                                             'predicted_value', 'label_confidence', 'wordshape_confidence',
#                                             'model_probability',
#                                             'model_confidence_score', 'final_confidence_score',
#                                             'vendor_masterdata_present'])
#
#    for idx, row in temp.iterrows():
#        client = row['Client']
#        format_ = row['Format']
#        file_name = row['FileName']
#
#        temp_df = DF.loc[(DF['Client'] == client) &
#                         (DF['Format'] == format_) & (DF['FileName'] == file_name)]
#
#        print("Processing:", client, format_, file_name)
#        prediction = post_process(temp_df)
#        print(prediction)
#        print("\n")
#
#        output_json = build_final_json(prediction)
#        print(output_json)
#        """
#        with open('test_output_latest/' + client + '-' + format_ + '-' + file_name + '.json', 'w') as f:
#            json.dump(output_json, f, default=convert)
#
#        for key, value in prediction.items():
#            row = [client, format_, file_name, key]
#            if value is not None:
#                row.append(value['text'])  # predicted_value
#                row.append(value['label_confidence'])  # label_confidence
#                row.append(value['wordshape_confidence'])  # wordshape_confidence
#                row.append(value['prob_' + key])  # model_probability
#                row.append(value['model_confidence'])  # model_confidence
#                row.append(value['final_confidence_score'])  # final_confidence_score
#                row.append(value['vendor_masterdata_present'])  # final_confidence_score
#            else:
#                row.extend(['', '', '', '', '', '', ''])
#            FINAL_PREDICTION.loc[FINAL_PREDICTION.shape[0]] = row
#        FINAL_PREDICTION.to_csv("PREDICTION.csv")
#        """
#
#
#def main():
#    """
#    Script needs input folder path to iterate over
#    :return:
#    """
#    helper_main()
#    print("Exiting main!!!")
#
#
#if __name__ == "__main__":
#    main()
