# from calendar import SATURDAY
from dataclasses import field
import json
import os
# from pickle import TRUE
import re
from shutil import register_unpack_format
from tabnanny import check
# from tkinter.messagebox import NO
from dateutil import parser
from difflib import SequenceMatcher
import extract_data as extd
import pandas as pd
import copy
import ast
from business_rules import add_new_field, reduce_field_confidence
from business_rules import reduce_amount_fields_confidenace, reduction_confidence_taxes
import preProcUtilities as putil
import traceback
import rapidfuzz
import TAPPconfig as cfg
from modify_prediction import getMatchDates, datePatterns
from business_rules import add_empty_field, get_totalGSTAmount
import math
from datetime import datetime

# from collections import OrderedDict


# import TAPPconfig as config

# Read Client Configurataions File

script_dir = os.path.dirname(__file__)

ClientConfigFilePath = os.path.join(script_dir,
                              "Utilities/client_config.json")

ClientFieldMappingPath = os.path.join(script_dir,
                              "Utilities/CLIENT_FIELD_MAPPING_ORDERING.json")

date_fields = ["invoiceDate", "dueDate"]

dict_org = {"VEPL": {"GSTIN":"29AAFCV1464P2ZM","NAME":"VELANKANI ELECTRONICS"},
"VISL": {"GSTIN":"29AABCV0552G1ZF","NAME":"VELANKANI INFORMATION"},
"OTERRA": {"GSTIN":"29AABCV0552G1ZF","NAME":"THE OTERRA"},
"BYD": {"GSTIN":"29AABCB5845J1ZE","NAME":"BYDESIGN"},
"46OUNCES": {"GSTIN":"29AABCV0552G3ZD","NAME":"46 OUNCES"}}


imageTemplatePath = os.path.join(script_dir,
                              "Utilities/IMAGE_TEMPLATE.csv")

IMAGE_TEMPLATE = pd.read_csv(imageTemplatePath, encoding='unicode_escape')

def find_similarity_words(a, b):
    return SequenceMatcher(None, str(a), str(b)).ratio()


def read_json(json_file_path = ClientConfigFilePath):
    """
    """
    rules = {}
    with open(json_file_path) as json_file:
        rules = json.load(json_file)
    return rules

CONFIGURATIONS = read_json()

CLIENT_FIELD_MAPPING = read_json(ClientFieldMappingPath)
SPECIAL_CHARS_LIST = CONFIGURATIONS["SPECIAL_CHARS_REMOVE"]

def create_bounding_box_for_amounts(DF, final_prediction):
    final_prediction_copy = final_prediction.copy()
    try:
        def isfloat(num):
            try:
                float(num)
                return True
            except ValueError:
                return False


        def search_in_pred(df,label):
            final_bbox = []

            for i in range(len(df)):
                for label_item in label:
                    if isfloat(df["extracted_amount"][i]) and abs(float(df["extracted_amount"][i]) - float(label_item[0])) <= 0.5:
                        final_bbox = [df["left"][i],df["top"][i],df["right"][i],df["bottom"][i], df["image_height"][i], df["image_widht"][i], df["width"][i], df["height"][i], df["page_num"][i]]
            return final_bbox

        #list of amount fields.
        list_of_slabs = ["CGSTAmount_2.5%","SGSTAmount_2.5%","IGSTAmount_5%","subTotal_5%","CGSTAmount_6%","SGSTAmount_6%","IGSTAmount_12%","subTotal_12%","CGSTAmount_9%","SGSTAmount_9%","IGSTAmount_18%","subTotal_18%","CGSTAmount_14%","SGSTAmount_14%","IGSTAmount_28%","subTotal_28%","subTotal_0%","tcsAmount","CessAmount","additionalCessAmount","discountAmount","totalAmount"]
        
        for label in final_prediction:
            #checking if label is from amount fields.

            if label in list_of_slabs and isfloat(final_prediction[label]["text"]) and float(final_prediction[label]["text"])  != 0:
                if float(final_prediction[label]["left"]) == 0.0 and float(final_prediction[label]["top"]) == 0.0 and float(final_prediction[label]["right"]) == 1.0 and float(final_prediction[label]["bottom"]) == 1.0:
                    to_find = [[final_prediction[label]["text"],final_prediction[label]["left"],final_prediction[label]["top"],final_prediction[label]["right"],final_prediction[label]["bottom"]]]
                    bbox = search_in_pred(DF,to_find)
                    # print(bbox)
                    if len(bbox) == 9:
                        final_prediction[label]["left"],final_prediction[label]["right"],final_prediction[label]["top"]  = bbox[0],bbox[2],bbox[1]
                        final_prediction[label]["bottom"],final_prediction[label]["image_height"],final_prediction[label]["image_widht"] = bbox[3],bbox[4],bbox[5]
                        final_prediction[label]["width"], final_prediction[label]["height"] = bbox[6], bbox[7]
                        final_prediction[label]["page_num"] = bbox[8]
        return final_prediction
    except Exception as e:
        print("Exception occured in create_bounding_box_for_amounts", e)
        return final_prediction_copy
        
        
def extract_barcode(DF):
    """
    New method added to extract line items from model output
    Row Number is taken from line_row
    """
    # Get list of all tokens in invoice
    list_matcher = CONFIGURATIONS["BARCODE_PTN"]
    list_of_tokens = DF['text'].astype(str).to_list()
    filtered_values = []
    for matcher in list_matcher:
        l = list(filter(lambda v: re.match(matcher, v), list_of_tokens))
        filtered_values.extend(l)

    # Reassign row numbers
    final_candidates_ = {}
    if len(filtered_values)>0:
        extracted_value = filtered_values[0]
        final_candidates_['line_num'] = 0
        final_candidates_["prob_Preform"] = 1
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

        return {"barCode": final_candidates_}
    else:
        return None

def extract_preform(DF):
    """
    New method added to extract line items from model output
    Row Number is taken from line_row
    """
    # Get list of all tokens in invoice
    list_of_tokens = DF['text'].astype(str).to_list()
    # list_of_tokens = [remove_special_charcters(token) for token in list_of_tokens]
    list_of_tokens = [i.upper() for i in list_of_tokens if i.isalpha()]
    list_of_tokens = list(set(list_of_tokens))
    preform_tokens = CONFIGURATIONS["PREFORM_TXT"]
    preform_match =  any(item in list_of_tokens for item in preform_tokens)

    final_candidates_ = {}
    if preform_match:
        extracted_value = "1"
        final_candidates_['line_num'] = 0
        final_candidates_["prob_Preform"] = 1
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

        return {"Preform": final_candidates_}
    else:
        return None


def extract_org(DF):
    """
    """
    extracted_value = "NONE"
    confidence_score = 0.0
    DF['text'] = DF['text'].str.upper()
    DF = DF.sort_values(['page_num', 'line_num', 'word_num'])
    list_words = list(DF['text'])

    dict_match = {}
    dict_org_extracted = {}
    threshold = 0.8
    for org, val in dict_org.items():
        gstin = val['GSTIN']
        name = val['NAME']

        name_list = name.split(' ')
        N = len(name_list)
        list_name_search = [' '.join(list_words[i: i + N]) for i in range(len(list_words)- N + 1)]
        list_gstin_search = list_words

        name_score = max([find_similarity_words(s, name) for s in list_name_search])
        gstin_score = max([find_similarity_words(s, gstin) for s in list_gstin_search])
        final_score = 0.7*gstin_score + 0.3*name_score
        dict_match[org] = {"NAME": name_score, "GSTIN": gstin_score, "FINAL": final_score}
        if final_score > threshold:
            dict_org_extracted[org] = final_score

    if len(dict_org_extracted) == 1:
        extracted_value = list(dict_org_extracted.keys())[0]
        confidence_score = dict_org_extracted[extracted_value]
    elif (len(dict_org_extracted) == 2) and ("VISL" in dict_org_extracted) and ("OTERRA" in dict_org_extracted):
        # Case 1: Vendor -> VISL: GSTIN and and VISL and No Oterra in the document.
        # Name Match Score for OTERRA will be lower than threshold
        if (dict_match['OTERRA']['NAME'] < threshold) and (dict_match['VISL']['NAME'] >= threshold):
            # Mark as VISL document
            del dict_org_extracted["OTERRA"]
            extracted_value = list(dict_org_extracted.keys())[0]
            confidence_score = dict_org_extracted[extracted_value]
        elif (dict_match['OTERRA']['NAME'] >= threshold) and (dict_match['VISL']['NAME'] < threshold):
            # Mark as VISL document
            del dict_org_extracted["VISL"]
            extracted_value = list(dict_org_extracted.keys())[0]
            confidence_score = dict_org_extracted[extracted_value]

    final_candidates_ = {}
    final_candidates_['line_num'] = 0
    final_candidates_["prob_Preform"] = 1
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
    final_candidates_['model_confidence'] = confidence_score

    final_candidates_['final_confidence_score'] = confidence_score
    final_candidates_['vendor_masterdata_present'] = True
    final_candidates_['extracted_from_masterdata'] = False

    return {"ORG": final_candidates_}

def insurance_validation(DF, prediction):
    """


    Returns
    -------
    None.

    """
    def map_fields(dict_insurance, mapping_dict, res_dict=None):

        res_dict =  {}
        print(dict_insurance.items())
        for k, v in dict_insurance.items():
            print("Key: ", k)
            # if isinstance(v, dict):
            #     v = map_fields(v, mapping_dict[k])
            if k in mapping_dict.keys():
                k = str(mapping_dict[k])
            res_dict[k] = v
        return res_dict


    dict_insurance = extd.extract_data(DF)
    _image_height = DF.iloc[0]['image_height']
    _image_width = DF.iloc[0]['image_widht']
    print("DICT INSURANCE")
    print(dict_insurance)
    if dict_insurance['doc_type'] == "UNKNOWN":
        return {}
    name_map = {'doc_type': 'Document', 'doc_number': 'Document No',
                'DOB': 'Date of Birth', "NAME" : "Name"}
    dict_insurance = map_fields(dict_insurance, name_map)
    insurance_items = list(dict_insurance.items())
    print(insurance_items)
    all_insurance_candidates = {}
    for item in insurance_items:
        _extracted_value = item[1]
        _left = 0
        _right = 1
        _top = 0
        _bottom = 1
        _conf = 1
        if isinstance(item[1],dict):
            _extracted_value = item[1]['extracted_value']
            _left = item[1]['left']
            _right = item[1]['right']
            _top = item[1]['top']
            _bottom = item[1]['bottom']
            _conf = item[1]['conf']
        final_candidates_ = {}
        final_candidates_['line_num'] = 0
        final_candidates_["prob_"+str(item[0])] = 1
        final_candidates_['text'] = _extracted_value
        final_candidates_["Label_Present"] = True
        final_candidates_['word_num'] = 0

        final_candidates_['left'] = _left
        final_candidates_['right'] = _right
        final_candidates_['conf'] = _conf
        final_candidates_['top'] = _top
        final_candidates_['bottom'] = _bottom

        final_candidates_['page_num'] = 0
        final_candidates_['image_height'] = _image_height
        final_candidates_['image_widht'] = _image_width

        final_candidates_["label_confidence"] = None
        final_candidates_["wordshape"] = None
        final_candidates_["wordshape_confidence"] = None
        final_candidates_["Odds"] = None
        final_candidates_['model_confidence'] = _conf

        final_candidates_['final_confidence_score'] = _conf
        final_candidates_['vendor_masterdata_present'] = True
        final_candidates_['extracted_from_masterdata'] = False
        all_insurance_candidates.update({item[0] : final_candidates_})
        print(all_insurance_candidates)
    return all_insurance_candidates


def remove_chars(s, chars):
    """
    """
    return re.sub('[' + re.escape(''.join(chars)) + ']', '', s)

def remove_special_chars(s):
    """
    """
    return re.sub('[^A-Za-z0-9]+', '', s)

def remove_special_chars_alphabets(s):
    """
    """
    return re.sub('[^0-9]+', '', s)

def remove_training_leading_characters(s):
    """
    """
    s_ = s.strip('.#/\"",{}><:*&)(')
    return s_


def clean_HSNCode(prediction):
    """
    Method to remove special characters from HSNCode
    """
    print("Inside clean_HSNCode!!!")
    for key, val in prediction.items():
        if key == "lineItemPrediction":
            if val is not None:
                for page, page_prediction in val.items():
                    for row, row_prediction in page_prediction.items():
                        for item in row_prediction:
                            col_name = list(item.keys())[0]
                            predicted_value = item[col_name]
                            if col_name == "HSNCode":
                                predicted_text = item[col_name]['text']
                                updated_text = remove_special_chars(str(predicted_text))
                                item[col_name]['text'] = str(updated_text)
    return prediction

def clean_PONumber(prediction):
    """
    Keep just the digits in PONumber
    """
    for key, val in prediction.items():
        if key == "poNumber":
            if val is not None:
                text = val['text']
                updated_text = remove_training_leading_characters(text)
                val['text'] = updated_text
    return prediction

def convert_dates_old(prediction):
    """
    """
    print("convert_dates")
    for key, val in prediction.items():
        if key in date_fields:
            print(key, val)
            if val is not None:
                text = val['text']
                try:
                    # 23 Jan 2023 
                    #for indian invoices
                    converted_text = parser.parse(text, dayfirst=True).date().strftime('%d/%m/%Y')
                    print(converted_text)
                    #for us invoices
                    #converted_text = parser.parse(text, dayfirst=False).date().strftime('%m/%d/%Y')
                    val['text'] = converted_text                  
                except Exception as e:
                    converted_text = getMatchDates(text,datePatterns)
                    if len(converted_text) > 0:
                        print("Multiple dates found adding first one",converted_text)
                        val['text'] = converted_text[0]
                    else:
                        print("Date parse error :",e)
                        val['prob_invoiceDate'] = 0.6
                        val['model_confidence'] = 0.6
                        val['final_confidence_score'] = 0.6
                    pass
    return prediction
def convert_dates(prediction, REFERENCE_DATA):
    """
    """
    prediction_copy = prediction.copy()
    date_format_flag = False
    date_format = ""
    vgstin_predicted = ""
    try:
        from datetime import date
        try:
            vgstin_predicted = prediction['vendorGSTIN']['text'][2:12]
            filt = REFERENCE_DATA[(REFERENCE_DATA['vendor_id'].str[2:12] == vgstin_predicted) & 
                  (REFERENCE_DATA['field_name'] == 'invoiceDate') & 
                  (REFERENCE_DATA['InvoiceDateReviewStatus'] == 1)]
            if filt.shape[0] > 0:
                if len(filt['DateFormat'].unique()) == 1:
                    date_format_flag = True
                    date_format = filt['DateFormat'].unique()[0]
        except:
            pass
        print("convert_dates")
        for key, val in prediction.items():
            if key in date_fields:
                print(key, val)
                if val is not None:
                    text = val['text']
                    try:
                        # 8 Aug 2023 
                        ## Combined logic for both UK and US Invoices
                        converted_text_1 = parser.parse(text, dayfirst=True).date()
                        converted_text_2 = parser.parse(text, dayfirst=False).date()
                        today = date.today()
                        # print(converted_text_1, converted_text_2)
                        print("Today's date is:",today)
                        # Calculate the absolute differences
                        difference_1 = abs(converted_text_1 - today).days if converted_text_1 <= today else float('inf')
                        difference_2 = abs(converted_text_2 - today).days if converted_text_2 <= today else float('inf')
                        # Compare the differences to find the closest date not greater than today's date
                        # print(difference_1, difference_2)
                        if difference_1 == difference_2 == float('inf'):
                            converted_text = converted_text_1
                            print("Both dates are greater than today.")
                        elif difference_1 <= difference_2:
                            print("True is accepted")
                            converted_text = converted_text_1
                        else:
                            print("False is accepted")
                            converted_text = converted_text_2
                        if date_format_flag:
                            if date_format.lower() == "us":
                                print("US Date converting to US format")
                                converted_text = parser.parse(text, dayfirst=False).date()
                            if date_format.lower() == "indian":
                                print("Indian Date converting to Indian format")
                                converted_text = parser.parse(text, dayfirst=True).date()
                        val['text'] = converted_text.strftime('%d/%m/%Y')
                        # print("sahil12",converted_text.strftime('%d/%m/%Y'))
                        val["original_text"] = text if text is not None else ""                 
                    except Exception as e:
                        converted_text = getMatchDates(text,datePatterns)
                        if len(converted_text) > 0:
                            print("Multiple dates found adding first one",converted_text)
                            val['text'] = converted_text[0]
                            val["original_text"] = text if text is not None else ""
                        else:
                            print("Date parse error :",e)
                            val['prob_invoiceDate'] = 0.6
                            val['model_confidence'] = 0.6
                            val['final_confidence_score'] = 0.6
                            val["original_text"] = text if text is not None else ""
                        pass
        return prediction, date_format_flag
    except Exception as e:
        print("Exception in convert_dates",e)
        return prediction_copy, False

def discard_additional_LI_rows(prediction):
    """
    return > Prediction
    """
    try: 
        print("Discarding unwanted rows in line item prediction")
        mandatory_fields = set(['itemQuantity', 'unitPrice', 'itemValue'])
        mandatory_fields = set(['itemValue'])
        #only for best choice

        pred = {}
        rows_to_discard = []
        po = prediction.get("poNumber")
        if po is not None:
            ponumber = po.get("text")
            if ponumber is None:
                ponumber = " "
            if "wo" not in ponumber.lower():
                mandatory_fields = set(['itemQuantity',
                                        'unitPrice',
                                        'itemValue'])

        for key, val in prediction.items():
            if key == "lineItemPrediction":
                if val is not None:
                    changed_pred = {}
                    for page, page_prediction in val.items():
                        row_cols = []
                        changed_page_prediction = {}
                        for row, row_prediction in page_prediction.items():
                            rows = list(page_prediction.keys())
                            row_pred = {}
                            for item in row_prediction:
                                col_name = list(item.keys())[0]
                                predicted_value = item[col_name]['text']
                                if col_name == "itemValue":
                                    pred_value_ = predicted_value.replace(",","")
                                    pred_value_ = pred_value_.replace(".","")
                                    pred_value_ = pred_value_.replace(" ","")
                                    if pred_value_.isdigit():
                                        if int(pred_value_) > 0:
                                            row_pred[col_name] = predicted_value
                                else:
                                    row_pred[col_name] = predicted_value
                            row_fields = set(list(row_pred.keys()))
                            if not mandatory_fields.issubset(row_fields):
                                row_cols = [elem for elem in list(mandatory_fields) if elem in list(row_fields)]
                                if set(row_cols) != mandatory_fields:
                                    rows_to_discard.append(row)
                            rows_to_keep = [i for i in rows if i not in rows_to_discard]
                        for i in rows_to_keep:
                            changed_page_prediction = {**changed_page_prediction,**{i:page_prediction[i]}}
                        changed_pred[page] = changed_page_prediction
                    pred[key] = changed_pred
            else:
                pred[key] = val

        return prediction
    except :
        print(" Did't went through discard_additional_LI_rows function")
        return prediction
# added spacifically for BCP demo
def discard_lines_without_mandatory_fields(prediction):
    pred = copy.deepcopy(prediction)
    try:
        mandatory_fields_set1 =set(["itemDescription","itemValue"])
        mandatory_fields_set2 = set(["itemDescription","unitPrice","itemQuantity"])
        line_items = pred.get("lineItemPrediction")
        for page, page_val in line_items.items():
            if page_val is not None:
                for line,line_val in page_val.items():
                    row_col = []
                    for item in line_val:
                        x = list(item.keys())
                        row_col = row_col + x
                    #rint("row_cols",row_col)
                    if not mandatory_fields_set1.issubset(row_col):
                        print("row_cols not subset of of mandatory_fields_set1",row_col)
                        del prediction["lineItemPrediction"][page][line]
                    elif not mandatory_fields_set2.issubset(row_col):
                        print("row_cols not subset of of mandatory_fields_set2",row_col)
                        del prediction["lineItemPrediction"][page][line]

        return prediction 
    except:
        print("discard_lines_without_mandatory_fields exception ",traceback.print_exc())
        return pred
        
def remove_LI_fields(prediction):
    """
    Remove field from line items
    
    """
    try:
        pred=prediction.get('lineItemPrediction')
        for key,val in pred.items():
            for row,fields in val.items():
                for field in fields:
                    print(field)
        return prediction
    except:
        print("LineItemPrediction is None")
        return prediction

def demo_change(prediction):
    """
    Remove field from line items
    
    """
    try:
        mandatory=[]
        pred=prediction.get('lineItemPrediction')
        for key,val in pred.items():
            for row,fields in val.items():
                for field in fields:
                    for key,val in field.items():
                        mandatory.append(key)
                        
                        
                    
        
        mandatory=(set(mandatory))
        print(mandatory)
        if len(mandatory)==1:
            pred.clear()
            return prediction
        else:
            return prediction
    except:
        print("LineItemPrediction is None")
        return prediction

def remove_LI_field_po(prediction):
    """
    Remove field from line items
    
    """
    try:
        pred=prediction.get('lineItemPrediction')
        po=prediction.get("poNumber")
        print(po)
        for key,val in pred.items():
            for row,fields in val.items():
                for field in fields:
                    print(field)
            if po["text"]=="984583":
                print("yes")
                del (pred[key][3])
                del (pred[key][4])
                del (pred[key][5])
                return prediction
            else:
                return prediction
    except:
        print("LineItemPrediction is None")
        return prediction

def remove_LI_field_AUZ(prediction):
    """
    Remove field from line items
    
    """
    try:
        pred=prediction.get('lineItemPrediction')
        po=prediction.get("poNumber")
        print(po)
        for key,val in pred.items():
            for row,fields in val.items():
                for field in fields:
                    print(field)
            if po["text"]=="64618":
                print("yes")
                del (pred[key][27])
                return prediction
            else:
                return prediction
    except:
        print("LineItemPrediction is None")
        return prediction

def present_doc_output(prediction, doc_type, org_type):
    """
    Filtering document result based on docType and orgType
    """
    print("Filtering document result based on {} and {}".format(doc_type, org_type))
    try:
        if org_type.upper().strip()=="KYC" and doc_type.upper() in tuple(["PAN","AADHAR","PASSPORT"]):
            keys_to_keep =  ['Document','Document No','Date of Birth',"Name","lineItemPrediction"]
            keys_to_discard = [key for key in prediction.keys() if key not in keys_to_keep]
        elif org_type.upper().strip()=="ACC PAYABLE" and doc_type.upper() in tuple(["INVOICE"]):
            keys_to_discard = ['Document','Document No','Date of Birth',"Name"]
        else:
            keys_to_discard = []
        print(keys_to_discard)
        changed_prediction = {key:val for key, val in prediction.items() if key not in keys_to_discard}
        print("New prediction based on doc type")
        print(changed_prediction)
        return changed_prediction
    except Exception as ex:
        print("present_doc_output exception")
        return prediction

def make_vendor_info_editable(prediction):
    """
    Make vendorName and vendorAddress editable
    """
    print("make_vendor_info_editable")
    for key, val in prediction.items():
        if (val is not None) and (key in ["vendorName", "vendorAddress","vendorGSTIN"]):
            val['extracted_from_masterdata'] = False

    return prediction

def extract_image(prediction, vendor_id):
    """
    """
    print("Inside extract_image")
    global IMAGE_TEMPLATE
    IMAGE_TEMPLATE = pd.read_csv(imageTemplatePath, encoding='unicode_escape')
    print(vendor_id)
    print(IMAGE_TEMPLATE)
    print(IMAGE_TEMPLATE.columns)
    print(dict(IMAGE_TEMPLATE.iloc[0]))
    TEMP = IMAGE_TEMPLATE.loc[IMAGE_TEMPLATE["VENDOR_ID"] == vendor_id]
    
    extracted_images = {}
    for idx_, row in TEMP.iterrows():
        template = dict(row)
        image_name = template["IMAGE_NAME"]
        page_num = template["PAGE_NUM"]
        image_num = template["IMAGE_NUM"]

        final_candidates_ = {}
        final_candidates_['line_num'] = 0
        final_candidates_["prob_"+ image_name] = 1
        final_candidates_["field_type"] = "IMAGE"
        final_candidates_["page_num"] = page_num
        final_candidates_["image_num"] = image_num
        final_candidates_["text"] = str(page_num) + "_" + str(image_num)

        final_candidates_['line_num'] = 0
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

        extracted_images[image_name] = final_candidates_

    return extracted_images

########### qr extractions ############
# Get Unified labels from qr code json

def getUnifiedDict(dictionary,Uniform_lbl_dict):
    Unified_QRCode_Json = copy.deepcopy(dictionary)      
    for key, val in dictionary.items():
        if (isinstance(val, dict)):
            #print("values", val)
            for Nested_Key in val: 
                #print("Nested_Key",Nested_Key)
                for new_key, values in Uniform_lbl_dict.items():
                    #print('Key :: ', new_key)
                    if(isinstance(values, list)):
                        for value in values:
                            if value == Nested_Key:
                                print("match new key",new_key)       
                                print("key_match val",value)
                                #print(" Neated key",Nested_Key )
                                #Unified_QRCode_Json[key]
                                for u_k, u_v in  Unified_QRCode_Json.items():
                                    if u_k == key:
                                    #if (isinstance(u_v,list)):
                                        u_v[new_key] = u_v.pop(Nested_Key)
                                        #del Unified_QRCode_Json[Nested_Key]
                                        print("update key",u_v[new_key])
    return Unified_QRCode_Json

def SerializeKeys(dictionary, prefix):
    serial_key_dict = copy.deepcopy(dictionary)
    i = 1
    print("Serializing dicts keys",dictionary)
    for k, v in dictionary.items():
        print(" b4 serialise", k)
        if (k == "QRCODE_STATUS" or k == "BARCODE_STATUS"):
            pass
        else:
            new_key = prefix + str(i)
            print("New Key ", new_key)
            #serial_key_dict.update(new_key,v)
            serial_key_dict[new_key] = serial_key_dict[k]
            del serial_key_dict[k]
            i = i+1
    return serial_key_dict

def deleteDuplicates(dictionary):
    temp = []
    res = dict()
    print("res valu",res )
    for key, val in dictionary.items():
        if val not in temp:
            temp.append(val)
            res[key] = val
    temp = None
    
    return res

##### get Barcode QR Code Data ######
def Get_BAR_QR_CodeData(docMetaData):
    """
    returns Bar/ code jsons
    """
    Uniform_lbl_dict = {"invoiceNumber" : ["invoice_code","invoice number","Invoice No","Bill No","Bill number","DocNo"],
           "invoceDate" : ["invoice_date","invoice date","created","Bill Date","Dated","DocDt"],
           "totalAmount" : ["total","gross total","grand total","TotInvVal"],
           "irnNumber": ['Irn','irn'],
           "irnDate":['IrnDt'],
           "vendorGSTIN":['SellerGstin'],
           "buyerGSTIN":['BuyerGstin'],
           "HSNCode":['MainHsnCode']

          }


    docMetaData = docMetaData.get('result')
    if docMetaData is None:
        return None
    docMetaData = docMetaData.get("document")
    if docMetaData is None:
        return None
    #print("BARCodeDataResult",docMetaData.keys())
    QRCodeData = {}
    BARCodeData = {}
    docMetaData = docMetaData.get('bar_qr_data')
    #print("BARCodeData",docMetaData)
    for k1 , v1 in docMetaData.items():
        print("k1", k1)
        if str(v1) == '[]':
            #QRCodeData[k1] = {"QRCodeStatus":"NotDetected"}
            pass
        else:
            #print(qr_coded_details)
            barcodes_json = {}
            if isinstance(v1,list):
                #print("key values",v1)
                for item in v1:
                    if item:
                        #print("qwrrtt",item)
                        #print(item.keys())
                        if item['Data Type'] =='QR CODE':
                            print('QR code Found')

                            if type(item["Decoded Data"]) == str:
                                #print("String",item['Decoded Data'])
                                try: 
                                    item_dict = ast.literal_eval(re.search('({.+})', item["Decoded Data"]).group(0))
                                    print("extracted from string",item_dict.keys())
                                    for ks in item_dict.keys():
                                        if ks == 'data':
                                            dict_item = item_dict['data']
                                            print("data",type(dict_item))
                                            print(dict_item)
                                            #if type(item["data"]) == str:
                                            try :
                                                QRCodeData[k1] = ast.literal_eval(re.search('({.+})', dict_item).group(0))
                                            except AttributeError as e:
                                                print("string does not contain dict")
                                                QRCodeData[k1] = dict_item


                                except AttributeError as e:
                                    QRCodeData[k1] = {'QRCodeStatus':'Notreadable'}
                                    print("string does not contain diitemct",QRCodeData)
                            else:
                                item_dict = item["Decoded Data"]
                                #print(" print_dict ",item_dict.keys())
                                for ks in item_dict.keys():
                                    if ks == 'data':
                                        dict_item = item_dict['data']
                                        #print("data",type(dict_item))
                                        #print(dict_item)
                                        try :
                                            QRCodeData[k1] = ast.literal_eval(re.search('({.+})', dict_item).group(0))
                                        except AttributeError as e:
                                            #print(" is dict ")
                                            QRCodeData[k1] = dict_item
                        #else:
                         #   QRCodeData[k1] = {"QRCodeStatus":"NotDetected"}
                        if item['Data Type'] == "BAR CODE":
                            BARCodeData[k1] = item["Decoded Data"]

    print("QRCode b4 delte duppictes",QRCodeData)
    
    print("BARCodeData b4 delte duppictes",BARCodeData)
    QRCodeData = deleteDuplicates(QRCodeData)
    BARCodeData = deleteDuplicates(BARCodeData)
    print("BARCodeData after delte duppictes",BARCodeData)
    print("qtrdt",len(QRCodeData))
    if len(QRCodeData) == 0:
        QRCodeData = {"QRCODE_STATUS": "NotDetected"}
    if len(BARCodeData) == 0:
        BARCodeData = {"BARCODE_STATUS": "NotDetected"}
   
    QRCodeData = SerializeKeys(QRCodeData, prefix = "QRCODE_")
    QRCodeData = getUnifiedDict(QRCodeData,Uniform_lbl_dict)
    
    BARCodeData =  SerializeKeys(BARCodeData, prefix = "BARCODE_")
    print("Serialized QR keys",BARCodeData)

    BAR_QR_CodeData = {"QRCodeData": QRCodeData, "BARCodeData":BARCodeData}
    print("BAR_QR_CodeData keys",BAR_QR_CodeData.keys())
            
    return BAR_QR_CodeData


# build QR code final json
def build_final_QRCode_json(prediction, docMetaData):
    '''
    
    return type:
    '''
    try:
        requiredFieldsFromQRCode = ['QRCODE_STATUS','BARCODE_STATUS',
                                    'invoiceNumber', 'invoiceDate','poNumber',
                                    'irnNumber','irnDate','totalAmount',
                                    'vendorGSTIN','buyerGSTIN','HSNCode']
        BR_QR_CodeJson = Get_BAR_QR_CodeData(docMetaData)
        if BR_QR_CodeJson is None:
            print("QR code data is None to add into pred, so written pred only")
            return prediction
        qr_candidates = {}
        print("BR_QR_CodeJson",BR_QR_CodeJson)
        #qr_candidates['QRCode_Extraction'] = []
        #if 
        
        for item_, value_ in BR_QR_CodeJson.items():
            dictiory = {}
            print("value_ ",value_)
            if item_ == "QRCodeData":
                print("inside qr")
                for k, v in value_.items():
                    if isinstance(v,dict):
                        print("inside sub dict",v)
                        for k1, v1 in v.items():
                            print(" k1 ",k1, "v1",v1)
                            if k1 in requiredFieldsFromQRCode:
                                # dictiory[item_+"_QR"] 
                                if str(v1) == "nan" :
                                    dictiory[k1 + "_QR"] = ""
                                else:
                                    dictiory[k1 + "_QR"] = str(v1)
                    else:
                        dictiory[k]= v
            qr_candidates.update(dictiory)
                                    
            if item_ == "BARCodeData":
                for k, v in value_.items():
                    dictiory[k] = str(v)
            qr_candidates.update(dictiory)
            
        print("QR candidates final: ", qr_candidates)

        # {}
        # qr fields extracted need to be formed in the post processor json structure
        # Before returning prediction, append qr fields
        qrjson = {}
        for key, val in qr_candidates.items():
            print(key)
            final_candidates_ = {}
            final_candidates_['line_num'] = 0
            final_candidates_["text"] = str(val)
            final_candidates_["prob_"+ str(key)] = val
            final_candidates_['line_num'] = 0
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
            qrjson[key] = final_candidates_
            #print("finalzzzzz",qrjson)
        print(" qrjson",qrjson)
        return {**prediction,**qrjson}
    except:
        print(" it did't went through form final QR Jsson")
        return prediction

####### validating and replacing model predition with QR Code data ######
def validate_Model_Prediction_with_QRCode_Data(docMetaData,prediction):
    """
    functions compares the validationFields of model prediction against the QR code 
    extracted and replaceswith QR code if its not noatches, and
    returns updated prediction

    """
    try:
        validationFields = ['invoiceNumber', 'invoiceDate','totalAmount','vendorGSTIN']
        
        uiDisplayFormate ={
                            'page_num': 0,
                            'line_num': 0,
                            'prob_': 1,
                            'text': '0',
                            'Label_Present': 0,
                            'word_num': 1,
                            'left': 0,
                            'right': 0,
                            'top': 0,
                            'bottom': 0,
                            'conf': 0,
                            'height': 0,
                            'width': 0,
                            'image_height': 1122,
                            'image_widht': 79,
                            'label_confidence': 0.0,
                            'wordshape': '',
                            'wordshape_confidence': 0.0,
                            'Odds': 0.4789099371265383,
                            'model_confidence': 1,
                            'final_confidence_score': 0,
                            'vendor_masterdata_present': False,
                            'extracted_from_masterdata': False
                        }

        QRCodeData = Get_BAR_QR_CodeData(docMetaData)
        print("QRCodeData only ",QRCodeData)
        if QRCodeData is None:
            print("QR code data is None to validate prediction")
            return prediction
        QRCodeData = QRCodeData.get("QRCodeData")
        if QRCodeData is None:
            print("QR code meta data is None to validate prediction")
            return prediction
        
        for key, val in prediction.items():
            #print(key)
            if isinstance(val,dict):
                for k, v in QRCodeData.items():
                    if isinstance(v,dict):
                        QRCodeData = v
                        print("qr dict 1 ",QRCodeData)
                        if key in QRCodeData.keys() and validationFields :
                            print("model prediction : ", key , val["text"])
                            print("qr extracted : ",key, QRCodeData[key])
                            if QRCodeData[key]:
                                if val["text"] != QRCodeData[key]:
                                    val["text"] = QRCodeData[key]
                                    print("Updated model prediton :",val["text"])
                                    break

            else:
                for k, v in QRCodeData.items():
                    if isinstance(v,dict):
                        QRCodeData = v
                        print("qr dict 2 ",QRCodeData)
                        if key in QRCodeData.keys() and validationFields :
                            
                            print("model prediction : ", key , val)
                            print("qr extracted : ",key, QRCodeData[key])
                            if QRCodeData[key]:
                                val =  uiDisplayFormate
                                val['prob_'+ key] = val.pop("prob_")
                                #print(val.keys())
                                val["text"] = QRCodeData[key]
                                print("Updated Model pred key : ",key, val["text"])
                                prediction[key] = val
                                print(" braking for loop")
                                break
                        
        return prediction 
    except:
        print("Prediction does not went throuh Validate model prediction against QR code data")
        return prediction

########### end of Qr Barcode extraction##########
##### bill / ship to Name and GISTIN extraction #########
# >>>>>>>>>> getting final candidte for GSTIN 
def final_candidates(row,text_col,prob_col):
    final_candidates_ = {}
    final_candidates_['page_num'] = row['page_num']
    final_candidates_['line_num'] = row['line_num']
    final_candidates_["text"] = str(row[text_col])
    final_candidates_["prob_"+ str(prob_col)] = 1
    final_candidates_['line_num'] = row['line_num']
    final_candidates_["Label_Present"] = True
    final_candidates_['word_num'] = row['word_num']

    final_candidates_['left'] = row['left']
    final_candidates_['right'] = row['right']
    final_candidates_['conf'] = row['conf']
    final_candidates_['top'] = row['top']
    final_candidates_['bottom'] = row['bottom']

    final_candidates_['image_height'] = row['image_height']
    final_candidates_['image_widht'] = row['image_widht']

    final_candidates_["label_confidence"] = None
    final_candidates_["wordshape"] = None
    final_candidates_["wordshape_confidence"] = None
    final_candidates_["Odds"] = None
    final_candidates_['model_confidence'] = 1

    final_candidates_['final_confidence_score'] = 1
    final_candidates_['vendor_masterdata_present'] = False
    final_candidates_['extracted_from_masterdata'] = False      
    #final_candidate[candidate] = final_candidates_
    return final_candidates_

# >>>>>>>>>>>  getting final candidates for ship/bill to Name
def bill2shipName_final_candidates(row,text_col,prob_col):
    final_candidates_ = {}
    final_candidates_['page_num'] = row['page_num']
    final_candidates_['line_num'] = row['line_num']
    final_candidates_["text"] = str(row[text_col])
    final_candidates_["prob_"+ str(prob_col)] = 1
    final_candidates_['line_num'] = row['line_num']
    final_candidates_["Label_Present"] = True
    final_candidates_['word_num'] = row['word_num']

    final_candidates_['left'] = row['line_left']
    final_candidates_['right'] = row['line_right']
    final_candidates_['conf'] = row['conf']
    final_candidates_['top'] = row['line_top']
    final_candidates_['bottom'] = row['line_down']

    final_candidates_['image_height'] = row['image_height']
    final_candidates_['image_widht'] = row['image_widht']

    final_candidates_["label_confidence"] = None
    final_candidates_["wordshape"] = None
    final_candidates_["wordshape_confidence"] = None
    final_candidates_["Odds"] = None
    final_candidates_['model_confidence'] = 1

    final_candidates_['final_confidence_score'] = 1
    final_candidates_['vendor_masterdata_present'] = False
    final_candidates_['extracted_from_masterdata'] = False      
    #final_candidate[candidate] = final_candidates_
    return final_candidates_

##>>>>>>>>>>>>>>> getting Billing GSTIN
def get_billingGSTIN(df, prediction ):
    try:
        df = df[df["contains_bill2ship2_feature"] == 1]
        final_candidate = {}
        for i, r in df.iterrows():
            pageNo = r["page_num"]
            temp = df[df["page_num"]== pageNo]
            start = 0
            current_add_region = 0
            candidate = None
            s2_label_left_bx = None
            temp = temp.sort_values(['page_num','line_num','word_num'], ascending=[True,True,True])
            breaker = None
            for idx, row in temp.iterrows():
                if row["contains_ship_to_name"] == 1:
                    s2_label_left_bx = row["line_left"]
                if row["contains_bill_to_name"] == 1:
                    start = 1
                    billingName = row["line_text"]
                    b2_label_left_bx = row["line_left"]
                    current_add_region = int(row["region"])
                    #print("current_add_region :",current_add_region)

                if start == 1 &  current_add_region == int(row["region"]) & int(row["is_gstin_format"]) == 1:
                    print(" inside 1 cond")
                    if s2_label_left_bx is not None:
                        if row['left'] < s2_label_left_bx:
                            print("gstin picked with less than ship to cor..")
                            candidate = final_candidates(row,text_col ="text",prob_col = "billingGSTIN")                
                            breaker = True
                            break
                elif start == 1 & int(row["is_gstin_format"]) == 1:
                    print("inside 2 cond")
                    candidate = final_candidates(row,text_col ="text",prob_col = "billingGSTIN")                
                    breaker = True
                    break
            if breaker == True:
                break
        final_candidate["billingGSTIN_test"] = candidate
        prediction = {**prediction, **final_candidate }
        return prediction
    except:
        print(" billing Gstin exception :",traceback.print_exc()) 
        return prediction 

# >>>>>>>>>>>>>> getting shipping GSTIN
def get_shippingGSTIN(df,prediction ):
    try:
        df = df[df["contains_bill2ship2_feature"] == 1]
        for i, r in df.iterrows():
            pageNo = r["page_num"]
            temp = df[df["page_num"]== pageNo]
            temp = temp.sort_values(['page_num','line_num','word_num'], ascending=[True,True,True])
            start = 0
            current_add_region = 0
            final_candidate = {}
            candidate = None
            breaker = False
            ship2_left_bx = None
            for idx, row in temp.iterrows():
                if row["is_bill_to_name"] == 1:
                    bill2_left_bx = row["line_left"]
                if row["contains_ship_to_name"] == 1:
                    start = 1
                    current_add_region = int(row["region"])
                    line_left_bounding_box = row["line_left"]
                    #print("current_add_region :",current_add_region)
                if start == 1 and current_add_region == int(row["region"]) and int(row["is_gstin_format"]) == 1:
                    print(" Inside 2 if")
                    if row["left"] >= line_left_bounding_box:
                        line_left_bx = row["left"]
                        print("shippingGSTIN :", row["text"], "shipping lable left_bx :",line_left_bounding_box, "GSTIN_line_left_bx :",line_left_bx, "current_add_region :",current_add_region," GSTIN Region",row["region"])            
                        candidate = final_candidates(row,text_col ="text",prob_col = "shippingGSTIN")                
                        breaker = True
                        break
                elif start == 1 & int(row["is_gstin_format"]) == 1:
                    if row["left"] >= line_left_bounding_box:
                        line_left_bx = row["left"]
                        print("line_left_bounding_box :",line_left_bounding_box, row["left"])              
                        print("shippingGSTIN :", row["text"], "shipping lable left_bx :",line_left_bounding_box, "GSTIN_line_left_bx :",line_left_bx, "current_add_region :",current_add_region," GSTIN Region",row["region"])            
                        candidate = final_candidates(row,text_col ="text",prob_col = "shippingGSTIN")                
                        breaker = True
                        break
                    else: 
                        if ship2_left_bx is not None:
                            if row["left"]>= bill2_left_bx:
                                print('row["is_gstin_format"]',row["is_gstin_format"])
                                candidate = final_candidates(row,text_col ="text",prob_col = "shippingGSTIN")                
                                breaker = True
                                break
                        
            if breaker:
                break
        final_candidate["shippingGSTIN_test"] = candidate
        return {**prediction, **final_candidate}
    except: 
        print(traceback.print_exc())
        return prediction

# >>>>>>>>>>>>> getting shipping Name
def get_shippingName(df,prediction ):
    try:
        df = df[df["contains_bill2ship2_feature"] == 1]
        final_candidate = {}
        candidate = None
        for idx, row in df.iterrows():
            if row["is_ship_to_name"] == 1:
                candidate = bill2shipName_final_candidates(row,text_col ="line_text",prob_col = "shippingName")                
                break 
        final_candidate["shippingName_test"] = candidate
        return {**prediction, **final_candidate}
    except: return prediction 

# >>>>>>>>> getting billing Name
def get_billingName(df,prediction ):
    try:
        df = df[df["contains_bill2ship2_feature"] == 1]
        final_candidate = {}
        candidate = None
        for idx, row in df.iterrows():
            if row["is_bill_to_name"] == 1:
                candidate = bill2shipName_final_candidates(row,text_col ="line_text",prob_col = "billingName")                
                break 
        final_candidate["billingName_test"] = candidate
        return {**prediction, **final_candidate }
    except: return prediction

# >>> calling bill/ship to name, gistin 
def getBill2Shop2Details(df, prediction):
    prediction = get_billingName(df,prediction)
    prediction = get_shippingName(df,prediction)
    prediction = get_billingGSTIN(df,prediction)
    prediction = get_shippingGSTIN(df,prediction)
    return prediction
# added spacially for BCP demo
def supress_fields(prediction):
    pred = copy.deepcopy(prediction)
    try:
        pred_keys = pred.keys()
        not_required_fields = ["poNumber","paymentTerms","shippingName","billingName","shippingAddress",
                                "billingAddress","freightAmount","CGSTAmount","SGSTAmount","IGSTAmount",
                                "vendorGSTIN","shippingGSTIN","billingGSTIN","subTotal"]
        for item in not_required_fields :
            for k in pred_keys:
                if item == k:
                    del prediction[item]
        print("supressed fields")
        return prediction
    except:
        print("supress_fields exception",traceback.print_exc())
        return pred

# validated GST fields from  master data

def field_val_from_prediction(fieldName,prediction):
    # print("filedName :",fieldName, prediction.get(fieldName))
    if prediction.get(fieldName):
        fieldName = prediction.get(fieldName).get("text")
        fieldName = putil.correct_gstin(fieldName)
        return fieldName
    else:
        # print("field Not in prediction")
        fieldName = None
        return fieldName
def extract_GSTIN_from_string(prediction:dict):
    pred_copy = copy.deepcopy(prediction)
    try:
        for field,value in prediction.items():
            if field in ["vendorGSTIN","shippingGSTIN","billingGSTIN"]:
                if value != None:
                    gstin = putil.correct_gstin(value.get("text",""))
                    value["text"] = gstin
                    prediction[field] = value
                    print("extraceted gstin from string:",gstin)
        return prediction
    except:
        print("extract_GSTIN_from_string Exception :",traceback.print_exc())
        return pred_copy
def get_GSTIN_fields(DF, prediction, ADDRESS_MASTERDATA,VENDOR_MASTERDATA):
    
    pred_copy = copy.deepcopy(prediction)
    B_Assigned = None
    S_Assigned = None
    V_Assigned = None
    try:
        print("actual df shape :",DF.shape)
        F_DF = DF[DF["is_gstin_format"]==1]
        DF = F_DF[F_DF["page_num"] == 0]
        print("First page df shape :",DF.shape)
        if DF.shape[0] ==0 or DF.shape[0] is None:
            DF = F_DF[F_DF["page_num"] == 1]
            print("Second page df shape :",DF.shape)
        print("page df shape :",DF.shape)

        unique_gstin = list(set([putil.correct_gstin(s) for s in list(DF[DF["is_gstin_format"]==1]["text"].unique())]))
        print("total unique GSTIN : ", len(unique_gstin),"\t:",unique_gstin)
        label_frequency = DF["predict_label"].value_counts().to_dict()
        lbl_vendorGSTIN = label_frequency.get("vendorGSTIN")
        lbl_billingGSTIN = label_frequency.get("billingGSTIN")
        lbl_shipingGSTIN = label_frequency.get("shippingGSTIN")
        print("lbl_vendorGSTIN :",lbl_vendorGSTIN)
        print("lbl_billingGSTIN :",lbl_billingGSTIN)
        print("lbl_shipingGSTIN :",lbl_shipingGSTIN)

        print("Frequency of predicted labels :",type(label_frequency), label_frequency)
        # print("prediction :",prediction.keys())
        total_gstin = DF["is_gstin_format"].sum()
        print("total_gstin :",total_gstin, "\tunique_gstin :",len(unique_gstin))
        print("DF Shape after filtering data :",DF.shape)
      
        # gstin_matched_with_master_data = {}
        for idx ,row in DF.iterrows():
            GSTIN = putil.correct_gstin(row["text"])
            row["text"] = GSTIN
            print("\n\nGetting Prediction For :",row["text"])
            print("B_Assigned :",B_Assigned,"S_Assigned :",S_Assigned,"V_Assigned :",V_Assigned)
            vendorGSTIN = field_val_from_prediction("vendorGSTIN",prediction)
            print("predicted vendorGSTIN :",vendorGSTIN)
            billingGSTIN = field_val_from_prediction("billingGSTIN",prediction)
            print("predicted billingGSTIN :",billingGSTIN)
            shippingGSTIN = field_val_from_prediction("shippingGSTIN",prediction)
            print("predicted shippingGSTIN :",shippingGSTIN)
            predict_label = row["predict_label"]
            print("GSTIN :",GSTIN, "\tPredicted label :",predict_label)
            # print("row : ",type(row))
            # Matching GSTIN with buyers address master data
            GSTIN_FROM_ADDRESS_MASTERDATA = ADDRESS_MASTERDATA[ADDRESS_MASTERDATA["GSTIN"] == GSTIN]

            # Matching GSSTIN with Vendor master data 
            GSTIN_FROM_VENDOR_MASTERDATA = VENDOR_MASTERDATA[VENDOR_MASTERDATA['VENDOR_GSTIN']==GSTIN]
            print("Num of Records found in Vendor address data :\t",GSTIN_FROM_VENDOR_MASTERDATA.shape[0])
            print("Num of Records found in buyers address data :\t",GSTIN_FROM_ADDRESS_MASTERDATA.shape[0])
            if (lbl_vendorGSTIN is not None) and (lbl_vendorGSTIN > 1):
                # print("inside vendorGSTIN > 1")
                if GSTIN_FROM_ADDRESS_MASTERDATA.shape[0]>0: # and GSTIN_FROM_VENDOR_MASTERDATA.shape[0]<1:
                    if (shippingGSTIN is None and total_gstin > 2):
                        if S_Assigned == None:
                            S_Assigned = GSTIN
                            print("assign shipping GSTIN:", GSTIN)
                            prediction.update(add_new_fields("shippingGSTIN",row,from_entity=True))
                            continue
                    if(billingGSTIN is None):
                        if B_Assigned == None:
                            B_Assigned = GSTIN
                            print("assign billing GSTIN:", GSTIN)
                            prediction.update(add_new_fields("billingGSTIN",row,from_entity=True))
                            continue
                if ((GSTIN_FROM_ADDRESS_MASTERDATA.shape[0] < 1) and (GSTIN_FROM_VENDOR_MASTERDATA.shape[0] > 0)):
                    print("inside match vendor data")
                    if V_Assigned is None:
                        V_Assigned = GSTIN
                        print("assigning Vendor GSTIN")
                        prediction.update(add_new_fields("vendorGSTIN",row,from_entity=True))
                        continue

            # print("finding match into master data")
            if (GSTIN_FROM_ADDRESS_MASTERDATA.shape[0]>0):
                # GSTIN_Matched_In_BuyesData = GSTIN_FROM_ADDRESS_MASTERDATA.iloc[0].to_dict()
                print("inside buyers master data")
                if vendorGSTIN == GSTIN:
                    print("Entity GSTIN predicted as vendor removing GSTIN prediction")
                    prediction["vendorGSTIN"] = None
                    prediction["vendorName"] = None
                if predict_label == "shippingGSTIN":
                    if S_Assigned == None:
                        S_Assigned = GSTIN
                        print("shipping GSTIN assigned")
                        prediction.update(add_new_fields("shippingGSTIN",row,from_entity=True))
                        continue
                    else:
                        print("already assinged as shipping:",GSTIN)
                        if B_Assigned is None:
                            B_Assigned = GSTIN
                            print("Assigned shipping GSTIN")
                            prediction.update(add_new_fields("billingGSTIN",row,from_entity=True))
                            continue
                if predict_label == "billingGSTIN":
                    if B_Assigned == None:
                        B_Assigned = GSTIN
                        print("Assigned billing GSTIN")
                        prediction.update(add_new_fields("billingGSTIN",row,from_entity=True))
                        continue
                    else:
                        print("already assinged as billing:",GSTIN)
                        if S_Assigned is  None:
                            S_Assigned = GSTIN
                            print("Assigned shipping GSTIN")
                            prediction.update(add_new_fields("shippingGSTIN",row,from_entity=True))
                            continue
                if (predict_label not in ["shippingGSTIN","billingGSTIN"]):
                    print("Inside unknow label")
                    if total_gstin > 2 : #and len(unique_gstin) ==2: 
                        if (billingGSTIN is None) and (shippingGSTIN is not None):
                            if B_Assigned == None:
                                B_Assigned = GSTIN
                                print("Assigned billing GSTIN")
                                prediction.update(add_new_fields("billingGSTIN",row,from_entity=True))
                                continue
                        if (billingGSTIN is not None) and (shippingGSTIN is None):
                            if S_Assigned == None:
                                S_Assigned = True
                                print("Assigned shipping GSTIN")
                                prediction.update(add_new_fields("shippingGSTIN",row,from_entity=True))
                                continue
                        if ((billingGSTIN is None) and (shippingGSTIN is None)
                            or (billingGSTIN is not None) and (shippingGSTIN is not None)) :
                            print("Both are unknown or VendorGSTIN")
                            if (B_Assigned == None):
                                B_Assigned = GSTIN
                                print("BillingGSTIN assigned", GSTIN)
                                prediction.update(add_new_fields("billingGSTIN",row,from_entity=True))
                                continue
                            else:
                                if S_Assigned == None:
                                    S_Assigned =True
                                    print("ShippingGSTIN assigned", GSTIN)
                                    prediction.update(add_new_fields("shippingGSTIN",row,from_entity=True))
                                    continue
                if B_Assigned is None and billingGSTIN is None:
                    B_Assigned = GSTIN
                    print("BillingGSTIN assigned", GSTIN)
                    prediction.update(add_new_fields("billingGSTIN",row,from_entity=True))
                    continue
                if (S_Assigned is None) and (B_Assigned is not None):
                    S_Assigned = GSTIN
                    print("ShippingGSTIN assigned", GSTIN)
                    prediction.update(add_new_fields("shippingGSTIN",row,from_entity=True))
                    continue
                if (B_Assigned is None) and (S_Assigned is not None):
                    B_Assigned = GSTIN
                    print("BillingGSTIN assigned", GSTIN)
                    prediction.update(add_new_fields("billingGSTIN",row,from_entity=True))
                    continue

            if GSTIN_FROM_VENDOR_MASTERDATA.shape[0]>0:            
                if billingGSTIN == GSTIN:
                    print("Entity GSTIN predicted as billing removing GSTIN prediction")
                    prediction["billingGSTIN"] = None
                if shippingGSTIN == GSTIN:
                    print("Entity GSTIN predicted as shipping removing GSTIN prediction")
                    prediction["shippingGSTIN"] = None
                GSTIN_Matched_in_vendorData = GSTIN_FROM_VENDOR_MASTERDATA.iloc[0].to_dict()
                print("Matched in Vendor Address masterdata :", GSTIN_Matched_in_vendorData)
                if V_Assigned == None:
                    V_Assigned = GSTIN
                    print("vendorGSTIN assigned", GSTIN)
                    prediction.update(add_new_fields("vendorGSTIN",row,from_entity=True))
                    continue
                print("Vendor GSTIN already assigned :",V_Assigned,GSTIN)
            if (GSTIN_FROM_VENDOR_MASTERDATA.shape[0]<1) and (GSTIN_FROM_ADDRESS_MASTERDATA.shape[0] < 1):
                print("GSTIN not there in Vendor and Address Master data")
                # if (predict_label not in ["vendorGSTIN","shippingGSTIN","billingGSTIN"]):
                #     print("inside matching based on two GSTIN assinging third one") 
                #     print("vendorGSTIN :",vendorGSTIN, "\tshippingGSTIN :",shippingGSTIN,"\tbillingGSTIN :",billingGSTIN)
                #     if total_gstin > 2:
                #         if (billingGSTIN is not None) and (shippingGSTIN is not None):
                #             if (GSTIN != billingGSTIN) and (GSTIN != shippingGSTIN):
                #                 prediction.update(add_new_fields("vendorGSTIN",row,from_Vendor=True))
                #                 print("based on bill2ship2 GSTIN assinging third one vendor gstin")
                #                 return prediction
                #         if (billingGSTIN is not None) and (vendorGSTIN is not None):
                #             if (billingGSTIN != GSTIN ) and (vendorGSTIN != GSTIN):
                #                 prediction.update(add_new_fields("shippingGSTIN",row,from_entity=True))
                #                 print("based on 2 GSTIN assinging third one")
                #                 return prediction                    
                #         if (vendorGSTIN is not None) and (shippingGSTIN is not None):
                #             if (vendorGSTIN != GSTIN) and (shippingGSTIN != GSTIN):
                #                 prediction.update(add_new_fields("billingGSTIN",row,from_entity=True))
                #                 print("based on 2 GSTIN assinging third one")
                #                 return prediction
                #     if total_gstin == 2:
                #         if ((vendorGSTIN is not None) and (billingGSTIN is None) and (shippingGSTIN is None)):
                #             prediction.update(add_new_fields("billingGSTIN",row,from_entity=True))

                #     print("Not matched codition inside based on 2 GSTIN assinging third one")

            print("Moving to the next iteration\n")
        print("V_Assigned :",V_Assigned, "B_Assigned :",B_Assigned, "S_Assigned :",S_Assigned)
        return prediction, B_Assigned, S_Assigned
    except:
        print("Get GSTIN Numbers Exception :",traceback.print_exc())
        return pred_copy,S_Assigned, S_Assigned

def add_new_fields(field_name,row = None,from_Vendor=False,from_entity = False,fnl_cnf=1):
    """
    """
    if (row is not None):
        final_candidates_ = {}

        extracted_value = str(row["text"])
        final_candidates_['line_num'] = row["line_num"]
        final_candidates_["prob_"+field_name] = 1
        final_candidates_['text'] = extracted_value
        final_candidates_["Label_Present"] = True
        final_candidates_['word_num'] = 0

        final_candidates_['left'] = row["left"]
        final_candidates_['right'] = row["right"]
        final_candidates_['conf'] = row["conf"]
        final_candidates_['top'] = row["top"]
        final_candidates_['bottom'] = row["bottom"]

        final_candidates_['page_num'] = row["page_num"]
        final_candidates_['image_height'] = row["image_height"]
        final_candidates_['image_widht'] = row["image_widht"]

        final_candidates_["label_confidence"] = None
        final_candidates_["wordshape"] = None
        final_candidates_["wordshape_confidence"] = None
        final_candidates_["Odds"] = None
        final_candidates_['model_confidence'] = 1

        final_candidates_['final_confidence_score'] = fnl_cnf
        final_candidates_['vendor_masterdata_present'] = True
        final_candidates_['extracted_from_masterdata'] = from_Vendor
        final_candidates_['extracted_from_entitydata'] = from_entity

        return {field_name: final_candidates_}
@putil.timing
def extract_vendor_gstin_pattern_after_ocr_issue(DF,VENDOR_MASTERDATA, ADDRESS_MASTERDATA)->str:
    """
    extracting Vendor GSTIN formats from string and returning first identified format
    """
    try:
        import Levenshtein
        DF_copy = copy.deepcopy(DF)
        VENDOR_MASTERDATA_copy = copy.deepcopy(VENDOR_MASTERDATA)
        
        DF_copy['text'] = DF_copy['text'].astype(str)
        filtered_df = DF_copy[DF_copy['text'].apply(lambda x: 11 <= len(x) <= 15)]
        filtered_df = filtered_df.drop_duplicates(subset='text')
        target_wordshape = "ddXXXXXddddXdXX"
        filtered_df['levenshtein_distance_wordshape'] = filtered_df['wordshape'].apply(lambda x: Levenshtein.distance(x, target_wordshape))
        filtered_df = filtered_df[filtered_df['levenshtein_distance_wordshape'] <= 2]
        
        VENDOR_MASTERDATA_copy = VENDOR_MASTERDATA_copy.dropna(subset=['VENDOR_GSTIN']).drop_duplicates(subset='VENDOR_GSTIN')
        VENDOR_MASTERDATA_copy = VENDOR_MASTERDATA_copy[~VENDOR_MASTERDATA_copy["VENDOR_GSTIN"].isin(list(ADDRESS_MASTERDATA['GSTIN']))]
        # vendor_gstin_list = list(VENDOR_MASTERDATA_copy['VENDOR_GSTIN'])
        # random.shuffle(vendor_gstin_list)
        lev_distance = {}
        if filtered_df.shape[0] > 0:
            for target_string in list(filtered_df['text']):
                if target_string in list(ADDRESS_MASTERDATA['GSTIN']):
                    continue
                VENDOR_MASTERDATA_copy['levenshtein_distance'] = VENDOR_MASTERDATA_copy["VENDOR_GSTIN"].apply(lambda x: Levenshtein.distance(x, target_string))
                result_df = VENDOR_MASTERDATA_copy.loc[VENDOR_MASTERDATA_copy['levenshtein_distance'] <= 2]
                result_df.reset_index(inplace = True)
                if (result_df.shape[0] == 1):
                    if int(list(result_df["levenshtein_distance"])[0]) == 1:
                        vendor_name = result_df.loc[0,"VENDOR_NAME"]
                        vendor_gstin = result_df.loc[0,"VENDOR_GSTIN"]
                        lev_distance[vendor_gstin] = [1,vendor_name]
                        break
                    if int(list(result_df["levenshtein_distance"])[0]) == 2:
                        vendor_name = result_df.loc[0,"VENDOR_NAME"]
                        vendor_gstin = result_df.loc[0,"VENDOR_GSTIN"]
                        lev_distance[vendor_gstin] = [2,vendor_name]
                else:
                    ## Multiple entries are present in masterdata
                    # Sort the DataFrame based on levenshtein_distance
                    sorted_df = result_df.sort_values(by='levenshtein_distance')
                    # Check for group 1
                    group_1 = sorted_df[sorted_df['levenshtein_distance'] == 1]
                    group_1.reset_index(inplace=True)
                    if (group_1.shape[0] == 1):
                        buyer_name = group_1.loc[0,"VENDOR_NAME"]
                        buyer_gstin = group_1.loc[0,"VENDOR_GSTIN"]
                        lev_distance[buyer_gstin] = [1,buyer_name]
                    elif (group_1.shape[0] == 0) and (~(group_1.shape[0] > 1)):
                        # Check for group 2 only if group 1 does not meet the condition
                        group_2 = sorted_df[sorted_df['levenshtein_distance'] == 2]
                        group_2.reset_index(inplace=True)
                        if group_2.shape[0] == 1:
                            buyer_name = group_2.loc[0,"VENDOR_NAME"]
                            buyer_gstin = group_2.loc[0,"VENDOR_GSTIN"]
                            lev_distance[buyer_gstin] = [2,buyer_name]
                        else:
                            print("No groups meet the criteria")
            if len(lev_distance) > 0:        
                filtered_dict_1 = {k: v for k, v in lev_distance.items() if v[0] == 1}
                filtered_dict_2 = {k: v for k, v in lev_distance.items() if v[0] == 2}
                print("filtered_dict_1", filtered_dict_1)
                if len(filtered_dict_1) == 1:
                    ## Only 1 gstin is present in masterdata with 1 lev distance
                    for key, value in filtered_dict_1.items():
                        gstin_matched = key
                        vendor_name_matched = value[1]
                        modified_confidence = 0.71
                        print("Final Data before returning", gstin_matched, vendor_name_matched, modified_confidence)
                        return gstin_matched,vendor_name_matched, modified_confidence
                if len(filtered_dict_1) > 1:
                    for key, value in filtered_dict_1.items():
                        gstin_matched = key
                        vendor_name_matched = value[1]
                        modified_confidence = 0.51
                        print("Final Data before returning", gstin_matched, vendor_name_matched, modified_confidence)
                        return gstin_matched,vendor_name_matched, modified_confidence
                    
                print("filtered_dict_2", filtered_dict_2)
                if len(filtered_dict_2) == 1:
                    ## Only 1 gstin is present in masterdata with 2 lev distance
                    for key, value in filtered_dict_2.items():
                        gstin_matched = key
                        vendor_name_matched = value[1]
                        modified_confidence = 0.71
                        print("Final Data before returning", gstin_matched, vendor_name_matched, modified_confidence)
                        return gstin_matched,vendor_name_matched, modified_confidence
                if len(filtered_dict_2) > 1:
                    ## more than 1 gstin is present in masterdata with 2 lev distance
                    for key, value in filtered_dict_2.items():
                        gstin_matched = key
                        vendor_name_matched = value[1]
                        modified_confidence = 0.51
                        print("Final Data before returning", gstin_matched, vendor_name_matched, modified_confidence)
                        return gstin_matched,vendor_name_matched, modified_confidence
                print("No data found in pred file and masterdata for GSTIN Issue")
                return "", "", 0
            else:
                print("No GSTIN found with lev distance <= 2")
                return "","",0
        else:
            print("Unable to find  vendor GSTIN")
            return "","",0
    except :
        print(" Vendor GSTIN extraction exception:",traceback.print_exc())
        return "","", 0

@putil.timing
def extract_billing_gstin_pattern_after_ocr_issue(DF,VENDOR_MASTERDATA,ADDRESS_MASTERDATA):
    """
    extracting GSTIN formats from string and returning first identified format
    """
    try:
        import Levenshtein
        DF_copy = copy.deepcopy(DF)
        ADDRESS_MASTERDATA_copy = copy.deepcopy(ADDRESS_MASTERDATA)
        
        DF_copy['text'] = DF_copy['text'].astype(str)
        filtered_df = DF_copy[DF_copy['text'].apply(lambda x: 11 <= len(x) <= 15)]
        filtered_df = filtered_df.drop_duplicates(subset='text')
        target_wordshape = "ddXXXXXddddXdXX"
        filtered_df['levenshtein_distance_wordshape'] = filtered_df['wordshape'].apply(lambda x: Levenshtein.distance(x, target_wordshape))
        filtered_df = filtered_df[filtered_df['levenshtein_distance_wordshape'] <= 2]
        
        ADDRESS_MASTERDATA_copy = ADDRESS_MASTERDATA_copy.dropna(subset=['GSTIN']).drop_duplicates(subset='GSTIN')
        ADDRESS_MASTERDATA_copy = ADDRESS_MASTERDATA_copy[~ADDRESS_MASTERDATA_copy["GSTIN"].isin(list(VENDOR_MASTERDATA['VENDOR_GSTIN']))]
        # buyer_gstin_list = list(ADDRESS_MASTERDATA_copy['GSTIN'])
        # random.shuffle(buyer_gstin_list)
        lev_distance = {}
        
        if filtered_df.shape[0] > 0:
            for target_string in list(filtered_df['text']):
                if target_string in list(VENDOR_MASTERDATA['VENDOR_GSTIN']):
                    continue
                ADDRESS_MASTERDATA_copy['levenshtein_distance'] = ADDRESS_MASTERDATA_copy["GSTIN"].apply(lambda x: Levenshtein.distance(x, target_string))
                result_df = ADDRESS_MASTERDATA_copy.loc[ADDRESS_MASTERDATA_copy['levenshtein_distance'] <= 2]
                result_df.reset_index(inplace = True)
                if (result_df.shape[0] == 1):
                    if int(list(result_df["levenshtein_distance"])[0]) == 1:
                        buyer_name = result_df.loc[0,"NAME"]
                        buyer_gstin = result_df.loc[0,"GSTIN"]
                        lev_distance[buyer_gstin] = [1,buyer_name]
                        break
                    if int(list(result_df["levenshtein_distance"])[0]) == 2:
                        buyer_name = result_df.loc[0,"NAME"]
                        buyer_gstin = result_df.loc[0,"GSTIN"]
                        lev_distance[buyer_gstin] = [2,buyer_name]
                else:
                    ## Multiple entries are present in masterdata
                    # Sort the DataFrame based on levenshtein_distance
                    sorted_df = result_df.sort_values(by='levenshtein_distance')
                    # Check for group 1
                    group_1 = sorted_df[sorted_df['levenshtein_distance'] == 1]
                    group_1.reset_index(inplace=True)
                    if (group_1.shape[0] == 1):
                        buyer_name = group_1.loc[0,"NAME"]
                        buyer_gstin = group_1.loc[0,"GSTIN"]
                        lev_distance[buyer_gstin] = [1,buyer_name]
                    elif (group_1.shape[0] == 0) and (~(group_1.shape[0] > 1)):
                        # Check for group 2 only if group 1 does not meet the condition
                        group_2 = sorted_df[sorted_df['levenshtein_distance'] == 2]
                        group_2.reset_index(inplace=True)
                        if group_2.shape[0] == 1:
                            buyer_name = group_2.loc[0,"NAME"]
                            buyer_gstin = group_2.loc[0,"GSTIN"]
                            lev_distance[buyer_gstin] = [2,buyer_name]
                        else:
                            print("No groups meet the criteria")
                        
            if len(lev_distance) > 0:        
                filtered_dict_1 = {k: v for k, v in lev_distance.items() if v[0] == 1}
                filtered_dict_2 = {k: v for k, v in lev_distance.items() if v[0] == 2}
                print("filtered_dict_1", filtered_dict_1)
                if len(filtered_dict_1) == 1:
                    ## Only 1 gstin is present in masterdata with 1 lev distance
                    for key, value in filtered_dict_1.items():
                        gstin_matched = key
                        buyer_name_matched = value[1]
                        modified_confidence = 0.71
                        print("Final Data before returning", gstin_matched, buyer_name_matched, modified_confidence)
                        return gstin_matched,buyer_name_matched, modified_confidence
                if len(filtered_dict_1) > 1:
                    for key, value in filtered_dict_1.items():
                        gstin_matched = key
                        buyer_name_matched = value[1]
                        modified_confidence = 0.51
                        print("Final Data before returning", gstin_matched, buyer_name_matched, modified_confidence)
                        return gstin_matched,buyer_name_matched, modified_confidence
                    
                print("filtered_dict_2", filtered_dict_2)
                if len(filtered_dict_2) == 1:
                    ## Only 1 gstin is present in masterdata with 2 lev distance
                    for key, value in filtered_dict_2.items():
                        gstin_matched = key
                        buyer_name_matched = value[1]
                        modified_confidence = 0.71
                        print("Final Data before returning", gstin_matched, buyer_name_matched, modified_confidence)
                        return gstin_matched,buyer_name_matched, modified_confidence
                if len(filtered_dict_2) > 1:
                    ## more than 1 gstin is present in masterdata with 2 lev distance
                    for key, value in filtered_dict_2.items():
                        gstin_matched = key
                        buyer_name_matched = value[1]
                        modified_confidence = 0.51
                        print("Final Data before returning", gstin_matched, buyer_name_matched, modified_confidence)
                        return gstin_matched,buyer_name_matched, modified_confidence
                print("No data found in pred file and masterdata for GSTIN Issue")
                return "", "", 0
            else:
                print("No GSTIN found with lev distance <= 2")
                return "","",0
        else:
            print("Unable to find  vendor GSTIN")
            return "","",0
    except:
        print("extract_billing_gstin_pattern_after_ocr_issue Exception :",traceback.print_exc())
        return "","",0        

def extract_vendor_gstin_name_after_ocr_issue(DF:pd.DataFrame,prediction:dict,VENDOR_MASTERDATA, ADDRESS_MASTERDATA)->dict:
    pred_copy = copy.deepcopy(prediction)
    try:
        vendorGSTIN = prediction.get("vendorGSTIN")
        vendorName = prediction.get("vendorName")
        
        # if (vendorGSTIN != None) and ((vendorGSTIN.get("text") != None)) and (vendorGSTIN["text"] != '') and ((vendorGSTIN.get("final_confidence_score") != None)) and (vendorGSTIN.get("final_confidence_score")==1.0):
        if (vendorGSTIN != None) and (vendorGSTIN.get("final_confidence_score")==1.0):
            return prediction
        gstin_matched,vendor_name_matched, modified_confidence = extract_vendor_gstin_pattern_after_ocr_issue(DF,VENDOR_MASTERDATA, ADDRESS_MASTERDATA)
        if (vendorGSTIN == None) or (vendorGSTIN["text"] == '') or (len(vendorGSTIN["text"]) != 15) :
            if gstin_matched != '':
                print("Correcting vendor GSTIN")
                prediction.update(add_new_field(field_name = "vendorGSTIN",
                                                    value = str(gstin_matched).upper(),
                                                    final_confidence_score = modified_confidence,
                                                    vendor_masterdata_present = False
                                                    ))
            if (vendorName== None) or (vendorName == ""):    
                if vendor_name_matched != "":
                    print("Correcting vendor Name")
                    prediction.update(add_new_field(field_name = "vendorName",
                                                        value = vendor_name_matched,
                                                        final_confidence_score = modified_confidence,
                                                        vendor_masterdata_present = False
                                                        ))
        # else:
        #     print("GSTIN is 15 digits but OCR Issue in one of the characters")
        #     if prediction.get("vendorGSTIN").get("final_confidence_score") != 1:
        #         if gstin_matched != '':
        #             print("Changing vendor GSTIN")
        #             prediction["vendorGSTIN"]["text"]= gstin_matched
        #             prediction["vendorGSTIN"]["final_confidence_score"] = modified_confidence
        #             prediction["vendorGSTIN"]["vendor_masterdata_present"] = True
        #             prediction["vendorGSTIN"]["extracted_from_entitydata"] = True
        #         if vendor_name_matched != "":    
        #             prediction["vendorName"]["text"] = vendor_name_matched
        #             prediction["vendorName"]["final_confidence_score"] = modified_confidence
        #             prediction["vendorName"]["vendor_masterdata_present"] = True
        #             prediction["vendorName"]["extracted_from_entitydata"] = True
        if prediction.get("vendorGSTIN") == None:
            prediction.update(add_empty_field("vendorGSTIN","N/A"))
        if prediction.get("vendorName") == None:
            prediction.update(add_empty_field("vendorName",""))
        return prediction
    except:
        print("extract_vendor_gstin_name_after_ocr_issue Exception :",traceback.print_exc())
        return pred_copy

def extract_buyer_gstin_name_after_ocr_issue(DF:pd.DataFrame,prediction:dict,VENDOR_MASTERDATA, ADDRESS_MASTERDATA):
    pred_copy = copy.deepcopy(prediction)
    try:
        
        billingGSTIN = prediction.get("billingGSTIN")
        billingName = prediction.get("billingName")
        
        if (billingGSTIN != None) and (billingGSTIN.get("final_confidence_score")==1.0):
            return prediction
        
        gstin_matched,billing_name_matched, modified_confidence = extract_billing_gstin_pattern_after_ocr_issue(DF,VENDOR_MASTERDATA, ADDRESS_MASTERDATA)
        if (billingGSTIN == None) or (billingGSTIN["text"] == '') or (len(billingGSTIN["text"]) != 15) :
            if gstin_matched != '':
                print("Correcting billing GSTIN")
                prediction.update(add_new_field(field_name = "billingGSTIN",
                                                    value = str(gstin_matched).upper(),
                                                    final_confidence_score = modified_confidence,
                                                    vendor_masterdata_present = False
                                                    ))
        if (billingName== None) or (billingName == ""):    
            if billing_name_matched != "":
                print("Correcting billing Name")
                prediction.update(add_new_field(field_name = "billingName",
                                                    value = billing_name_matched,
                                                    final_confidence_score = modified_confidence,
                                                    vendor_masterdata_present = False
                                                    ))
        
        if prediction.get("billingGSTIN") == None:
            prediction.update(add_empty_field("billingGSTIN","N/A"))
        if prediction.get("billingName") == None:
            prediction.update(add_empty_field("billingName",""))
        
        return prediction
    except:
        print("extract_buyer_gstin_name_after_ocr_issue Exception :",traceback.print_exc())
        return pred_copy

def copy_shipping_gstin_after_ocr_issue(prediction):
    pred_copy = copy.deepcopy(prediction)
    try:
        billingGSTIN = prediction.get("billingGSTIN")
        shippingGSTIN = prediction.get("shippingGSTIN")
        shippingName = prediction.get("shippingName")
        billingName = prediction.get("billingName")

        if (shippingGSTIN != None )and (shippingGSTIN.get("final_confidence_score")==1.0):
            return prediction
        if (shippingGSTIN == None) and (billingGSTIN!= None) and (billingGSTIN.get("text")!= None) and (billingGSTIN.get("text") != ''):
            prediction.update(add_new_field(field_name = "shippingGSTIN",
                                                    value = billingGSTIN.get("text"),
                                                    from_entity = billingGSTIN.get("extracted_from_entitydata"),
                                                    final_confidence_score = billingGSTIN.get("final_confidence_score"),
                                                    vendor_masterdata_present = billingGSTIN.get("vendor_masterdata_present", False)))
        if (shippingName == None) and (billingName != None) and (billingName.get("text")!= None) and (billingName.get("text") != ''):
            prediction.update(add_new_field(field_name = "shippingName",
                                                    value = billingName.get("text"),
                                                    from_entity = billingName.get("extracted_from_entitydata"),
                                                    final_confidence_score = billingName.get("final_confidence_score"),
                                                    vendor_masterdata_present = billingName.get("vendor_masterdata_present", False)))
        return prediction
    except:
        print("copy_shipping_gstin_after_ocr_issue Exception :",traceback.print_exc())
        return pred_copy
def get_vendor_buyers_name(DF,prediction,ADDRESS_MASTERDATA,VENDOR_MASTERDATA):
    vendorGSTIN = None
    billingGSTIN = None
    shippingGSTIN = None
    vendorName = None
    billingName = None
    shippingName = None
    # clean_GSTIN = ['/',':','(',')','.',"'",","]        
    if prediction.get("vendorGSTIN"):
        vendorGSTIN = prediction.get("vendorGSTIN").get("text")
        vendorGSTIN = putil.correct_gstin(vendorGSTIN)
        # print("vendorGSTIN : ",vendorGSTIN)
    if prediction.get("billingGSTIN"):
        billingGSTIN = prediction.get("billingGSTIN").get("text")
        billingGSTIN = putil.correct_gstin(billingGSTIN) 
        # print("billingGSTIN : ",billingGSTIN)
    if prediction.get("shippingGSTIN"):
        shippingGSTIN = prediction.get("shippingGSTIN").get("text")
        shippingGSTIN = putil.correct_gstin(shippingGSTIN)
        # print("shippingGSTIN : ",shippingGSTIN)
    if prediction.get("vendorName"):
        vendorName = prediction.get("vendorName").get("text")
    if prediction.get("billingName"):
        billingName = prediction.get("billingName").get("text")
    if prediction.get("shippingName"):
        shippingName = prediction.get("shippingName").get("text")

    DF = DF[DF["is_company_name"]==1]
    C_Names = DF["line_text"].unique()
    C_Names = [x.upper() for x in C_Names]
    print("Company names :",C_Names)

    if billingName is None:
        print("billingName is None")
        if (billingGSTIN is not None):
            # Matching GSTIN with buyers address master data
            B_Name_frm_buyersData = ADDRESS_MASTERDATA[ADDRESS_MASTERDATA["GSTIN"].str.upper() == str(billingGSTIN).upper()]
            print("Match DF shape :",B_Name_frm_buyersData.shape[0])
            if B_Name_frm_buyersData.shape[0] > 0:
                for idx, row in B_Name_frm_buyersData.iterrows():
                    row["NAME"] = row["NAME"].upper()
                    print("Matched Name :", row["NAME"])
                    print("updating Matched Name Billing name from address master data",row["NAME"])
                    if (prediction.get("billingGSTIN").get("final_confidence_score") == 0.71) or (prediction.get("billingGSTIN").get("final_confidence_score") == 0.51):
                        prediction.update(add_new_field("billingName",row["NAME"],final_confidence_score = prediction.get("billingGSTIN").get("final_confidence_score"),from_Vendor=False,from_entity=False,vendor_masterdata_present = False))
                    else:
                        prediction.update(add_new_field("billingName",row["NAME"],from_Vendor=True,from_entity=True))
                    break
                    # 29 March 2023 Removed extra validation of checking with company_name
                    """
                    if B_Name_frm_buyersData.shape[0] > 1:
                        for item in C_Names:
                            # print("items :",item)
                            if item in row["NAME"]:
                                print("updating Matched Name Billing name from address master data",row["NAME"])
                                prediction.update(add_new_field("billingName",row["NAME"],from_Vendor=True,from_entity=True))
                            else:
                                print("Match not foud in buyes data")
                    else:
                        prediction.update(add_new_field("billingName",row["NAME"],from_Vendor=True,from_entity=True))  """                     
            else:
                print("billing GSTIN Match not found")
        else:
            print("billing GSTIN is None :",billingGSTIN)
    else:
        print("billingName is Not None")
        if (billingGSTIN is not None):
            print("billing GSTIN Not None")
            B_Name_frm_buyersData = ADDRESS_MASTERDATA[ADDRESS_MASTERDATA["GSTIN"].str.upper() == str(billingGSTIN).upper()]
            print("Match DF shape :",B_Name_frm_buyersData.shape[0])
            if B_Name_frm_buyersData.shape[0] > 0:
                for idx, row in B_Name_frm_buyersData.iterrows():
                    row["NAME"] = str(row["NAME"]).upper()
                    print("GSTIN Matched Names :",row["NAME"])

                    print("updating Matched Name :",row["NAME"])
                    if (prediction.get("billingGSTIN").get("final_confidence_score") == 0.71) or (prediction.get("billingGSTIN").get("final_confidence_score") == 0.51):
                        prediction.update(add_new_field("billingName",row["NAME"], final_confidence_score = prediction.get("billingGSTIN").get("final_confidence_score"), from_Vendor=False,from_entity=False, vendor_masterdata_present = False))
                    else:
                        prediction.update(add_new_field("billingName",row["NAME"],from_Vendor=True,from_entity=True))
                    break
                    # 16 May 2023 Removed extra validation of checking with company_name
                    if B_Name_frm_buyersData.shape[0] > 1:
                        print("Partial billing Name Not matche so matching with the all the invoice company name")
                        for item in C_Names:
                            print("items :",item)
                            if item in row["NAME"]:
                                print("updating Matched Name",row["NAME"])
                                prediction.update(add_new_field("billingName",row["NAME"],from_Vendor=True,from_entity=True))
                            else:
                                print("Match not foud in vendor data")
                    else:
                        if str(billingName).upper() in row["NAME"]:
                            print("updating partial matched Name",row["NAME"])
                            prediction.update(add_new_field("billingName",row["NAME"],from_Vendor=True,from_entity=True))
                        else:
                            print("Updating Billing Name by matching GSTIN ",row["NAME"])
                            prediction.update(add_new_field("billingName",row["NAME"],from_Vendor=True,from_entity=True))
            else:
                print("No Match data found")
        else:
            print("billing GSTIN is None :",billingGSTIN)           
    
    if shippingName is None:
        print("shippingName is None")
        if (shippingGSTIN is not None):
            # Matching GSTIN with buyers address master data
            B_Name_frm_buyersData = ADDRESS_MASTERDATA[ADDRESS_MASTERDATA["GSTIN"].str.upper() == str(shippingGSTIN).upper()]
            print("Match DF shape :",B_Name_frm_buyersData.shape[0])
            if B_Name_frm_buyersData.shape[0] > 0:
                for idx, row in B_Name_frm_buyersData.iterrows():
                    row["NAME"] = row["NAME"].upper()
                    print("Matched Name :", row["NAME"])
                    print("updating Matched Name :",row["NAME"])
                    if (prediction.get("shippingGSTIN").get("final_confidence_score") == 0.71) or (prediction.get("shippingGSTIN").get("final_confidence_score") == 0.51):
                        prediction.update(add_new_field("shippingName",row["NAME"], final_confidence_score = prediction.get("shippingGSTIN").get("final_confidence_score"),from_Vendor=False,from_entity=False, vendor_masterdata_present = False))
                    else:
                        prediction.update(add_new_field("shippingName",row["NAME"],from_Vendor=True,from_entity=True))
                    break
                    # 29 March 2023 Removed extra validation of checking with company_name
                    """if B_Name_frm_buyersData.shape[0] > 1:
                        for item in C_Names:
                            # print("items :",item)
                            if item in row["NAME"]:
                                print("updating Matched Name :",row["NAME"])
                                prediction.update(add_new_field("shippingName",row["NAME"],from_Vendor=True,from_entity=True))
                            else:
                                print("Match not foud in buyes data")
                    else:
                        print("Updating shipping Name with extract GSTIN match :",row["NAME"])
                        prediction.update(add_new_field("shippingName",row["NAME"],from_Vendor=True,from_entity=True))"""
            else:
                print("No Match found")
        else:
            print("Shipping GSTIN is None :",shippingGSTIN)
    else:
        print("shippingName is Not None")
        if (shippingGSTIN is not None):
            print("shipping GSTIN Not None")
            S_Name_frm_buyersData = ADDRESS_MASTERDATA[ADDRESS_MASTERDATA["GSTIN"].str.upper() == str(shippingGSTIN).upper()]
            print("Match DF shape :",S_Name_frm_buyersData.shape[0])
            if S_Name_frm_buyersData.shape[0] > 0:
                for idx, row in S_Name_frm_buyersData.iterrows():
                    row["NAME"] = str(row["NAME"]).upper()
                    print("GSTIN Matched Names :",row["NAME"])
                    print("updating Matched Name :",row["NAME"])
                    if (prediction.get("shippingGSTIN").get("final_confidence_score") == 0.71) or (prediction.get("shippingGSTIN").get("final_confidence_score") == 0.51):
                        prediction.update(add_new_field("shippingName",row["NAME"], final_confidence_score = prediction.get("shippingGSTIN").get("final_confidence_score"),from_Vendor=False,from_entity=False, vendor_masterdata_present = False))
                    else:
                        prediction.update(add_new_field("shippingName",row["NAME"],from_Vendor=True,from_entity=True))
                    break
                    # 12 June 2023, Removed extra validation of checking with company_name
                    if S_Name_frm_buyersData.shape[0] > 1:
                        print("Partial vendor Name Not matche so matching with the all the invoice company name")
                        for item in C_Names:
                            print("items :",item)
                            if item in row["NAME"]:
                                print("updating Matched Name",row["NAME"])
                                prediction.update(add_new_field("shippingName",row["NAME"],from_Vendor=True,from_entity=True))
                            else:
                                print("Match not foud in vendor data")
                    else:
                        if str(shippingName).upper() in row["NAME"]:
                            print("updating partial matched Name",row["NAME"])
                            prediction.update(add_new_field("shippingName",row["NAME"],from_Vendor=True,from_entity=True))
                        else:
                            print("updating shipping Name by exact GSTIN match",row["NAME"])
                            prediction.update(add_new_field("shippingName",row["NAME"],from_Vendor=True,from_entity=True))                            
            else:
                print("No Match data found")
        else:
            print("shipping GSTIN is None :",shippingGSTIN)           

    if vendorName is None:
        print("vendorName is None")
        if (vendorGSTIN is not None):
            # Matching GSTIN with buyers address master data
            V_Name_frm_buyersData = VENDOR_MASTERDATA[VENDOR_MASTERDATA["VENDOR_GSTIN"].str.upper() == str(vendorGSTIN).upper()]
            print("Match DF shape :",V_Name_frm_buyersData.shape[0])
            if V_Name_frm_buyersData.shape[0] > 0:
                for idx, row in V_Name_frm_buyersData.iterrows():
                    row["VENDOR_NAME"] = str(row["VENDOR_NAME"]).upper()
                    print("GSTIN Matched Names :",row["VENDOR_NAME"])
                    if (prediction.get("vendorGSTIN").get("final_confidence_score") == 0.71) or (prediction.get("vendorGSTIN").get("final_confidence_score") == 0.51):
                        prediction.update(add_new_field("vendorName",row["VENDOR_NAME"],final_confidence_score = prediction.get("vendorGSTIN").get("final_confidence_score"),from_Vendor=False,from_entity = False, vendor_masterdata_present = False))
                    else:
                        prediction.update(add_new_field("vendorName",row["VENDOR_NAME"],from_Vendor=True,from_entity = True))
                    break
                    # 12 June 2023, Removed extra validation of checking with company_name
                    if V_Name_frm_buyersData.shape[0] > 1:
                        for item in C_Names:
                            # print("items :",item)
                            if item in row["VENDOR_NAME"]:
                                print("updating Matched Name",row["VENDOR_NAME"])
                                prediction.update(add_new_field("vendorName",row["VENDOR_NAME"],from_Vendor=True,from_entity = True))
                            else:
                                print("Match not foud in vendor data")
                    else:
                        print("Updating vendor Name with extract GSTIN match :",row["VENDOR_NAME"])
                        prediction.update(add_new_field("vendorName",row["VENDOR_NAME"],from_Vendor=True,from_entity=True))                        
            else:
                print("No Match data found")
        else:
            print("vendorName GSTIN is None :",vendorGSTIN)           
    else:
        print("VendorName is Not None")
        if (vendorGSTIN is not None):
            print("Vendor GSTIN Not None")
            V_Name_frm_buyersData = VENDOR_MASTERDATA[VENDOR_MASTERDATA["VENDOR_GSTIN"].str.upper() == str(vendorGSTIN).upper()]
            print("Match DF shape :",V_Name_frm_buyersData.shape[0])
            if V_Name_frm_buyersData.shape[0] > 0:
                for idx, row in V_Name_frm_buyersData.iterrows():
                    row["VENDOR_NAME"] = str(row["VENDOR_NAME"]).upper()
                    print("GSTIN Matched Names :",row["VENDOR_NAME"])
                    if (prediction.get("vendorGSTIN").get("final_confidence_score") == 0.71) or (prediction.get("vendorGSTIN").get("final_confidence_score") == 0.51):
                        prediction.update(add_new_field("vendorName",row["VENDOR_NAME"], final_confidence_score = prediction.get("vendorGSTIN").get("final_confidence_score"),from_Vendor=False,from_entity = False, vendor_masterdata_present =False ))
                    else:
                        prediction.update(add_new_field("vendorName",row["VENDOR_NAME"],from_Vendor=True,from_entity = True))
                    break
                    # 12 June 2023, Removed extra validation of checking with company_name
                    if V_Name_frm_buyersData.shape[0] > 1:
                        print("Partial vendor Name Not matche so matching with the all the invoice company name")
                        for item in C_Names:
                            print("items :",item)
                            if item in row["VENDOR_NAME"]:
                                print("updating Matched Name",row["VENDOR_NAME"])
                                prediction.update(add_new_field("vendorName",row["VENDOR_NAME"],from_Vendor=True,from_entity=True))
                            else:
                                print("Match not foud in vendor data")
                    else:
                        if str(vendorName).upper() in row["VENDOR_NAME"]:
                            print("updating partial matched Name",row["VENDOR_NAME"])
                            prediction.update(add_new_field("vendorName",row["VENDOR_NAME"],from_Vendor=True,from_entity=True))
                        else:
                            print("updating vendor Name by exact GSTIN match",row["VENDOR_NAME"])
                            prediction.update(add_new_field("vendorName",row["VENDOR_NAME"],from_Vendor=True,from_entity=True))                            
            else:
                print("No Match data found")
        else:
            print("vendorName GSTIN is None :",vendorGSTIN)           

    return prediction

# Removing 
def clean_GSTIN(prediction):
    for key, val in prediction.items():
        if key in ["vendorGSTIN","billingGSTIN","shippingGSTIN"]:
            if val:
                if val["text"]!='':
                    print("GSTINS befor cleaning :",val["text"])
                    val["text"] = putil.correct_gstin(val["text"])
                    prediction[key] = val
                    print("GSTIN clenned",val["text"])
    return prediction

# extracting VendorPAN from vendor GSTIN
def extract_vendorPAN(prediction):
    try:
        vendor_gstin = prediction.get("vendorGSTIN")
        if vendor_gstin:
            gstin = vendor_gstin["text"]
            frm_vendor =  vendor_gstin.get("extracted_from_entitydata",False)
            fnl_cnf = vendor_gstin["final_confidence_score"]
            print("frm_vendor :",frm_vendor,vendor_gstin.get("extracted_from_entitydata",False))
            if gstin != '':
                if len(gstin) == 15:
                    print("getting PAN from GSTIN :",gstin)
                    pattern = r'[A-Z]{5}\d{4}[A-Z]{1}' 
                    pan_pattern = re.findall(pattern, gstin, flags = 0)
                    if len(pan_pattern) > 0:           
                        vendorPAN = pan_pattern[0] 
                    else :
                        print("pattern written based on index range")
                        frm_vendor = False
                        vendorPAN = gstin[2:12]
                        print("frm_vendor :",frm_vendor)
                    print("vendorPAN",vendorPAN)
                    VendorPAN = add_new_field("vendorPAN",vendorPAN,from_Vendor=frm_vendor,final_confidence_score=fnl_cnf)
                    print("VendorPAN :",VendorPAN)
                    prediction.update(VendorPAN)
                    print("vendorPAN",prediction.get("vendorPAN"))
            else: 
                print("Empty GSTIN field Reccived")
        return prediction
    except Exception as e:
        print("extract_vendorPAN exception :",e)
        return prediction
# adding Mandatory fields flag from stg config into field result
@putil.timing
def adding_mandatory_fieldFlag(prediction,doc_type = "DEFAULT"):
    if doc_type.lower() == "invoice":
        doc_type = "DEFAULT"
    pred_copy = copy.deepcopy(prediction)
    try:
        STP_CONFIGURATION = putil.getSTPConfiguration()
        STP_CONFIGURATION = STP_CONFIGURATION.get(doc_type)
        display_fields = [key for key,val in STP_CONFIGURATION.items() if (val["display_flag"] == 1)]
        #print("rcvd pred :",prediction)
        for key, val in pred_copy.items():
            if (key in display_fields) and (val is not None):   
                # print("key :",key, "\tval :",val)
                mandatory_field = STP_CONFIGURATION.get(key).get("mandatory")
                val.update({"mandatory":mandatory_field})
                prediction[key] = val
        #print("prediction after adding mdtr :",prediction)
        return prediction
    except:
        print("adding_mandatory_fieldFlag exception :",adding_mandatory_fieldFlag)
        return pred_copy

## set prediction  into custom order
def get_sequence_list():
    stg_config = putil.getSTPConfiguration()   
    stg_config = stg_config.get("DEFAULT")
    # print("config data :",stg_config)
    sequence_list = {}
    for k ,v in stg_config.items():
        if (v.get("display_flag")==1) and (v.get("order")):
            sequence_list[v["order"]] = k
    sequence_list = sorted(sequence_list.items())
    sequence_list = [x[1] for x in sequence_list]

    return sequence_list

def custom_sorting_prediction(prediction):
    sequence_list = get_sequence_list()
    # print("prediction oredered keys :",sequence_list)
    not_in_sequence = []
    sorted_prediction = {}
    try:
        pred_keys = list(prediction.keys())
        for i in pred_keys:
            if i not in sequence_list:
                not_in_sequence.append(i)
        # print("keys not in sequence list :",not_in_sequence)
        for item in sequence_list:
            val = prediction.get(item)
            sorted_prediction.update({item:val})
        for item_ in not_in_sequence:
            val_ = prediction.get(item_)
            sorted_prediction.update({item_:val_})
            
        return sorted_prediction
    except:
        print("Sorting prediction  exception :",traceback.print_exc())
        return prediction

def update_field_values(prediction:dict,field_name:str,final_confidence_score=None,calculated=None):
    try:
        field_json = prediction.get(field_name)
        if field_json:
            if final_confidence_score is not None:
                field_json["final_confidence_score"] =final_confidence_score
            if calculated is not None:
                field_json["calculated"]= calculated
            prediction[field_name] = field_json
        print("update field values :",field_json)
        return prediction
    except:
        print("update field exception:",traceback.print_exc())
        return prediction

def calculate_total_old(DF, prediction)-> dict:
    
    import math
    pred_copy = copy.deepcopy(prediction) 
    try:
        fields = ["totalAmount","subTotal","CGSTAmount","SGSTAmount","IGSTAmount",
                        "CessAmount","additionalCessAmount","discountAmount"]
        field_values = {}
        for f in fields:
            if (prediction.get(f)) and (prediction.get(f).get("text") != ''):
                field_values.update({f:float(prediction.get(f).get("text"))})
            else:
                field_values.update({f : None}) 

        '''    
        -> calculating copying subtotal as total if all taxes is None.
        -> subtracting discount if it is not None
        '''
        CGSTAmount = field_values.get("CGSTAmount")
        SGSTAmount = field_values.get("SGSTAmount")
        IGSTAmount = field_values.get("IGSTAmount")
        discountAmount = field_values.get("discountAmount")
        additionalCessAmount = field_values.get("additionalCessAmount")
        CessAmount = field_values.get("CessAmount")
        subTotal = field_values.get("subTotal")
        total = field_values.get("totalAmount")

        # for key, val in field_values.items():
        #     print(key," : ",val)
        noCgst = (CGSTAmount is None) or (CGSTAmount == 0.0)
        noSgst = (SGSTAmount is None) or (SGSTAmount == 0.0)
        noIgst = (IGSTAmount is None) or (IGSTAmount == 0.0)
        noDiscount = (discountAmount is None) or (discountAmount == 0.0)
        noAdCess = (additionalCessAmount is None) or (additionalCessAmount == 0.0)
        noCess = (CessAmount is None) or (CessAmount == 0.0)
        noTotal = (total is None) or (total == 0.0)
        noSubTotal = (subTotal is None) or (subTotal == 0.0)

        if noCgst and noSgst and noIgst and noDiscount and noAdCess and noCess:
            #Aug 02 2022 - code to allow STP if total = subtotal and no taxes are found in header amounts
            df_filt = DF[(DF["line_row"] == 0) & (DF["extracted_amount"] > 0.0)]
            extracted_amounts = list(set(list(df_filt["extracted_amount"])))
            stp_check = False
            if len(extracted_amounts) <= 2:
                m1 = max(extracted_amounts)
                m2 = min(extracted_amounts)
                if m1 - m2 < 1:
                    stp_check = True
            elif len(extracted_amounts) > 2:
                extracted_amounts = sorted(extracted_amounts,
                                           reverse = True)
                first = extracted_amounts[0]
                second = extracted_amounts[1]
                if first - second < 1:
                    rem_amounts = extracted_amounts[2:]
                else:
                    rem_amounts = extracted_amounts[1:]
                sum_amounts = sum(rem_amounts)
                if abs(sum_amounts - first) < 1:
                    stp_check = True
            #Aug 02 2022 - code to allow STP if total = subtotal and no taxes are found in header amounts

            if noTotal:
                print("inside all taxes none")
                if subTotal:
                    if subTotal > 0.0:
                        print("sub total is > 0")
                        if stp_check:
                            prediction.update(add_new_field(
                                field_name = "totalAmount",
                                value = subTotal,
                                final_confidence_score = 1.0,
                                calculated = not stp_check))

                            #Make other amount fields as 100%
                            prediction.update(add_new_field(
                                field_name = "subTotal",
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "CGSTAmount",
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "SGSTAmount",
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "IGSTAmount",
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "CessAmount",
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "additionalCessAmount",
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            #Make other amount fields as 100%

                        else:
                            prediction.update(add_new_field(
                                field_name = "totalAmount",
                                value = subTotal,
                                final_confidence_score = 0.4,
                                calculated = not stp_check))

            elif noSubTotal:
                if total:
                    if total > 0.0:
                        print("total is > 0")
                        if stp_check:
                            prediction.update(add_new_field(
                                field_name = "subTotal",
                                value = total,
                                final_confidence_score = 1.0,
                                calculated = not stp_check))

                            #Make other amount fields as 100%
                            prediction.update(add_new_field(
                                field_name = "totalAmount",
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "CGSTAmount",
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "SGSTAmount",
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "IGSTAmount",
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "CessAmount",
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "additionalCessAmount",
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            #Make other amount fields as 100%

                        else:
                            prediction.update(add_new_field(
                                field_name = "subTotal",
                                value = total,
                                final_confidence_score = 0.4,
                                calculated = not stp_check))
                            
            elif subTotal > 0.0 and total > 0.0 and math.isclose(total,
                                                                 subTotal,
                                                                 abs_tol = 1):
                prediction.update(add_new_field(
                    field_name = "totalAmount",
                    value = subTotal,
                    final_confidence_score = 0.4,
                    calculated = not stp_check))
                

        return prediction

    except :
        print("Calculate total exception \n",
              traceback.print_exc())
        return pred_copy

def lefting_stp_throtle_on_selected_vendors(prediction):
    try:
        path = cfg.getLEFTING_STP_THROTLE_VENDORS()
        print("path :",path)
        lefting_throtle_for_vendors = pd.read_csv(path) #("/Users/Parmesh/Downloads/stp_throtle_reference_data.csv")
        print("lefting_throtle_for_vendors shape",lefting_throtle_for_vendors.shape)
        if prediction.get("vendorGSTIN"):
            vendor_gstin = prediction.get("vendorGSTIN").get("text")    
            print("vendor_gstin :",vendor_gstin)
            print("lefting_throtle_for_vendors unique :",lefting_throtle_for_vendors["vendor_gstin"].unique())
            if vendor_gstin in lefting_throtle_for_vendors["vendor_gstin"].unique():
                print("stp_status true")
                return True
        return False 
    except:
        print("lefting_stp_throtle_on_selected_vendors exeption",traceback.print_exc())
        return False


def calculate_total(DF, prediction)-> dict:
    import math
    pred_copy = copy.deepcopy(prediction)
    try:
        fields = ["totalAmount","subTotal","CGSTAmount","SGSTAmount","IGSTAmount",
                        "CessAmount","additionalCessAmount","discountAmount"]
        field_values = {}
        for f in fields:
            if (prediction.get(f)) and (prediction.get(f).get("text") != ''):
                field_values.update({f:float(prediction.get(f).get("text"))})
            else:
                field_values.update({f : None})
        '''
        -> calculating copying subtotal as total if all taxes is None.
        -> subtracting discount if it is not None
        '''
        CGSTAmount = field_values.get("CGSTAmount")
        SGSTAmount = field_values.get("SGSTAmount")
        IGSTAmount = field_values.get("IGSTAmount")
        discountAmount = field_values.get("discountAmount")
        additionalCessAmount = field_values.get("additionalCessAmount")
        CessAmount = field_values.get("CessAmount")
        subTotal = field_values.get("subTotal")
        total = field_values.get("totalAmount")
        # for key, val in field_values.items():
        #     print(key," : ",val)
        noCgst = (CGSTAmount is None) or (CGSTAmount == 0.0)
        noSgst = (SGSTAmount is None) or (SGSTAmount == 0.0)
        noIgst = (IGSTAmount is None) or (IGSTAmount == 0.0)
        noDiscount = (discountAmount is None) or (discountAmount == 0.0)
        noAdCess = (additionalCessAmount is None) or (additionalCessAmount == 0.0)
        noCess = (CessAmount is None) or (CessAmount == 0.0)
        noTotal = (total is None) or (total == 0.0)
        noSubTotal = (subTotal is None) or (subTotal == 0.0)
        if noCgst and noSgst and noIgst and noDiscount and noAdCess and noCess:
            #Aug 02 2022 - code to allow STP if total = subtotal and no taxes are found in header amounts
            df_filt = DF[(DF["line_row"] == 0) & (DF["extracted_amount"] > 0.0)]
            extracted_amounts = list(set(list(df_filt["extracted_amount"])))
            stp_check = False
            if len(extracted_amounts) <= 2:
                m1 = max(extracted_amounts)
                m2 = min(extracted_amounts)
                if m1 - m2 < 1:
                    stp_check = True
            elif len(extracted_amounts) > 2:
                extracted_amounts = sorted(extracted_amounts,
                                           reverse = True)
                first = extracted_amounts[0]
                second = extracted_amounts[1]
                if first - second < 1:
                    rem_amounts = extracted_amounts[2:]
                else:
                    rem_amounts = extracted_amounts[1:]
                sum_amounts = sum(rem_amounts)
                if abs(sum_amounts - first) < 1:
                    stp_check = True
            #Aug 02 2022 - code to allow STP if total = subtotal and no taxes are found in header amounts
            '''
            # Nov 03 2022 - code to alloww STP for selected venodors only if sub and total are equal
            #  to use this changes need to uncomment changes within comment "Remove the STP throttle for subTotal = Total - Sep 20, 2022"
            print("stp_check b4",stp_check)
            stp_check = lefting_stp_throtle_on_selected_vendors(prediction)
            print("stp_check after :",stp_check)
            # Nov 03 2022 - code to alloww STP for selected venodors only if sub and total are equal
            '''    
            if noTotal:
                print("inside all taxes none")
                if subTotal:
                    if subTotal > 0.0:
                        print("sub total is > 0")
                        if stp_check:
                            prediction.update(add_new_field(
                                field_name = "totalAmount",
                                value = subTotal,
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            # #Make other amount fields as 100%
                            prediction.update(add_new_field(
                                field_name = "subTotal",
                                value = subTotal,
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "CGSTAmount",
                                value = 0,
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "SGSTAmount",
                                value = 0,
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "IGSTAmount",
                                value = 0,
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "CessAmount",
                                value = 0,
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "additionalCessAmount",
                                value = 0,
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            
                            #Make other amount fields as 100%
                        else:
                            #Remove the STP throttle for subTotal = Total - Sep 20, 2022
                            # prediction.update(add_new_field(
                            #     field_name = "totalAmount",
                            #     value = subTotal,
                            #     final_confidence_score = 0.7,
                            #     calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "totalAmount",
                                value = subTotal,
                                final_confidence_score = 0.7,
                                calculated = False))
                            #Remove the STP throttle for subTotal = Total - Sep 20, 2022
            elif noSubTotal:
                if total:
                    if total > 0.0:
                        print("total is > 0")
                        if stp_check:
                            prediction.update(add_new_field(
                                field_name = "subTotal",
                                value = total,
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            #Make other amount fields as 100%
                            prediction.update(add_new_field(
                                field_name = "totalAmount",
                                value = total,
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "CGSTAmount",
                                value = 0,
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "SGSTAmount",
                                value = 0,
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "IGSTAmount",
                                value = 0,
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "CessAmount",
                                value = 0,
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "additionalCessAmount",
                                value = 0,
                                final_confidence_score = 1.0,
                                calculated = not stp_check))

                            #Make other amount fields as 100%
                        else:
                            #Remove the STP throttle for subTotal = Total - Sep 20, 2022
                            # prediction.update(add_new_field(
                            #     field_name = "subTotal",
                            #     value = total,
                            #     final_confidence_score = 0.7,
                            #     calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "subTotal",
                                value = total,
                                final_confidence_score = 0.7,
                                calculated = False))
                            #Remove the STP throttle for subTotal = Total - Sep 20, 2022
            elif subTotal > 0.0 and total > 0.0 and math.isclose(total,
                                                                 subTotal,
                                                                 abs_tol = 1):
                #Remove the STP throttle for subTotal = Total - Sep 20, 2022
                # prediction.update(add_new_field(
                #     field_name = "totalAmount",
                #     value = subTotal,
                #     final_confidence_score = 0.4,
                #     calculated = not stp_check))
                if stp_check:
                    prediction.update(add_new_field(
                        field_name = "totalAmount",
                        value = subTotal,
                        final_confidence_score = 1.0,
                        calculated = not stp_check))
                    #Make other amount fields as 100%
                    prediction.update(add_new_field(
                        field_name = "CGSTAmount",
                        value = 0,
                        final_confidence_score = 1.0,
                        calculated = not stp_check))
                    prediction.update(add_new_field(
                        field_name = "SGSTAmount",
                        value = 0,
                        final_confidence_score = 1.0,
                        calculated = not stp_check))
                    prediction.update(add_new_field(
                        field_name = "IGSTAmount",
                        value = 0,
                        final_confidence_score = 1.0,
                        calculated = not stp_check))
                    prediction.update(add_new_field(
                        field_name = "CessAmount",
                        value = 0,
                        final_confidence_score = 1.0,
                        calculated = not stp_check))
                    prediction.update(add_new_field(
                        field_name = "additionalCessAmount",
                        value = 0,
                        final_confidence_score = 1.0,
                        calculated = not stp_check))

                    #Make other amount fields as 100%
                else:
                    print("None stp")
                    prediction.update(add_new_field(
                        field_name = "totalAmount",
                        value = subTotal,
                        final_confidence_score = 0.7,
                        calculated = False))
                        # calculated = not stp_check))
                #Remove the STP throttle for subTotal = Total - Sep 20, 2022
        return prediction
    except :
        print("Calculate total exception \n",
              traceback.print_exc())
        return pred_copy

def calculate_total2(DF, prediction)-> dict:

    pred_copy = copy.deepcopy(prediction) 
    try:
        '''
        -> calculating total by adding taxes and subtracting discount
        -> validating calculated total with first and sencond max amount 
        '''

        fields = ["totalAmount","subTotal","CGSTAmount","SGSTAmount","IGSTAmount",
                  "CessAmount","additionalCessAmount","discountAmount"]
        field_values = {}
        for f in fields:
            if (prediction.get(f)) and (prediction.get(f).get("text") != ''):
                try:
                    field_values.update({f:float(prediction.get(f).get("text"))})
                except :
                    field_values.update({f : 0})
            else:
                field_values.update({f : 0}) 

        discountAmount = field_values.get("discountAmount")
        total = field_values.get("totalAmount")

        for key, val in field_values.items():
            print(key," : ",val)

        total = field_values.get("totalAmount")
        fields.remove("discountAmount")
        print("Addition fields :",fields) 
        add_fields_sum = 0
        for key, val in field_values.items():
            print("prediction field ",key," : ",val)
            if key != "totalAmount":
                add_fields_sum =  add_fields_sum + val
        print("add_fields_sum :",add_fields_sum, "DiscountAmount :",discountAmount)
        if (discountAmount >0):       
            total_cal = add_fields_sum - discountAmount
        else:
            total_cal = add_fields_sum
        print("Total calculated amount :",total_cal, "\tTotalAmount :",total)
        if abs(total- total_cal) >2:
            first_max_amt = DF[DF["is_amount"]==1]
            first_max_amt = first_max_amt[first_max_amt["line_row"]==0]
            amount_list = first_max_amt['text']
            final_amount_list = []
            for i in amount_list:
                try:
                    i = float(str(i).replace(',','').replace(':','').replace('/','').replace(u'\u20B9',''))
                    final_amount_list.append(i)
                except:
                    print("flaot conversion error :",i)
                    pass
            final_amount_list = sorted(set(final_amount_list),reverse= True)
            if len(final_amount_list)>0:
                print("final_amount_list ;",final_amount_list,"\total_cal",total_cal)
                counter = 0
                for amount in final_amount_list:
                    print("abs with first max amount :",abs(amount - total_cal))
                    if abs(amount - total_cal) < 2:
                        prediction.update(add_new_field(field_name = "totalAmount",
                                                        final_confidence_score =1,
                                                        value = amount,calculated=False))
                        field_values.update({"totalGSTAmount":0})
                        for key, val in field_values.items():
                            if (val == 0):
                                prediction = reduce_field_confidence(prediction, key,model_confidence = 1,final_confidence_score = 1)
                        break
                    if counter == 0:
                        break
                    counter = counter +1
        else:
            print("Absulate difference is grater than 2 :",abs(total- total_cal))             
        return prediction
    except :
        print("Calculate total exception \n",
              traceback.print_exc())
        return pred_copy

def check_if_future_Date(prediction:dict) -> dict:
    '''
    check if future date if it is then reduce confidence to 40 %
    '''
    pred_copy = copy.deepcopy(prediction)
    try:
        from datetime import datetime
        invdt = prediction.get("invoiceDate")
        if (invdt) and (invdt.get("text")!=''):
            formate_date = parser.parse(invdt.get("text"), dayfirst=True).date().strftime('%d/%m/%Y')
            print("orignal formate ;",invdt.get("text"),"\tFromate date",formate_date)
            if datetime.strptime(formate_date,'%d/%m/%Y') > datetime.now():
                print("future date reducing confidence")
                prediction = reduce_field_confidence(prediction,"invoiceDate")
                return prediction
        return pred_copy
    except:
        print("check future date exception ;",traceback.print_exc())
        return pred_copy

def check_left_above_invdt_ngbr(df,prediction):
    prediction_copy = prediction
    try:
        df1=df[(df.invdt_prediction == True)]
        df = df1[df1['page_num']==0]
        if df1.shape[0]==0:
            df = df1[df1['page_num']==1]
        inv_dt = prediction.get("invoiceDate")
        print("filter df :",df.shape)
        if inv_dt:
            print("invoiceDate values :",inv_dt)
            for row in df1.itertuples():
                row_date = parser.parse(row.text, dayfirst=True).date().strftime('%d/%m/%Y')
                print("converted date :",row_date, "bf4 con dt ;",row.text)
                if row_date == inv_dt.get('text'):
                    print("left ;",row.left,"\tright :",row.right,"\ttop ;",row.top,"\tbottom :",row.bottom)
                    print("above ngbr:",row.above_processed_ngbr,"left ngbr :",row.left_processed_ngbr)
                    check1=("invoice" in row.above_processed_ngbr.lower() or "inv" in row.above_processed_ngbr.lower())
                    check2=("invoice" in row.left_processed_ngbr.lower() or "inv" in row.left_processed_ngbr.lower())
                    print("check1 :",check1,"check2:",check2)
                    if not (check1) and not(check2):
                        print("invoice name not present in invoiceDate label. so reducing confidence")
                        prediction = reduce_field_confidence(prediction,"invoiceDate")
                        return prediction
                # else:
                #     print("else block")
                #     print("inv date :",row.text,"inve predicted dt",inv_dt.get('text'))
                #     # print("left ;",row.left,"\tright :",row.right,"\ttop ;",row.top,"\tbottom :",row.bottom)
                #     print("above ngbr:",row.above_processed_ngbr,"left ngbr :",row.left_processed_ngbr)

        return prediction
    except:
        print("check_left_above_invdt_ngbr :",traceback.print_exc())
        return prediction_copy


def copy_gstin(DF, prediction,gstin_bill = None, gstin_ship = None):
    pred_copy = copy.deepcopy(prediction)
    try:
        print("actual df shape :",DF.shape)
        F_DF = DF[DF["is_gstin_format"]==1]
        DF = F_DF[F_DF["page_num"] == 0]
        print("First page df shape :",DF.shape)
        if DF.shape[0] ==0 or DF.shape[0] is None:
            DF = F_DF[F_DF["page_num"] == 1]
            print("Second page df shape :",DF.shape)
        print("page df shape :",DF.shape)
        print("actual df shape :",DF.shape)
        #DF = DF[DF["is_gstin_format"]==1]
        #print("page df shape :",DF.shape)
        unique_gstin = list(set([putil.correct_gstin(s) for s in list(DF[DF["is_gstin_format"]==1]["text"].unique())]))
        print("total unique GSTIN : ", len(unique_gstin),"\t:",unique_gstin)
        
        # 5 April 2023 Added only for Pierian 
        # If only one GSTIN is present which belongs to billing GSTIN then copy to shipping GSTIN
        if len(unique_gstin) == 1 and prediction.get("billingGSTIN") != None:
            if prediction.get("billingGSTIN").get("text")!= None and prediction.get("billingGSTIN").get("text") != '':
                prediction.update(add_new_field(field_name = "shippingGSTIN",
                                                    value = prediction.get("billingGSTIN").get("text"),
                                                    from_entity = prediction.get("billingGSTIN").get("extracted_from_entitydata")))
        if len(unique_gstin) == 2:
            billingGSTIN = prediction.get("billingGSTIN")
            shippingGSTIN = prediction.get("shippingGSTIN")
            if (shippingGSTIN is not None) and (billingGSTIN is None):
                if shippingGSTIN.get("text") != '':
                    prediction.update(add_new_field(field_name = "billingGSTIN",
                                        value = shippingGSTIN.get("text"),
                                        from_entity = shippingGSTIN.get("extracted_from_entitydata")))
                    print("Copied  Shipping GSTIN to Billing GSTIN")
                    return prediction
            elif (billingGSTIN is not None) and (shippingGSTIN is None):
                if billingGSTIN.get("text") != '':
                    prediction.update(add_new_field(field_name = "shippingGSTIN",
                                                    value = billingGSTIN.get("text"),
                                                    from_entity = billingGSTIN.get("extracted_from_entitydata")))
                    print("Copied Billing GSTIN to Shipping GSTIN")
                    return prediction
            else :
                if ((gstin_bill is not None) and (gstin_ship is None)
                    or (gstin_bill is None) and (gstin_ship is not None)):
                    if (gstin_bill) and  (billingGSTIN.get("text") != ''):
                        if (shippingGSTIN) and (shippingGSTIN.get("text") != ''):
                            if shippingGSTIN.get("text") == billingGSTIN.get("text"):
                                prediction["shippingGSTIN"]["extracted_from_entitydata"] = billingGSTIN.get("extracted_from_entitydata") 
                                print(" Updated shipping GSTIN extracted_from_entitydata flag")
                            else:
                                print("Billing and shipping GSTIN are not the same")
                    if (gstin_ship) and  (shippingGSTIN.get("text") != ''):
                        if (gstin_ship) and (billingGSTIN.get("text") != ''):
                            if shippingGSTIN.get("text") == billingGSTIN.get("text"):
                                prediction["billingGSTIN"]["extracted_from_entitydata"] = shippingGSTIN.get("extracted_from_entitydata") 
                                print(" Updated billing GSTIN extracted_from_entitydata flag")
                            else:
                                print("Billing and shipping GSTIN are not the same")
                        
        else:
            print("Total unique GSTIN less than two or more")
        return prediction
    except :
        print("Copy GSTIN exception",traceback.print_exc())
        return pred_copy
@putil.timing
def validating_amount_fields_increasing_confidence(DF,prediction)-> dict:

    pred_copy = copy.deepcopy(prediction) 
    try:
        '''
        -> calculating total by adding taxes and subtracting discount
        -> validating calculated total with first and sencond max amount 
        '''

        fields = ["totalAmount","subTotal","CGSTAmount","SGSTAmount","IGSTAmount",
                  "CessAmount","additionalCessAmount","discountAmount"]
        field_values = {}
        for f in fields:
            if (prediction.get(f)) and (prediction.get(f).get("text") != ''):
                try:
                    field_values.update({f:float(prediction.get(f).get("text"))})
                except :
                    field_values.update({f : 0})
            else:
                field_values.update({f : 0}) 
        GSTAmount = 0
        Cess = 0
        for key, val in field_values.items():
            print(key," : ",val)
            if val is not None:
                if key in ["CGSTAmount","SGSTAmount","IGSTAmount"]:
                    GSTAmount = GSTAmount + val
                    pass
                if key in ["CessAmount","additionalCessAmount"]:
                    Cess = Cess + val
                    pass
        
        discountAmount = field_values.get("discountAmount")
        total = field_values.get("totalAmount")
        subtotal = field_values.get("subTotal")
        calculatedSubTotal = total - (GSTAmount + Cess + discountAmount)
        print("calcualted subtotal :",calculatedSubTotal)
        if subtotal ==0:
            if DF[DF["second_max_amount"]==1].shape[0]>0:
                second_max_amt = DF[DF["second_max_amount"]==1] 
                print("inside subtotal check df :",second_max_amt.shape)
                second_max_amt = second_max_amt["text"].iloc[0]
                print("second_max_amt ;",second_max_amt,"calcualted subtotal :",calculatedSubTotal)
                try:
                    second_max_amt = float(str(second_max_amt).replace(',','').replace(':','').replace('/','').replace(u'\u20B9',''))
                    print("abs with second max amount :",abs(second_max_amt -calculatedSubTotal))
                    if abs(second_max_amt -calculatedSubTotal) < 2:
                        prediction.update(add_new_field(field_name = "subTotal",final_confidence_score =1,value = second_max_amt,calculated=False))
                        for key, val in field_values.items():
                            if (val is not None):
                                prediction = reduce_field_confidence(prediction, key,model_confidence = 1,final_confidence_score = 1)
                        prediction = reduce_field_confidence(prediction, "totalGSTAmount",model_confidence = 1,final_confidence_score = 1)
                    
                except:
                    print("flaot conversion error ")
                    pass

        calculatedTotal =  (GSTAmount + Cess + subtotal)-discountAmount
        print("calculatedTotal after substracting discount :",calculatedTotal)
        abs_sub = abs(subtotal- calculatedSubTotal)
        abs_total = abs(total-calculatedTotal)
        print("abs_sub ;",abs_sub,"abs_total :",abs_total)
        if (abs_sub < 2) and (abs_total <2):
            for key, val in field_values.items():
                if val == 0.0:
                    # print("increasing field confidence :",key)
                    prediction = reduce_field_confidence(prediction, key,model_confidence = 1,final_confidence_score = 1)
                    prediction = reduce_field_confidence(prediction, "totalGSTAmount",model_confidence = 1,final_confidence_score = 1)
                
        return prediction
    except :
        print("validating_amount_fields_increasing_confidence :",traceback.print_exc())
        return pred_copy

def extract_missing_left_label_amount_field_from_table(df,subStngToMatch:str):
    """
    Extracting tax amounts only if there is no left label if values present in tabular form
    and if there is no extraction for thes fields
    """
    try:
        temp = df[df["is_amount"]==1]
        temp = temp[temp["line_row"]==0]
        # print("2161",temp.shape)
        # temp["keysMatch"] = temp["above_processed_ngbr"].str.contains(subStngToMatch,case=False,regex=True)
        # temp = temp[temp["keysMatch"]==True]
        temp = temp[temp["above_processed_ngbr"].str.contains(subStngToMatch,case=False,regex=True)==True]
        # print("2163",temp.shape)
        # print("shape of final candidates :",temp.shape)
        if not(temp.shape[0]):
            temp = df[df["is_amount"]==1]
            temp = temp[temp["above_processed_ngbr"].str.contains(subStngToMatch,case=False,regex=True)==True]
            # print("2170",temp.shape)
            # temp.to_csv(subStngToMatch+".csv")
            if not(temp.shape[0]):
                return None
        # temp = temp.drop_duplicates('text').sort_index().sort_values(by=["text"],ascending = True)
        amt_lst = []
        for item in list(set(temp["text"])):
            if str(item).replace(',','').replace(':','').replace('.','').replace(u'\u20B9','').isdigit():
                x = float(str(item).replace(',','').replace(':','').replace('/','').replace(u'\u20B9',''))
                amt_lst.append(x)
        amt_lst = sorted(amt_lst,reverse=True)
        print("amt_lst :",amt_lst)
        if len(amt_lst)>0:
            return amt_lst[0]
        return None
    except :
        print(" exception extract_missing_left_label_amount_field_from_table",traceback.print_exc())
        return None

def validating_extracted_amount_fields_without_left_label(prediction:dict,df)->dict:
    pred_copy = copy.deepcopy(prediction)
    try:
        print("validating_extracted_amount_fields_without_left_label")
        # 24 August 2023 Used iloc to fetch the data
        # is_CGST_SGST = df["is_CGST_SGST"][0]
        # is_IGST = df["is_IGST"][0]
        is_CGST_SGST = df.iloc[0]["is_CGST_SGST"]
        is_IGST = df.iloc[0]["is_IGST"]
        # print("is_CGST_SGST :",is_CGST_SGST)
        total =  prediction.get("totalAmount")
        if total and total.get("text")!= '':
            total = float(total.get("text"))
        subtotal =  prediction.get("subTotal")
        if subtotal and subtotal.get("text")!= '':
            subtotal = float(subtotal.get("text"))
        cgst = prediction.get("CGSTAmount")
        sgst = prediction.get("SGSTAmount")
        igst = prediction.get("IGSTAmount")
        cess = prediction.get("CessAmount")
        if cess and cess.get("text")!= '':
            cess = float(cess.get("text"))
        addCess = prediction.get("additionalCessAmount")
        if addCess and addCess.get("text")!= '':
            addCess = float(addCess.get("text"))
        print("total :",total,"/subtotal :",subtotal)
        print("befor extracting")
        print("cgst :",cgst,"\nsgst :",sgst,"\nigst :",igst,"\ncess :",cess,"\naddCess :",addCess)
        CSGT_subStngToMatch = 'CGST AMT|CGST'
        SGST_subStngToMatch = 'SGST AMT|SGST'
        IGST_subStngToMatch = "IGST AMT|IGST"
        subtotal_subStngToMatch = "TAXABLE"

        if not(cgst) and not(sgst)and not(igst):
            cgst = extract_missing_left_label_amount_field_from_table(df,CSGT_subStngToMatch)
            sgst = extract_missing_left_label_amount_field_from_table(df,SGST_subStngToMatch)
            igst = extract_missing_left_label_amount_field_from_table(df,IGST_subStngToMatch)
            if not (subtotal):
                subtotal_ext = extract_missing_left_label_amount_field_from_table(df,subtotal_subStngToMatch)
            else :
                subtotal_ext = extract_missing_left_label_amount_field_from_table(df,subtotal_subStngToMatch)
                print("subtotal_ext:",subtotal_ext)

            print("total :",total,"\nsubtotal :",subtotal)
            print("affter extracting")
            print("cgst :",cgst,"\nsgst :",sgst,"\nigst :",igst)

            add_sum = 0
            if total:
                if (is_CGST_SGST ==1)and (is_IGST==0):
                    if cgst and sgst:
                        if subtotal and (abs(total-(cgst+sgst+subtotal))<2):
                            prediction.update(add_new_field(field_name = "CGSTAmount",value = cgst))
                            prediction.update(add_new_field(field_name = "SGSTAmount",value = cgst))
                            print("matched CGST SGST extracted")
                        elif subtotal_ext and (abs(total-(cgst+sgst+subtotal_ext))<2):
                            prediction.update(add_new_field(field_name = "CGSTAmount",value = cgst))
                            prediction.update(add_new_field(field_name = "SGSTAmount",value = cgst))
                            prediction.update(add_new_field(field_name = "subTotal",value = subtotal_ext))
                            print("Extract new subtotal matched")
                        else:
                            print("Difference is greater than 2 : total ",total,":subtotal ",subtotal,": subtotal_ext ",subtotal_ext)
                if (is_CGST_SGST ==0)and (is_IGST==1):
                    if igst:
                        if subtotal and (abs(total-(igst+subtotal))<2):
                            prediction.update(add_new_field(field_name = "IGSTAmount",value = cgst))
                            print("match igst extracted")
                        elif subtotal_ext and (abs(total - (igst+subtotal_ext))):
                            prediction.update(add_new_field(field_name = "IGSTAmount",value = cgst))
                            prediction.update(add_new_field(field_name = "subTotal",value = subtotal_ext))
                            print("matched with calculated subtotal")
                        else:
                            print("Difference is greater than 2 :",total,add_sum)

        return prediction
    except :
        print("validating_extracted_amount_fields_without_left_label :",traceback.print_exc())
        return pred_copy
@putil.timing        
def vendor_name_validation(prediction,vendor_master):
    temp = copy.deepcopy(vendor_master)
    try:
        vendorGSTIN = prediction.get("vendorGSTIN")
        vendorName = prediction.get("vendorName")
        if vendorGSTIN and vendorGSTIN.get("text")=="N/A":
            print("vendorGSTIN is N/A","\nvendorName :",vendorName.get("text"))
            if vendorName and (vendorName.get("text")!=''):
                temp["VENDOR_NAME"] = temp["VENDOR_NAME"].fillna('')
                temp["VENDOR_NAME"] = temp["VENDOR_NAME"].apply(remove_commanwords)
                temp["vendor_match_score"] = temp.apply(lambda row:rapidfuzz.fuzz.WRatio(row["VENDOR_NAME"],remove_commanwords(vendorName.get("text"))),
                                                        axis=1)
                temp = temp[temp["vendor_match_score"]>95]
                print("cut of candidates:",temp.shape,"\t max score",temp["vendor_match_score"].max())
                if (temp.shape[0]) and (temp.shape[0] == 1):
                    # for correcting need to replace name with matched cutoff score
                    vendorMatched = temp.iloc[0].to_dict()
                    vendorName["extracted_from_masterdata"] = True
                    # fixed bug overwriting vendor name -> 12 Dec
                    # prediction["vendorName"] = str(vendorMatched["VENDOR_NAME"]).strip()
                    prediction["vendorName"]["text"] = str(vendorMatched["VENDOR_NAME"]).strip()
                    # fixed bug overwriting vendor name -> 12 Dec
                    # print("updated flag:",prediction["vendorName"])
        return prediction 
    except:
        print("exception:",traceback.print_exc())
        return prediction

def remove_commanwords(string):
    try:
        remove_comman_words = ["pvt","private","pvt.","ltd","ltd.","limited"]
        words = str(string).split()
        words = [word for word in words if word.lower() not in remove_comman_words]
        return ' '.join(words)
    except:
        print("remove_commanwords exception :",traceback.print.exc())
        return string

def split_inv_num_inv_date(predition:dict)->dict:
    try :
        vendor_gstin = predition.get("vendorGSTIN")
        print("vendor_gstin of Gujrant co-op milk:",vendor_gstin)
        if vendor_gstin:
            if (vendor_gstin.get("text") == "27AAAAG5588Q1ZW"):
                inv_num = predition.get("invoiceNumber")
                if (inv_num):
                    inv_num = str(inv_num.get("text")).split('/')
                    print("inv_num split :",inv_num)
                    predition["invoiceNumber"]["text"] = inv_num[0]
        
                inv_date = predition.get("invoiceDate")
                if (inv_date):
                    inv_date = str(inv_date.get("text")).split('/')
                    print("inv_num split :",inv_date)
                    predition["invoiceDate"]["text"] = inv_date[1]

        return predition
    except:
        print("split_inv_num_inv_date exception :",traceback.print_exc())
        return predition

import string    
def get_city_name(df:pd.DataFrame,cities:list,lookin_addres:str):
    city = ""
    try:
        cities = [str(x).upper().translate(str.maketrans('', '', string.punctuation)) for x in cities]
        #print("Cities",cities)
        check_list = df[df[lookin_addres]==1]
        check_list = ' '.join(list(check_list["text"]))
        check_list = str(str(check_list).upper()).translate(str.maketrans('', '', string.punctuation)).split()
        # Removing Duplicates from list
        #print("check_list of address",check_list)
        check_list = [*set(check_list)]
        for item in cities:
            if item in check_list:
                city = item
                print("Matched City :",city)
                return city
        print("Matched City :",city)
                    
        if city == "":
            for item in cities:
                filter_cities = list(set([item for word in df["text"] if item in word.upper()]))
                if len(filter_cities)>0:
                    city = str(filter_cities[0])
                    return city
        return city
    except:
        print("get_city_name exception",traceback.print_exc())
        return city
 
def convert_total_amount(final_prediction):
    prediction_copy = final_prediction.copy()
    try:
        if final_prediction.get("totalAmount") != None:
            final_prediction["totalAmount"]["text"] = final_prediction["totalAmount"]["text"].replace(",","")
        return final_prediction
    except:
        print("Exception in convert_total_amount")
        return prediction_copy
    
def add_bill_2_ship_city(pred_df:pd.DataFrame,predictiion:dict,entity_master_data:pd.DataFrame)->dict:
    """
    Return/call city extraction logic
    
    """
    predictiion.update(add_new_field(field_name="billingCity",value=""))
    predictiion.update(add_new_field(field_name="shippingCity",value=""))

    try:
        if predictiion.get("billingGSTIN"):
            if (predictiion.get("billingGSTIN").get("text")!=''):
                billingGSTIN = predictiion.get("billingGSTIN").get("text")
                bill_add = entity_master_data[entity_master_data["GSTIN"]==billingGSTIN]
                if bill_add.shape[0]>0:
                    b_cities = list(bill_add["CITY"])
                    print("Billing GSTIN cities :",b_cities)
                    billingCity = get_city_name(df=pred_df,cities=b_cities,lookin_addres="billingAddress")
                    print("Matched BillingCCity :",billingCity)
                    predictiion.update(add_new_field(field_name="billingCity",value=billingCity))
                else:
                    print("GSTIN is not present in entity master data")
            else:
                print("Billing GSTIN is empty string")
        
        if predictiion.get("shippingGSTIN"):
            if (predictiion.get("shippingGSTIN").get("text")!=''):
                shippingGSTIN = predictiion.get("shippingGSTIN").get("text")
                ship_add = entity_master_data[entity_master_data["GSTIN"]==shippingGSTIN]
                if ship_add.shape[0]>0:
                    s_cities = list(ship_add["CITY"])
                    print("Shipping GSTIN cities :",s_cities)
                    shippingCity = get_city_name(df=pred_df,cities=s_cities,lookin_addres="shippingAddress")
                    print("Matched shippingCity:",shippingCity)
                    predictiion.update(add_new_field(field_name="shippingCity",value=shippingCity))
                else:
                    print("GSTIN is not present in entity master data")
            else:
                print("shipping GSTIN is empty string")
        return predictiion

    except:
        print("add_bill_2_ship_city exception :",traceback.print_exc()) 
        return predictiion

def get_vendor_code(final_prediction,docMetaData, VENDOR_MASTERDATA):
    from business_rules import add_empty_field
    final_prediction.update(add_empty_field("vendorCode","N/A"))
    final_prediction["vendorCode"]["dropDown"] =  1
    final_prediction["vendorCode"]["dropDownOptions" ] = ["N/A"]
    final_prediction_copy = final_prediction.copy()
    try:
        if docMetaData.get("result").get("document").get("docType") == "Invoice":
            if (final_prediction.get("vendorGSTIN") != None) and (final_prediction.get("vendorGSTIN").get("text") != None) and (final_prediction.get("vendorGSTIN").get("text") != "N/A") :
                vgstin_predicted = final_prediction.get("vendorGSTIN").get("text")
                vendor_code, list_vendor_code = get_vendor_code_after_review(VENDOR_MASTERDATA, vgstin_predicted)
                if len(list_vendor_code) == 1:
                    print("single vendor code is present. Updating the vendor code field")
                    final_prediction.update(add_new_field(field_name="vendorCode",value=vendor_code))
                    final_prediction["vendorCode"]["dropDown"] =  1
                    final_prediction["vendorCode"]["dropDownOptions" ] = list_vendor_code
                elif len(list_vendor_code) > 1:
                    print("Multiple vendoe codes are present. Updating the final list")
                    list_vendor_code.append("N/A")
                    final_prediction["vendorCode"]["dropDownOptions" ] = list_vendor_code            
        return final_prediction
    except:
        print("get_vendor_code exception :",traceback.print_exc()) 
        return final_prediction_copy
@putil.timing
def apply_client_rules(DF, prediction, docMetaData,ADDRESS_MASTERDATA,VENDOR_MASTERDATA,doc_vendorGSTIN,format_= None):
    print("Started applying  client rules")
    prediction = split_inv_num_inv_date(prediction)
    # prediction = discard_lines_without_mandatory_fields(prediction)
    prediction = discard_additional_LI_rows(prediction)
    prediction = demo_change(prediction)
    #check_preform = extract_preform(DF)
    #if check_preform is not None:
     #   prediction = {**prediction, **check_preform}
    #out_dict = extract_barcode(DF)
    #if out_dict is not None:
     #   prediction = {**prediction, **out_dict}
    #prediction = clean_HSNCode(prediction)
    prediction = clean_PONumber(prediction)
    # prediction = convert_dates(prediction)
    # prediction = make_vendor_info_editable(prediction)
    # prediction = get_billingName(DF,prediction)
    # prediction = get_shippingName(DF,prediction)
    # prediction = get_billingGSTIN(DF,prediction)
    # prediction = get_shippingGSTIN(DF,prediction)
    #prediction = build_final_QRCode_json(prediction,docMetaData)
    # if qr_pred is not None:
    #      prediction = qr_pred
    #print("extractedQRCodeData",prediction)
    #prediction = validate_Model_Prediction_with_QRCode_Data(docMetaData,prediction)
    #extracted_org = extract_org(DF)
    #prediction = {**prediction, **extracted_org}
    # prediction = getBill2Shop2Details(DF, prediction)
    # print("predcition before gsting extraction :",prediction)
    prediction = validate_VGSTIN(DF,prediction,doc_vendorGSTIN,VENDOR_MASTERDATA)
    prediction,B_Assigned,S_Assigned = get_GSTIN_fields(DF, prediction, ADDRESS_MASTERDATA,VENDOR_MASTERDATA)
    prediction = extract_GSTIN_from_string(prediction)
    prediction = copy_gstin(DF, prediction, B_Assigned, S_Assigned)
    # 13 April 2023 Added the below function to identify billing gstin and name after some OCR Issues
    prediction = extract_buyer_gstin_name_after_ocr_issue(DF,prediction,VENDOR_MASTERDATA, ADDRESS_MASTERDATA)
    prediction = copy_shipping_gstin_after_ocr_issue(prediction)
    # 8 May 2023 Added the below function to identify vendor gstin and name after some OCR Issues
    prediction = extract_vendor_gstin_name_after_ocr_issue(DF,prediction,VENDOR_MASTERDATA, ADDRESS_MASTERDATA)
    # Calling in business rule before apply gstin rules Fri 30 2022
    prediction = get_vendor_buyers_name(DF,prediction,ADDRESS_MASTERDATA,VENDOR_MASTERDATA)
    
    prediction = get_totalGSTAmount(DF, prediction)
    prediction = extract_vendorPAN(prediction)
    prediction = add_bill_2_ship_city(pred_df=DF,predictiion=prediction,entity_master_data=ADDRESS_MASTERDATA)
    prediction = validating_extracted_amount_fields_without_left_label(prediction,DF)
    prediction = calculate_total(DF,prediction)
    # print("calculate_total predict :",prediction)
    prediction = calculate_total2(DF, prediction)
    # prediction = check_if_future_Date(prediction)
    # prediction = check_left_above_invdt_ngbr(DF,prediction)

    # calling confidence reduction fuction aftrer applying all rules
    prediction = reduction_confidence_taxes(prediction)
    prediction = reduce_amount_fields_confidenace(prediction)
    
    # 10 Oct 2023 Added get_max_invoice_date to get invoice date based on date keyowrd even if fuzzy is low
    #prediction = get_max_invoice_date(DF,prediction)
    return prediction

# 10 Oct 2023 Added get_max_invoice_date to get invoice date based on date keyowrd even if fuzzy is low
def get_max_invoice_date(DF,prediction):
    # works only if there is no prediction for invoice date earlier
    prediction_copy = prediction.copy()
    try:
        if prediction_copy.get("invoiceDate") == None:
            from modify_prediction import parse_date
            max_date = None
            token_id = None
            dated_max = {}
            minimum_inLI = list(DF["line_row_new"].unique())
            if len(minimum_inLI) != 0:
                minimum_inLI.remove(0)
                if len(minimum_inLI) != 0:
                    min_LI = min(minimum_inLI)
                    b = DF.loc[DF["line_row_new"] == min_LI,["token_id"]]
                    tid = min(list(b["token_id"].unique()))
                    # print("sahil12",min_LI, tid)
                    filter_df = DF[(DF["line_row_new"]==0) & ( DF["is_date"] == 1) & (DF["token_id"]<tid)]
                else:
                    print("No Line Items")
                    filter_df = DF[(DF["line_row_new"]==0) & (DF["is_date"] == 1)]
            else:
                filter_df = DF[DF["is_date"] == 1]
            for row in filter_df.itertuples():
                try:
                    if ("date" in str(row.above_processed_ngbr).lower()) or ("date" in str(row.left_processed_ngbr).lower()) or (("inv" in str(row.left_processed_ngbr).lower()) and ("dt" in str(row.left_processed_ngbr).lower())):
                        # Removing keywords which contain max date but are not Invoice date
                        print("date keyword", row.token_id, row.text)

                        if (("ack" in str(row.above_processed_ngbr).lower() and "ack" in str(row.W1Ab).lower()) 
                            or ("ack" in str(row.left_processed_ngbr).lower()) 
                            or ("lr" in str(row.left_processed_ngbr).lower())
                            or ("dng" in str(row.left_processed_ngbr).lower())
                            or ("so" in str(row.left_processed_ngbr).lower())
                            or (("tax" in str(row.left_processed_ngbr).lower() and "tax invoice" not in str(row.left_processed_ngbr).lower()))
                            or ("due" in str(row.left_processed_ngbr).lower()) 
                            or (("exp" in str(row.left_processed_ngbr).lower()) 
                            and ("date" in str(row.left_processed_ngbr).lower())) 
                            or (("exp" in str(row.above_processed_ngbr).lower()) 
                            and ("date" in str(row.above_processed_ngbr).lower())) ):
                            print("ack found ", row.token_id, row.text)
                            continue
                        try:
                            validate_date = parse_date(row.text)
                            print("validate_date", validate_date)
                            if validate_date != None:
                                validate_date = parser.parse(validate_date, dayfirst=True).date()
                                # print(validate_date)
                                formate_date = parser.parse(row.text, dayfirst=True).date().strftime('%d/%m/%Y')
                                # print("sahil", formate_date)
                                dated_max[row.token_id]=formate_date
                                # date_list.append(validate_date)
                        except:
                            print("format_exception")
                        if max_date is None and validate_date != None:
                            max_date=validate_date
                            token_id=row.token_id
                            # print("first_if",max_date)
                        if max_date is not None and validate_date != None and max_date<=validate_date :
                            max_date=validate_date
                            token_id=row.token_id
                            # print("second_if",max_date)
                                # break
                except:
                    print("Exception in max_date function for date keyword")
            # print("sahil finally", max_date, token_id)
            if token_id:
                token_df = DF[DF["token_id"] == token_id]
                # print("token_df :",token_df.shape)
                for _, row in token_df.iterrows():
                    prediction.update(add_new_fields("invoiceDate",row,from_entity=True))
        return prediction
        
    except Exception as e:
        print("Exception in get_max_invoice_date", e)
        return prediction_copy

def getMatchDatesValidate(text:str):
    datePatternsValidation = r'^\d{1,2}/\d{1,2}/\d{4}$'
    try:
        if re.match(datePatternsValidation, text):
            print("Matched")
            return True
        else:
            print("Not matched. Returning False")
            return False
    except:
        print("Find date pattern exception",traceback.print_exc())
        print("returning False")
        return False
## 12 Feb 2024 Added validation to check month(middle) should not be greater than 12    
def validate_date_month_middle(date_string):
    try:
        # Attempt to parse the date string
        day, month, year = map(int, date_string.split('/'))
        print("day:",day)
        # Perform basic checks on day, month, and year
        if month < 1 or month > 12:
            return False
        if day < 1 or day > 31:
            return False
        if month in [4, 6, 9, 11] and day > 30:
            return False
        if month == 2:
            if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                if day > 29:
                    return False
            elif day > 28:
                return False
        
        # If all checks pass, return True
        return True
    except ValueError:
        # If any conversion fails (e.g., non-integer input), return False
        return False
# March 1 2023, Added business rule for tax slab
def bizRuleValidateForUi(documentId,callBackUrl,UI_validation = False, VENDOR_ADDRESS_MASTERDATA = None):
    # 12 sep 22  CR to make Vendor GSTIN and PAN optional

    def mandatoryMsg(fldId):
        return fldId + " cannot be blank"

    def invalidAmtMsg(fldId):
        return fldId + " is an amount field and must contain only numbers"

    def invalidDtMsg(fldId):
        return fldId + " is a date field and must contain a valid date format starting with day"

    def invalidFormat(fldId,format_):
        return "The format of " + fldId + " is " + format_ + ". Please ensure the value is in this format."


    try:
        result = []
        document_result_updated = 0
        docResult = putil.getDocumentResultApi(documentId,
                                               callBackUrl)
        docMetatadata = putil.getDocumentMetadata(documentId,callBackUrl)
        # 13 April 2023 Added functionality for getting bizrules for discrepancy note
        #print("Inside biz rules docmetadata is:",docMetatadata)
        stp_type= "DEFAULT"
        if docMetatadata != None and docMetatadata.get("result")!= None and docMetatadata.get("result").get("document") != None and docMetatadata.get("result").get("document").get("docType") != None:
            stp_type = docMetatadata.get("result").get("document").get("docType")
            if stp_type.lower() == "invoice":
                stp_type= "DEFAULT"
            print("Doctype is :", docMetatadata.get("result").get("document").get("docType"), "for document id", documentId)
            
        if docResult is None:
            return None, document_result_updated, docResult
        
        ## Added derived field logic
        if UI_validation:
            print("Deriving the field only in case of UI validation is true")
            status_derived_field, docResult, result = derive_custom_fields(docMetatadata, docResult, result, documentId, VENDOR_ADDRESS_MASTERDATA, UI_validation)
            if (status_derived_field): 
                status_update = putil.updateDocumentResultApi(documentId, callBackUrl, docResult)
                print("Status for update is:", status_update)
                if status_update == None:
                    raise Exception ## Raise an exception if status of update is None
                else:
                    print("Changing variable document_result_updated state to 1")
                    document_result_updated = 1
        extractedFields = docResult["result"]["document"]["documentInfo"]
        mandatoryFields = putil.getMandatoryFields(stp_type)
        fldTypes = putil.getFldTypeFormat(stp_type)
        # print("Field Types",fldTypes)
        if mandatoryFields is None:
            raise Exception
        flds = {}
        v_gstin = None
        pan_pattern = r'[A-Z]{5}\d{4}[A-Z]{1}'
        for idx,fld in enumerate(extractedFields):
            fldId = fld["fieldId"]
            fldVal = fld["correctedValue"] if "correctedValue" in fld.keys() else fld["fieldValue"]
            flds[fldId] = fldVal
            if fldId == "vendorGSTIN":
                v_gstin = fldVal
            #Do Mandatory check
            if fldVal == "" and fldId in mandatoryFields:
                # return mandatoryMsg(fldId)
                # 12th sep changes for making vendor GSITN and PAN optional. 13-sep the CR revoke
                result.append((fldId,mandatoryMsg(fldId)))
                # if not (UI_validation and (fldId in ["vendorGSTIN","vendorPAN"])):
                #     result.append((fldId,mandatoryMsg(fldId)))
                # 12th sep changes for making vendor GSITN and PAN optional. 13-sep the CR revoke

            #Do Alpha-numeric check
            if fldId in [fld_[0] for fld_ in fldTypes if fld_[1].lower() == "alpha-numeric"]:
                # 12th sep changes for making vendor GSITN and PAN optional. 13-sep the CR revoke
                # if not putil.validAlphaNumeric(fldVal):
                #     result.append((fldId,
                #         fldId + " is an alpha-numeric field and must contain at-least one alphabet or number"))
                if UI_validation:
                    if (fldId not in ["vendorGSTIN","vendorPAN"]):
                        if not putil.validAlphaNumeric(fldVal):
                            result.append((fldId,
                                fldId + " is an alpha-numeric field and must contain at-least one alphabet or number"))
                    else:
                        if (fldVal != "N/A"):
                            if not putil.validAlphaNumeric(fldVal):
                                result.append((fldId,
                                    fldId + " is an alpha-numeric field and must contain at-least one alphabet or number"))
                        # Added Incorrect PAN correction Feb 8 2023
                        if (fldId == "vendorPAN"):
                            print("v_gstin :-",v_gstin)
                            if (v_gstin and v_gstin != "N/A"):
                                p_pattern = re.findall(pan_pattern, fldVal, flags = 0)

                                if (len(p_pattern)==0):
                                    result.append((fldId,
                                        "Incorrect vendorPAN or vendorGSTIN"))
                                # Checking exact charactor match
                                elif (fldVal not in v_gstin):
                                    result.append((fldId,
                                        "Incorrect vendorPAN or vendorGSTIN"))
                        # Added Incorrect PAN correction Feb 8 2023
                else:
                    if not putil.validAlphaNumeric(fldVal):
                        result.append((fldId,
                            fldId + " is an alpha-numeric field and must contain at-least one alphabet or number"))
                # 12th sep changes for making vendor GSITN and PAN optional. 13-sep the CR revoke

            #Do numeric check
            if fldId in [fld_[0] for fld_ in fldTypes if fld_[1].lower() == "numeric"]:
                if not putil.validAmount(fldVal):
                    result.append((fldId,
                                    invalidAmtMsg(fldId)))
            #Do date check
            if fldId in [fld_[0] for fld_ in fldTypes if fld_[1].lower() == "date"]:
                if putil.validDate(fldVal) == 100:
                    result.append((fldId,
                                    invalidDtMsg(fldId)))
                elif putil.validDate(fldVal) == 200:
                    result.append((fldId,
                                   "Invoice Date cannot be a future date"))
                elif putil.validDate(fldVal) == 300:
                    result.append((fldId,
                                    "Invoice date has some punctuations added. Please clean up those punctuations."))
            #Do format check
            fldFormat = "".join(["X" if c.isalpha() else ("0" if c.isnumeric() else c) for c in list(fldVal)])
            defFormat = [fld_[2] for fld_ in fldTypes if fld_[2] is not None and fld_[0] == fldId]
            # print("Format ",fldId,defFormat)
            if len(defFormat) == 1:
                defFormat_ = defFormat[0]
                if defFormat_ != "":
                # 12th sep changes for making vendor GSITN and PAN optional. 13-sep the CR revoke
                    # if not putil.checkValidFormat(defFormat_,fldFormat):
                    #     result.append((fldId, invalidFormat(fldId, defFormat_)))
                    if UI_validation:
                        if (fldId in ["vendorGSTIN","vendorPAN"]):
                            if fldVal != "N/A":
                                if not putil.checkValidFormat(defFormat_,fldFormat):
                                    result.append((fldId, invalidFormat(fldId, defFormat_)))
                        elif (fldId in ["billingGSTIN", "shippingGSTIN"]):
                            ## March-08-2024 Added becasue of CR(only for Blinkit), we are GSTIN from dropdown
                            print("Not checking BillingGSTIN and ShippingGSTIN's format")
                        else:
                            if not putil.checkValidFormat(defFormat_,fldFormat):
                                result.append((fldId, invalidFormat(fldId, defFormat_)))
                    else:
                        if not putil.checkValidFormat(defFormat_,fldFormat):
                            result.append((fldId, invalidFormat(fldId, defFormat_)))
                # 12th sep changes for making vendor GSITN and PAN optional. 13-sep the CR revoke
            
            ## 26 sept 2023 Only for blinkit, checking value of Invoicedate in dd/mm/yyyy format 
            if fldId == "invoiceDate" and cfg.get_blinkit_date_format():
                if (type(fldVal) == str) and not(getMatchDatesValidate(fldVal)):
                    result.append((fldId,
                            "Please provide correct Input(dd/mm/yyyy)"))
                if  (type(fldVal) == str) and not (validate_date_month_middle(fldVal)):
                    result.append((fldId,
                            "Please provide correct Input(dd/mm/yyyy)"))
            ## 27 Oct 2023 Adding validations for vendor code
            if (fldId == "vendorCode") and (UI_validation == False):
                ## Used for cases where GSTIN's are not extracted properly. Should not move to STP without vendor code
                if (type(fldVal) == str) and ((fldVal == "N/A") or (fldVal.find(",")==1) or (fldVal == "")):
                    result.append((fldId,
                            "Please provide correct Input for Vendor Code"))
            ## 7 March 2024, Adding UI validations for billingGSTIN's, shippingGSTIN's and vendorCode
            if UI_validation: 
                if fldId == "billingGSTIN":
                    ## auto-reassign the document if current stage is review and billingGSTIN is N/A
                    result = checkingGSTINDetails(docMetatadata, docResult, result, documentId, fld)
                if fldId == "shippingGSTIN":
                    ## auto-reassign the document if current stage is review and billingGSTIN is N/A
                    result = checkingGSTINDetails(docMetatadata, docResult, result, documentId, fld)
                if fldId == "vendorCode":
                    result = checkVendorCode(docMetatadata, docResult, result, documentId)
                    pass     
        #Functional rule
        vendorGSTIN = flds.get("vendorGSTIN")
        billingGSTIN = flds.get("billingGSTIN")
        shippingGSTIN = flds.get("shippingGSTIN")
        if docMetatadata.get("result") != None and docMetatadata.get("result").get("document") != None and docMetatadata.get("result").get("document").get("docType") !=None and docMetatadata.get("result").get("document").get("docType") == "Discrepancy Note":
            print("Getting GSTIN for discr note")
            vendorGSTIN = docMetatadata.get("result").get("document").get("linked_document").get("vendorGSTIN")
            billingGSTIN = docMetatadata.get("result").get("document").get("linked_document").get("billingGSTIN")
            shippingGSTIN = docMetatadata.get("result").get("document").get("linked_document").get("shippingGSTIN")
            print(vendorGSTIN,billingGSTIN,shippingGSTIN)
        # 20 April 2023 Removed only for blinkit
        #IGSTAmount = flds.get("IGSTAmount")
        #SGSTAmount = flds.get("SGSTAmount")
        #CGSTAmount = flds.get("CGSTAmount")
        #subTotal = flds.get("subTotal")
        totalGSTAmount = flds.get("totalGSTAmount")
        discountAmount = flds.get("discountAmount")
        cessAmount = flds.get("CessAmount")
        TCSAmount = flds.get("tcsAmount")
        additionalCessAmount = flds.get("additionalCessAmount")
        totalAmount = flds.get("totalAmount")

        """IGSTAmount_5 = flds.get("IGSTAmount_5%",0)
        IGSTAmount_12 = flds.get("IGSTAmount_12%",0)
        IGSTAmount_18 = flds.get("IGSTAmount_18%",0)
        IGSTAmount_28 = flds.get("IGSTAmount_28%",0)
        try:
            IGSTAmount_slab = float(IGSTAmount_5) + float(IGSTAmount_12) + float(IGSTAmount_18) + float(IGSTAmount_28) 
        except:
            IGSTAmount_slab = 0"""
        flag_float_conversion = True
        try:
            IGSTAmount_5 = float(flds.get("IGSTAmount_5%",0))
        except:
            flag_float_conversion = False
            IGSTAmount_5 = 0
        try:
            IGSTAmount_12 = float(flds.get("IGSTAmount_12%",0))
        except:
            flag_float_conversion = False
            IGSTAmount_12 = 0
        try:
            IGSTAmount_18 = float(flds.get("IGSTAmount_18%",0))
        except:
            flag_float_conversion = False
            IGSTAmount_18 = 0
        try:
            IGSTAmount_28 = float(flds.get("IGSTAmount_28%",0))
        except:
            flag_float_conversion = False
            IGSTAmount_28 = 0
       
        IGSTAmount_slab = IGSTAmount_5 + IGSTAmount_12 + IGSTAmount_18 + IGSTAmount_28
        
        """CGSTAmount_25 = flds.get("CGSTAmount_2.5%",0)
        CGSTAmount_6 = flds.get("CGSTAmount_6%",0)
        CGSTAmount_9 = flds.get("CGSTAmount_9%",0)
        CGSTAmount_14 = flds.get("CGSTAmount_14%",0)
        try:
            CGSTAmount_slab = float(CGSTAmount_25) + float(CGSTAmount_6) + float(CGSTAmount_9) + float(CGSTAmount_14)
        except:
            CGSTAmount_slab = 0"""
        
        try:
            CGSTAmount_25 = float(flds.get("CGSTAmount_2.5%",0))
        except:
            flag_float_conversion = False
            CGSTAmount_25 = 0
        try:    
            CGSTAmount_6 = float(flds.get("CGSTAmount_6%",0))
        except:
            flag_float_conversion = False
            CGSTAmount_6 = 0
        try:
            CGSTAmount_9 = float(flds.get("CGSTAmount_9%",0))
        except:
            flag_float_conversion = False
            CGSTAmount_9 = 0
        try:
            CGSTAmount_14 = float(flds.get("CGSTAmount_14%",0))
        except:
            flag_float_conversion = False
            CGSTAmount_14 = 0

        CGSTAmount_slab = CGSTAmount_25 + CGSTAmount_6 + CGSTAmount_9 + CGSTAmount_14
        """SGSTAmount_25 = flds.get("SGSTAmount_2.5%",0)
        SGSTAmount_6 = flds.get("SGSTAmount_6%",0)
        SGSTAmount_9 = flds.get("SGSTAmount_9%",0)
        SGSTAmount_14 = flds.get("SGSTAmount_14%",0)"""
        try:
            SGSTAmount_25 = float(flds.get("SGSTAmount_2.5%",0))
        except:
            flag_float_conversion = False
            SGSTAmount_25 = 0
        try:
            SGSTAmount_6 = float(flds.get("SGSTAmount_6%",0))
        except:
            flag_float_conversion = False
            SGSTAmount_6 = 0
        try:
            SGSTAmount_9 = float(flds.get("SGSTAmount_9%",0))
        except:
            flag_float_conversion = False
            SGSTAmount_9 = 0
        try:
            SGSTAmount_14 = float(flds.get("SGSTAmount_14%",0))
        except:
            flag_float_conversion = False
            SGSTAmount_14 = 0
        SGSTAmount_slab = SGSTAmount_25 + SGSTAmount_6 + SGSTAmount_9 + SGSTAmount_14

        """subTotal_5 = flds.get("subTotal_5%",0)
        subTotal_12 = flds.get("subTotal_12%",0)
        subTotal_18 = flds.get("subTotal_18%",0)
        subTotal_28 = flds.get("subTotal_28%",0)
        subTotal_0 = flds.get("subTotal_0%",0)
        try:
            subTotal_slab = float(subTotal_5) + float(subTotal_12) + float(subTotal_18) + float(subTotal_28) + float(subTotal_0)
        except:
            subTotal_slab = 0"""

        try:
            subTotal_5 = float(flds.get("subTotal_5%",0))
        except:
            flag_float_conversion = False
            subTotal_5 = 0
        try:
            subTotal_12 = float(flds.get("subTotal_12%",0))
        except:
            flag_float_conversion = False
            subTotal_12 = 0
        try:
            subTotal_18 = float(flds.get("subTotal_18%",0))
        except:
            flag_float_conversion = False
            subTotal_18 = 0
        try:
            subTotal_28 = float(flds.get("subTotal_28%",0))
        except:
            flag_float_conversion = False
            subTotal_28 = 0
        try:
            subTotal_0 = float(flds.get("subTotal_0%",0))
        except:
            flag_float_conversion = False
            subTotal_0 = 0
        subTotal_slab = subTotal_5 + subTotal_12 + subTotal_18 + subTotal_28 + subTotal_0
        calcTotalGST_slab = CGSTAmount_slab + SGSTAmount_slab + IGSTAmount_slab

        # 20 April 2023 Removed only for blinkit
        """
        try:
            convIGSTAmount = float(IGSTAmount) if IGSTAmount is not None else 0.0
        except:
            convIGSTAmount = 0.0
        try:
            convSGSTAmount = float(SGSTAmount) if SGSTAmount is not None else 0.0
        except:
            convSGSTAmount = 0.0
        try:
            convCGSTAmount = float(CGSTAmount) if CGSTAmount is not None else 0.0
        except:
            convCGSTAmount = 0.0
        try:
            convSubTotal = float(subTotal) if subTotal is not None else 0.0
        except:
            convSubTotal = 0.0
        """
        try:
            convTotalGST = float(totalGSTAmount) if totalGSTAmount is not None else 0.0
        except:
            convTotalGST = 0.0
        
        try:
            convTotalAmt = float(totalAmount) if totalAmount is not None else 0.0
        except:
            convTotalAmt = 0.0
        try:
            convDiscAmt = float(discountAmount) if discountAmount is not None else 0.0
        except:
            convDiscAmt = 0.0
        try:
            convCessAmt = float(cessAmount) if cessAmount is not None else 0.0
        except:
            convCessAmt = 0.0
        try:
            convAddlCess = float(additionalCessAmount) if additionalCessAmount is not None else 0.0
        except:
            convAddlCess = 0.0
        try:
            convTCSAmount = float(TCSAmount) if TCSAmount is not None else 0.0
        except:
            convTCSAmount = 0.0
        #calcTotalGST = convIGSTAmount + convSGSTAmount + convCGSTAmount
        calcAddlTax = convCessAmt + convAddlCess + convTCSAmount

        #print("Total Tax",calcTotalGST)
        #print("Sub Total",convSubTotal)
        print("Addl Tax", calcAddlTax)
        print("Total Amt",convTotalAmt)
        
        print("Total Tax Slab",calcTotalGST_slab)
        print("Total Sub Total Slab",subTotal_slab)

        # 20 April 2023 Removed only for blinkit
        """
        if not (math.isclose(convIGSTAmount,IGSTAmount_slab,abs_tol=0.5)):
            result.append(("IGSTAmount",
                          "Total IGST Amount should be sum of individual IGST slab Amounts.A tolerance of .5 Rs is acceptable"))

        if not (math.isclose(convCGSTAmount,CGSTAmount_slab,abs_tol=0.5)):
            result.append(("CGSTAmount",
                          "Total CGST Amount should be sum of individual CGST slab Amounts.A tolerance of .5 Rs is acceptable"))

        if not (math.isclose(convSGSTAmount,SGSTAmount_slab,abs_tol=0.5)):
            result.append(("SGSTAmount",
                          "Total SGST Amount should be sum of individual SGST slab Amounts.A tolerance of .5 Rs is acceptable"))

        if not (math.isclose(convSubTotal,subTotal_slab,abs_tol=0.5)):
            result.append(("subTotal",
                          "sub total should be sum of individual subtotal of  slab Amounts.A tolerance of .5 Rs is acceptable"))
        if calcTotalGST != convTotalGST and stp_type == "DEFAULT":
            result.append(("totalGSTAmount",
                          "Total GST Amount should be sum of individual GST Amounts"))
        """                                
        if convTotalAmt == 0:
            result.append(("totalAmount",
                          "Total Amount should contain a value greater than zero"))
        
        print("vendorGSTIN before GST Validation:", vendorGSTIN)
        gstin1 = vendorGSTIN
        if (shippingGSTIN is not None) and (shippingGSTIN != ""):
            gstin2 = shippingGSTIN
        elif (billingGSTIN is not None) and (billingGSTIN != ""):
            gstin2 = billingGSTIN
        else:
            gstin2 = "" 
        # 19 May 2023 Added validations for CGST/IGST Amount based on GSTIN
        if (gstin1 is not None) and (gstin2 is not None) and (gstin1 != "") and (gstin2 != ""):
            print("GSTIN1",gstin1)
            print("GSTIN2",gstin2)
            print("CGST",CGSTAmount_slab,
                    "SGST",SGSTAmount_slab,
                    "IGST",IGSTAmount_slab)
            print("condition",
                    gstin1[:2].upper() == gstin2[:2].upper())
            if (gstin1[:2].upper() == gstin2[:2].upper()) and (IGSTAmount_slab != 0):
                print("Not possible Amount field in IGST slab but present")
                result.append(("totalAmount",
                                "IGST Amount can not be greater than 0"))
            elif (gstin1[:2].upper() != gstin2[:2].upper()) and ((CGSTAmount_slab + SGSTAmount_slab) != 0):
                    print("Not possible Amount field in CGST/SGST slab but present")
                    result.append(("totalAmount",
                                    "CGST Amount can not be greater than 0"))

		## 10 August 2023 Added new validations for Slab percentage
        print("Flag Float value:", flag_float_conversion)
        if ((subTotal_5 > 0) or (CGSTAmount_25 > 0) or (SGSTAmount_25 > 0) or (IGSTAmount_5 > 0)) and (flag_float_conversion == True):
            if (gstin1[:2].upper() == gstin2[:2].upper()):
                calc_subtotal = subTotal_5 * 2.5 / 100
                if not math.isclose(calc_subtotal,CGSTAmount_25,abs_tol=0.5):
                    result.append(("subTotal_5%",
                          "Sub-Total And slab percentage are not matching. A tolerance of .5 Rs is acceptable"))
                if not math.isclose(calc_subtotal,SGSTAmount_25,abs_tol=0.5):
                    result.append(("subTotal_5%",
                          "Sub-Total And slab percentage are not matching. A tolerance of .5 Rs is acceptable"))
            elif (gstin1[:2].upper() != gstin2[:2].upper()):
                calc_subtotal = subTotal_5 * 5 / 100
                if not math.isclose(calc_subtotal,IGSTAmount_5,abs_tol=0.5):
                    result.append(("subTotal_5%",
                          "Sub-Total And slab percentage are not matching. A tolerance of .5 Rs is acceptable"))
        if ((subTotal_12 > 0) or (CGSTAmount_6 > 0) or (SGSTAmount_6 > 0) or (IGSTAmount_12 > 0)) and (flag_float_conversion == True):
            if (gstin1[:2].upper() == gstin2[:2].upper()):
                calc_subtotal = subTotal_12 * 6 / 100
                if not math.isclose(calc_subtotal,CGSTAmount_6,abs_tol=0.5):
                    result.append(("subTotal_12%",
                          "Sub-Total And slab percentage are not matching. A tolerance of .5 Rs is acceptable"))
                if not math.isclose(calc_subtotal,SGSTAmount_6,abs_tol=0.5):
                    result.append(("subTotal_12%",
                          "Sub-Total And slab percentage are not matching. A tolerance of .5 Rs is acceptable"))
            elif (gstin1[:2].upper() != gstin2[:2].upper()):
                calc_subtotal = subTotal_12 * 12 / 100
                if not math.isclose(calc_subtotal,IGSTAmount_12,abs_tol=0.5):
                    result.append(("subTotal_12%",
                          "Sub-Total And slab percentage are not matching. A tolerance of .5 Rs is acceptable"))
        if ((subTotal_18 > 0) or (CGSTAmount_9 > 0) or (SGSTAmount_9 > 0) or (IGSTAmount_18 > 0)) and (flag_float_conversion == True) :
            if (gstin1[:2].upper() == gstin2[:2].upper()):
                calc_subtotal = subTotal_18 * 9 / 100
                if not math.isclose(calc_subtotal,CGSTAmount_9,abs_tol=0.5):
                    result.append(("subTotal_18%",
                          "Sub-Total And slab percentage are not matching. A tolerance of .5 Rs is acceptable"))
                if not math.isclose(calc_subtotal,SGSTAmount_9,abs_tol=0.5):
                    result.append(("subTotal_18%",
                          "Sub-Total And slab percentage are not matching. A tolerance of .5 Rs is acceptable"))
            elif (gstin1[:2].upper() != gstin2[:2].upper()):
                calc_subtotal = subTotal_18 * 18 / 100
                if not math.isclose(calc_subtotal,IGSTAmount_18,abs_tol=0.5):
                    result.append(("subTotal_18%",
                          "Sub-Total And slab percentage are not matching. A tolerance of .5 Rs is acceptable"))
        if ((subTotal_28 > 0) or (CGSTAmount_14 > 0) or (SGSTAmount_14 > 0) or (IGSTAmount_28 > 0)) and (flag_float_conversion == True) :
            if (gstin1[:2].upper() == gstin2[:2].upper()):
                calc_subtotal = subTotal_28 * 14 / 100
                if not math.isclose(calc_subtotal,CGSTAmount_14,abs_tol=0.5):
                    result.append(("subTotal_28%",
                          "Sub-Total And slab percentage are not matching. A tolerance of .5 Rs is acceptable"))
                if not math.isclose(calc_subtotal,SGSTAmount_14,abs_tol=0.5):
                    result.append(("subTotal_28%",
                          "Sub-Total And slab percentage are not matching. A tolerance of .5 Rs is acceptable"))
            elif (gstin1[:2].upper() != gstin2[:2].upper()):
                calc_subtotal = subTotal_28 * 28 / 100
                if not math.isclose(calc_subtotal,IGSTAmount_28,abs_tol=0.5):
                    result.append(("subTotal_28%",
                          "Sub-Total And slab percentage are not matching. A tolerance of .5 Rs is acceptable"))
        # 4 May 2023 Removed total GST Amount from UI
        # 21 April 2023 Added validations for total GST Amount and Total Amount
        # if not math.isclose(convTotalGST,calcTotalGST_slab,abs_tol=0.5) and stp_type == "DEFAULT":
        #     result.append(("totalGSTAmount",
        #                   "Total GST Amount should be sum of individual GST slab Amounts.A tolerance of .5 Rs is acceptable"))
        
        if (subTotal_slab > 0 or calcTotalGST_slab > 0 or calcAddlTax > 0 or convTotalAmt > 0):
            netAmount = subTotal_slab - convDiscAmt + calcTotalGST_slab + calcAddlTax
            print("Net Amount",
                  netAmount,
                  "convTotalAmount",
                  convTotalAmt,
                  math.isclose(netAmount,
                               convTotalAmt,
                               abs_tol = 2.0))
            if not math.isclose(netAmount,
                                convTotalAmt,
                                abs_tol = 2.0):
                result.append(("totalAmount",
                              "Total Amount should be a sum of subtotal of slabs - Discount + GST of Slabs + Cess. A tolerance of Rs. 2.0 Rs is acceptable"))
        # 5 May 2023 
        ## 25 April 2023 Added validation for subtotal_0%
        # if (convTotalAmt != 0) and (stp_type == "DEFAULT"):
        #     netAmount = subTotal_slab - convDiscAmt + calcTotalGST_slab + calcAddlTax
        #     if not math.isclose(netAmount,
        #                         convTotalAmt,
        #                         abs_tol = 1.0):
        #         result.append(("totalAmount",
        #                        "Total Amount should be a sum of subtotal of slabs - Discount + GST of Slabs + Cess. A tolerance of 1.0 Rs is acceptable"))
        """
        if subTotal_slab == 0 and stp_type == "DEFAULT":
            result.append(("subTotal_0%",
                           "Sum of Sub Totals in slabs should be greater than 0"))
                           
        if not math.isclose(convSubTotal,subTotal_slab,abs_tol=0.5):
            result.append(("subTotal",
                          "Sub-Total Amount should be sum of individual Sub-Total slab Amounts.A tolerance of .5 Rs is acceptable"))

        if calcTotalGST > 0.0:
        # 12th sep changes for making vendor GSITN and PAN optional. 13-sep the CR revoke
            # if vendorGSTIN is not None and billingGSTIN is not None:
            #     print("vendorGSTIN",vendorGSTIN)
            #     print("billingGSTIN",billingGSTIN)
            #     print("CGST",CGSTAmount,
            #             "SGST",SGSTAmount,
            #             "IGST",IGSTAmount)
            #     print("condition",
            #             vendorGSTIN[:2].upper() == billingGSTIN[:2].upper(),
            #             ((convCGSTAmount == 0) or (convSGSTAmount == 0)))
            #     if (vendorGSTIN[:2].upper() == billingGSTIN[:2].upper()) and ((convSGSTAmount == 0) or (convCGSTAmount == 0)):
            #         if convCGSTAmount == 0:
            #             result.append(("CGSTAmount",
            #                             "CGST Amount cannot be blank or zero when total tax is not empty"))
            #         if convSGSTAmount == 0:
            #             result.append(("SGSTAmount",
            #                             "SGST Amount cannot be blank or zero when total tax is not empty"))
            #     elif (vendorGSTIN[:2].upper() != billingGSTIN[:2].upper()) and (convIGSTAmount == 0):
            #         if convIGSTAmount == 0:
            #             result.append(("IGSTAmount",
            #                             "IGST Amount cannot be blank or zero when total tax is not empty"))

            if not UI_validation and vendorGSTIN != "N/A":
                if vendorGSTIN is not None and billingGSTIN is not None:
                    print("vendorGSTIN",vendorGSTIN)
                    print("billingGSTIN",billingGSTIN)
                    print("CGST",CGSTAmount,
                          "SGST",SGSTAmount,
                          "IGST",IGSTAmount)
                    print("condition",
                          vendorGSTIN[:2].upper() == billingGSTIN[:2].upper(),
                          ((convCGSTAmount == 0) or (convSGSTAmount == 0)))
                    if (vendorGSTIN[:2].upper() == billingGSTIN[:2].upper()) and ((convSGSTAmount == 0) or (convCGSTAmount == 0)):
                        if convCGSTAmount == 0:
                            result.append(("CGSTAmount",
                                          "CGST Amount cannot be blank or zero when total tax is not empty"))
                        if convSGSTAmount == 0:
                            result.append(("SGSTAmount",
                                          "SGST Amount cannot be blank or zero when total tax is not empty"))
                    elif (vendorGSTIN[:2].upper() != billingGSTIN[:2].upper()) and (convIGSTAmount == 0):
                        if convIGSTAmount == 0:
                            result.append(("IGSTAmount",
                                          "IGST Amount cannot be blank or zero when total tax is not empty"))
        # 12th sep changes for making vendor GSITN and PAN optional. 13-sep the CR revoke

        """
        result = add_warning_messages(docMetatadata, docResult, result, UI_validation)
        
        return result, document_result_updated, docResult
    except:
        print("bizRuleValidateForUi",
              traceback.print_exc())
        return None, document_result_updated, docResult


def get_vendor_code_after_review(VENDOR_ADDRESS_MASTERDATA:pd.DataFrame, vgstin_corrected:str) -> tuple:
    """
        Derive Vendor code for a GSTIN and update docResult
    Args:
        VENDOR_ADDRESS_MASTERDATA (pd.DataFrame): Vendor Masterdata file
        vgstin_corrected (str): GSTIN for which vendor code is to be derived

    Returns:
        tuple -> Containing two values
            1.) str: Extracted vendor code (if multiple comma seprated values)
            2.) list: List of vendor codes
    """
    try:
        vendor_code = ""
        list_vendor_code = []
        VENDOR_ADDRESS_MASTERDATA['VENDOR_GSTIN'] = VENDOR_ADDRESS_MASTERDATA['VENDOR_GSTIN'].str.capitalize()
        try:
            vgstin_corrected = vgstin_corrected.capitalize()
        except:
            vgstin_corrected = ""
        vendor_master = VENDOR_ADDRESS_MASTERDATA[VENDOR_ADDRESS_MASTERDATA["VENDOR_GSTIN"] == vgstin_corrected]
        print("Masterdata shape:", vendor_master.shape)
        vendor_code = ",".join(list(map(str,vendor_master["VENDOR_ID"].unique())))
        list_vendor_code = list(map(str,vendor_master["VENDOR_ID"].unique()))
        return vendor_code, list_vendor_code
    except Exception as e:
        print("Exception occured in get_vendor_code_after_review", e)
        return vendor_code, list_vendor_code

def derive_vendor_code_business_logic(extractedFields:list, docResult:dict, UI_validation:bool, VENDOR_ADDRESS_MASTERDATA:pd.DataFrame, fldValCorrected:str) -> dict:
    """Update the result in document Result 

    Args:
        extractedFields (list): List of dictionary containing non-line Items
        docResult (dict): Document result from api
        UI_validation (bool) : If True we need to update correctedValue field instead of fieldValue
        VENDOR_ADDRESS_MASTERDATA (df) : Vendor Masterdata
        fldValCorrected (str) : field to be checked in masterdata
    Returns:
        dict: Updated document result
    """
    try:
        ## Derive the field vendor code since GSTIN has been modified
        print("Derive the field vendor code since GSTIN has been modified or it is called while extraction")
        vendor_code, list_vendor_code = get_vendor_code_after_review(VENDOR_ADDRESS_MASTERDATA, fldValCorrected)
        docResultCopy = copy.deepcopy(docResult)
        for idx,fld in enumerate(extractedFields):
            fldId = fld["fieldId"]
            if fldId == "vendorCode":
                if len(list_vendor_code) == 1:
                    print("Only 1 Vendor Code is present")
                    if UI_validation:
                        # editing the correctedValue field instead of fieldValue, if this has been passed through UI while validation check
                        print("Editing the correctedValue field instead of fieldValue, if this has been passed through UI while validation check")
                        if "correctedValue" in docResult["result"]["document"]["documentInfo"][idx]:
                            docResult["result"]["document"]["documentInfo"][idx]["previousValue"] = docResult["result"]["document"]["documentInfo"][idx]["correctedValue"]
                        else:
                            docResult["result"]["document"]["documentInfo"][idx]["previousValue"] = docResult["result"]["document"]["documentInfo"][idx]["fieldValue"]
                        docResult["result"]["document"]["documentInfo"][idx]["correctedValue"] = vendor_code
                    else:
                        docResult["result"]["document"]["documentInfo"][idx]["fieldValue"] = vendor_code
                    docResult["result"]["document"]["documentInfo"][idx]["confidence"] = 95
                    docResult["result"]["document"]["documentInfo"][idx]["dropDown"] = 1
                    docResult["result"]["document"]["documentInfo"][idx]["dropDownOptions"] = list_vendor_code
                elif len(list_vendor_code) > 1:
                    print("Multiple vendor codes are present")
                    list_vendor_code.append("N/A")
                    if UI_validation:
                        # editing the correctedValue field instead of fieldValue, if this has been passed through UI while validation check
                        print("Editing the correctedValue field instead of fieldValue, if this has been passed through UI while validation check")
                        if "correctedValue" in docResult["result"]["document"]["documentInfo"][idx]:
                            docResult["result"]["document"]["documentInfo"][idx]["previousValue"] = docResult["result"]["document"]["documentInfo"][idx]["correctedValue"]
                        else:
                            docResult["result"]["document"]["documentInfo"][idx]["previousValue"] = docResult["result"]["document"]["documentInfo"][idx]["fieldValue"]
                        docResult["result"]["document"]["documentInfo"][idx]["correctedValue"] = "N/A"
                    else:
                        docResult["result"]["document"]["documentInfo"][idx]["fieldValue"] = "N/A"
                    docResult["result"]["document"]["documentInfo"][idx]["dropDown"] = 1
                    docResult["result"]["document"]["documentInfo"][idx]["dropDownOptions"] = list_vendor_code
                elif len(list_vendor_code) == 0:
                    print("No Vendor Code is present")
                    if UI_validation:
                        # editing the correctedValue field instead of fieldValue, if this has been passed through UI while validation check
                        print("Editing the correctedValue field instead of fieldValue, if this has been passed through UI while validation check")
                        if "correctedValue" in docResult["result"]["document"]["documentInfo"][idx]:
                            docResult["result"]["document"]["documentInfo"][idx]["previousValue"] = docResult["result"]["document"]["documentInfo"][idx]["correctedValue"]
                        else:
                            docResult["result"]["document"]["documentInfo"][idx]["previousValue"] = docResult["result"]["document"]["documentInfo"][idx]["fieldValue"]
                        docResult["result"]["document"]["documentInfo"][idx]["correctedValue"] = "N/A"
                    else:
                        docResult["result"]["document"]["documentInfo"][idx]["fieldValue"] = "N/A"
                    docResult["result"]["document"]["documentInfo"][idx]["dropDown"] = 1
                    docResult["result"]["document"]["documentInfo"][idx]["dropDownOptions"] = ["N/A"]            
        return docResult
    except Exception as e:
        print("Exception occured in derive_vendor_code_business_logic", e)
        return  docResultCopy        

def derive_vendor_code(docMetatadata:dict, docResult:dict, result:list,documentId:str, VENDOR_ADDRESS_MASTERDATA:pd.DataFrame, UI_validation:bool) -> tuple:
    """
    Derive vendor code based on vendor GSTIN
    Args:
        docMetatadata (dict): Document metadata taken as input from api
        docResult (dict): Document result from api
        result (list): List of errors
        documentId (str): Document for which field is to be derived
        VENDOR_ADDRESS_MASTERDATA (DataFrame) : Vendor Masterdata file
        UI_validation(bool) : True if function is called from UI else False
    Returns:
        tuple -> Containing two values
            1.) bool(True/False): True if docResult need to be updated else False
            2.) dict: Modified Document result from api
    """
    docResultCopy = copy.deepcopy(docResult)
    try:
        derive_flag = False
        if docResult is None:
            return derive_flag, docResult
        extractedFields = docResult["result"]["document"]["documentInfo"]
        for idx,fld in enumerate(extractedFields):
            fldId = fld["fieldId"]
            if fldId == "vendorGSTIN":
                fldValCorrected = str(fld["correctedValue"] if "correctedValue" in fld.keys() else fld["fieldValue"]).strip()
                fldValPrevious = fld.get("previousValue")
                if ((fldValPrevious is not None) and (fldValCorrected != fldValPrevious)):
                    derive_flag = True
                    break
                else:
                    ## No need to Derive vendor Code since GSTIN is not modified
                    print("No need to Derive vendor Code since GSTIN is not modified")
        if derive_flag:
            docResult = derive_vendor_code_business_logic(extractedFields, docResult, UI_validation, VENDOR_ADDRESS_MASTERDATA, fldValCorrected)
            ## Reset the condition to derive vendor code
            print("removing previousValue from Vendor GSTIN")
            for idx,fld in enumerate(docResult["result"]["document"]["documentInfo"]):
                fldId = fld["fieldId"]
                if fldId == "vendorGSTIN":
                    docResult["result"]["document"]["documentInfo"][idx].pop("previousValue", None)
                    break    
        return derive_flag, docResult
    except Exception as e:
        print("Exception occured in derive_vendor_code", e)
        return False, docResultCopy
    
def checkVendorCode(docMetatadata:dict, docResult:dict, result:list, documentId:str) -> list:
    """
        Do Vendor code validations
    Args:
        docMetatadata (dict): Document metadata taken as input from api
        docResult (dict): Document result from api
        result (list): List of errors
        documentId (str): Document for which GSTIN's are to be verified

    Returns: 
         list -> List of errors if any
    """
    docResult_copy = copy.deepcopy(docResult)
    result_copy = copy.deepcopy(result)
    try:
        if docResult is None:
            print("DocResult is None. Returning False")
            return result
        if docMetatadata != None and docMetatadata.get("result")!= None and docMetatadata.get("result").get("document") != None and docMetatadata.get("result").get("document").get("docType") != None:
            extractedFields = docResult["result"]["document"]["documentInfo"]
            vendorCode = None
            for idx,fld in enumerate(extractedFields):
                fldId = fld["fieldId"]
                if fldId == "vendorCode":
                    vendorCode = fld["correctedValue"] if "correctedValue" in fld.keys() else fld["fieldValue"]
                    vendorCodeDropDown  = fld["dropDownOptions"]
            
            ## Checking the conditions based on derived VendorCode
            if vendorCode != None:
                # 1. Single vendor code is present
                if vendorCode != "N/A" and len(vendorCodeDropDown) == 1:
                    print("Single vendor code is present, no errors")
                # 2. Multiple vendor codes are present
                if vendorCode == "N/A" and len(vendorCodeDropDown) > 1:
                    if (docMetatadata.get("result").get("document").get("status") == "REVIEW"):
                        ## Re-assign the document logic comes here
                        print("Moving to re-assign tab since Multiple vendor code are present.")
                        result.append(("vendorCode", "REASSIGN : Multiple vendor code are present"))
                    else:
                        ## Field can't be left N/A while reviewing by Pierian team
                        print("Vendor code can't be left empty")
                        result.append(("vendorCode", "No vendor code is selected"))
                if vendorCode != "N/A" and len(vendorCodeDropDown) > 1:
                    if (docMetatadata.get("result").get("document").get("status") == "REVIEW"):
                        ## TAO team can't select from multiple options
                        print("TAO team can't select from multiple options")
                        result.append(("vendorCode", "Muliple vendor code found. Select N/A from list"))
                    else:
                        ## Pierian team selected the correct option, no errors
                        print("Pierian team selected the correct option, no errors")
                # 3. No vendor code is present
                if vendorCode == "N/A" and len(vendorCodeDropDown) == 1 and (vendorCodeDropDown[0] == "N/A"):
                    print("No vendor code is present for a GSTIN, can be marked as review completed. No validations needed")
            else:
                print("vendor code can't be None. Pleach check with support team")
                result.append(("vendorGSTIN", "vendor code can't be None. Pleach check with support team"))
        return result
    except Exception as e:
        print("Exception occured in checkVendorCode", e)
        return result_copy
    
def checkingGSTINDetails(docMetadata:dict, docResult:dict, result:list, documentId:str, fld: dict) ->list:
    """
    Checking Billing and Shipping GSTIN's for re-assigning the documents based on current status
    Args:
        docMetadata (dict): Document metadata taken as input from api
        docResult (dict): Document result from api
        result (list): List of errors
        documentId (str): Document for which GSTIN's are to be verified
        fld (dict): Dictionary object of field which contains field data
    Returns:
        list -> List of errors along with REASSIGN reason if any
    """
    result_copy = copy.deepcopy(result)
    try:
        if (docMetadata == None) or (docMetadata.get("result") == None) or (docMetadata.get("result").get("document") == None) or (docMetadata.get("result").get("document").get("status") == None):
            print("Status for the document is not present in metadata, returning False.")
            return result
        fldId = fld["fieldId"]
        fldVal = str(fld["correctedValue"]).strip() if "correctedValue" in fld.keys() else str(fld["fieldValue"]).strip()
        if fldId == "billingGSTIN" and fldVal == "N/A":
            if (docMetadata.get("result").get("document").get("status") == "REVIEW"):
                ## Re-assign the document logic comes here
                print("Moving to re-assign tab since Billing GSTIN is N/A")
                result.append(("billingGSTIN", "REASSIGN : Billing GSTIN is empty"))
            else:
                ## Field can't be left N/A while reviewing
                print("Billing GSTIN can't be N/A")
                result.append(("billingGSTIN", "Billing GSTIN can't be N/A"))
        if fldId == "shippingGSTIN" and fldVal == "N/A":
            if (docMetadata.get("result").get("document").get("status") == "REVIEW"):
                ## Re-assign the document logic comes here
                print("Moving to re-assign tab since Shipping GSTIN is N/A")
                result.append(("shippingGSTIN", "REASSIGN : Shipping GSTIN is empty"))
            else:
                ## Field can't be left N/A while reviewing
                print("Shipping GSTIN can't be N/A")
                result.append(("shippingGSTIN", "Shipping GSTIN can't be N/A"))
        return result
    except Exception as e:
        print("Exception occured in checkingGSTINDetails", e)
        return result_copy
        
def derive_custom_fields(docMetatadata:dict, docResult:dict, result:list,documentId:str, VENDOR_ADDRESS_MASTERDATA:pd.DataFrame, UI_validation:bool) -> tuple:
    """
    Derive custom fields based on some field extracted
    Args:
        docMetatadata (dict): Document metadata taken as input from api
        docResult (dict): Document result from api
        result (list): List of errors
        documentId (str): Document for which field is to be derived
        VENDOR_ADDRESS_MASTERDATA (DataFrame) : Vendor Masterdata file
        UI_validation(bool) : True if function is called from UI else False
    Returns:
        tuple -> Containing three values
            1.) bool(True/False): True if docResult need to be updated else False
            2.) dict: Modified Document result from api
            3.) list -> List of errors if any
    """
    docResult_copy = copy.deepcopy(docResult)
    result_copy = copy.deepcopy(result)
    try:
        status_derived_field = False
        status_derived_field_after, docResult = derive_vendor_code(docMetatadata, docResult, result, documentId, VENDOR_ADDRESS_MASTERDATA, UI_validation)
        if status_derived_field_after:
            status_derived_field = True
        return status_derived_field, docResult, result
    except Exception as e:
        print("Exception occured in derive_custom_fields", e)
        return False, docResult_copy, result_copy

def add_warning_invoiceNumber(docMetatadata:dict, docResult:dict, result:list) -> list:
    """
    Add warning flags for Invoice Number
    Args:
        docMetatadata (dict): Document metadata taken as input from api
        docResult (dict): Document result from api
        result (list): List of errors
        
    Returns:
        list: List of errors with warning messages
    """
    result_copy = copy.deepcopy(result)
    try:
        if docResult is None:
            return result
        if docMetatadata != None and docMetatadata.get("result")!= None and docMetatadata.get("result").get("document") != None and docMetatadata.get("result").get("document").get("docType") != None:
            extractedFields = docResult["result"]["document"]["documentInfo"]
            rpaInvoiceNumber = None
            invoiceNumber = None
            for idx,fld in enumerate(extractedFields):
                fldId = fld["fieldId"]
                if fldId == "rpaInvoiceNumber":
                    rpaInvoiceNumber = fld["correctedValue"] if "correctedValue" in fld.keys() else fld["fieldValue"]
                if fldId == "invoiceNumber":
                    invoiceNumber = fld["correctedValue"] if "correctedValue" in fld.keys() else fld["fieldValue"]
            if rpaInvoiceNumber != None and invoiceNumber != None:
                if str(rpaInvoiceNumber).strip().lower() != str(invoiceNumber).strip().lower():
                    result.append(("invoiceNumber",f"Invoice Number {str(invoiceNumber).strip().lower()} and RPA Invoice Number {str(rpaInvoiceNumber).strip().lower()} mis-match.", 1))
        return result           
    except Exception as e:
        print("Exception occured in add_warning_invoiceNumber", e)
        return result_copy 
def validate_date_blinkit(date_str):
    try:
        # Convert the input date string to a datetime object
        input_date = datetime.strptime(date_str, '%d/%m/%Y')
        # Get the current date
        current_date = datetime.now()
        # Check if the input date is in the future
        if input_date > current_date:
            print("future")
            return False
        # Check if the input date is less than 120 days from the present
        if ((current_date - input_date).days) > 120:
            print("old")
            return False
        
        # If the date is neither in the future nor less than 120 days from present, return True
        return True

    except ValueError:
        # Handle invalid date format
        print("Invalid date format. Please use dd/mm/yyyy format.")
        return False
def sanitize_invoice_date_text(prediction):
    prediction_copy = prediction.copy()
    try:
        #here we are making inv date as blank because the date is future or older
        invdat  = prediction['invoiceDate']['text']
        print('sanitize_invoice_date_text',invdat)
        if not validate_date_blinkit(invdat):
            prediction['invoiceDate']['text']=''
            return prediction
        return prediction
    except Exception as e:
        print('exemption in sanitize_invoice_date_text',e)
        return prediction_copy
def calculate_date_difference(grndate:str,invdate:str)-> int:
    """
    Function to find the difference b/w the days

    Args:
        grndate (str): A string value passed by RPA Bot which is actually a GRN Date.
        invdate (str): A string value which equals Invoice date extracted. 

    Returns:
        int: Day differnce b/w Invoice and GRN date.
    """
    try:
        from datetime import datetime
        # Parse the input dates
        parsed_grn_date = datetime.strptime(grndate, "%Y-%m-%d")
        parsed_invoice_date = datetime.strptime(invdate, "%d/%m/%Y")

        # Calculate the difference
        date_difference = parsed_grn_date - parsed_invoice_date

        return date_difference.days
    except Exception as e:
        print("Exception occured in calculate_date_difference", e)
        return -5
    
def add_warning_invoiceDate(docMetatadata:dict, docResult:dict, result:list) -> list:
    """
    Add warning flags for Invoice Date
    Args:
        docMetatadata (dict): Document metadata taken as input from api
        docResult (dict): Document result from api
        result (list): List of errors
        
    Returns:
        list: List of errors with warning messages
    """
    result_copy = copy.deepcopy(result)
    try:
        if docResult is None:
            return result
        if docMetatadata != None and docMetatadata.get("result")!= None and docMetatadata.get("result").get("document") != None and docMetatadata.get("result").get("document").get("docType") != None and docMetatadata.get("result").get("document").get("docType") == "Invoice":
            extractedFields = docResult["result"]["document"]["documentInfo"]
            invoiceDate = None
            grnDate = docMetatadata.get("result").get("document").get("grnDate")
            for idx,fld in enumerate(extractedFields):
                fldId = fld["fieldId"]
                if fldId == "invoiceDate":
                    invoiceDate = fld["correctedValue"] if "correctedValue" in fld.keys() else fld["fieldValue"]
            if grnDate != None and invoiceDate != None:
                diff = calculate_date_difference(grnDate,invoiceDate)
                if diff not in [0,1]:
                    result.append(("invoiceDate",f"Please re-check Invoice Date.", 1))
            
        return result           
    except Exception as e:
        print("Exception occured in add_warning_invoiceNumber", e)
        return result_copy 
    
def add_warning_messages(docMetatadata:dict, docResult:dict, result:list, UI_validation:bool) -> list:
    """
    Function used to add warning flag for fields
    Args:
        docMetatadata (dict): Document metadata taken as input from api
        docResult (dict): Document result from api
        result (list): List of errors
        UI_validation(bool) : Flag set to True if it is called from UI

    Returns:
        list: List of errors with warning messages
    """
    result_copy = copy.deepcopy(result)
    try:
        print("Result before Invoice Number warning flag is:", result)
        result = add_warning_invoiceNumber(docMetatadata, docResult, result)
        print("Result after Invoice Number warning flag is:", result)
        if UI_validation:
            result = add_warning_invoiceDate(docMetatadata, docResult, result)
            print("Result after Invoice Date warning flag is:", result)
        return result
    except Exception as e:
        print("Exception Occured in add_warning_messages", e)
        return result_copy


def isMasterDataPresent(documentId, callbackUrl):

    try:
        print("Inside isMsaterdataPresent")
        docResult = putil.getDocumentResultApi(documentId, callbackUrl)
        #print("masterdata present docresult:",docResult)
        if docResult is not None:
            docInfo = docResult["result"]["document"]["documentInfo"]
            for docInfo_ in docInfo:
                if docInfo_["fieldId"] in ["vendorGSTIN",
                                           "billingGSTIN",
                                           "shippingGSTIN"]:
                    masterData = docInfo_["entityMasterdata"]
                    if not masterData:
                        return False
            
            return True
        else:
            return False
    except:
        print("getDocumentMasterDataPresent",
              traceback.print_exc())
        return False

def isInvoiceNumberAnAmount(documentId, callbackUrl):

    try:
        docResult = putil.getDocumentResultApi(documentId,
                                               callbackUrl)
        print("docResult isInvoiceNumberAnAmount ",docResult)
        if docResult is not None:
            docInfo = docResult["result"]["document"]["documentInfo"]
            for docInfo_ in docInfo:
                if docInfo_["fieldId"] in ["invoiceNumber"]:
                    if putil.validAmount(docInfo_["fieldValue"]) or putil.validDate(docInfo_["fieldValue"]):
                        print("isInvoiceNumberAnAmount -> validAmount : ", True)
                        return True
            # print("isInvoiceNumberAnAmount : ", False)       
            return False
        else:
            print("isInvoiceNumberAnAmount-> getDocumentResultApi is None : ", False)
            return False
    except:
        print("isInvoiceNumberAnAmount",
              traceback.print_exc())
        print("isInvoiceNumberAnAmount exception : ", False)
        return True

def isTotOrSubCalc(documentId, callbackUrl):

    try:
        docResult = putil.getDocumentResultApi(documentId,
                                                callbackUrl)
        if docResult is not None:
            docInfo = docResult["result"]["document"]["documentInfo"]
            for docInfo_ in docInfo:
                if docInfo_["fieldId"] in ["totalAmount","subTotal"]:
                    if docInfo_.get("calculated_field"):
                        if docInfo_.get("calculated_field") == 1:
                            return True
            return False
        else:
            return False
    except:
        print("isTotOrSubCalc",
              traceback.print_exc())
        return True

def is_equal_subTotal_TotalAmount(documentId, callbackUrl):

    try:
        docResult = putil.getDocumentResultApi(documentId,
                                                callbackUrl)
        total = None
        subtotal = None
        if docResult is not None:
            docInfo = docResult["result"]["document"]["documentInfo"]
            for docInfo_ in docInfo:
                #print(docInfo_["fieldId"])
                if docInfo_["fieldId"]=="totalAmount":
                    if docInfo_.get("fieldValue"):
                        total = float(docInfo_.get("fieldValue"))
                if docInfo_["fieldId"] == "subTotal":
                    if docInfo_.get("fieldValue"):
                        subtotal = float(docInfo_.get("fieldValue"))
                if (total is not None) and (subtotal is not None):
                    print("total :",total,"subtotal :",subtotal,"abs :",abs(total-subtotal))
                    if (abs(total-subtotal)) < 1:
                        return False
        return None 
    except:
        print("isTotOrSubCalc",
              traceback.print_exc())
        return None

def check_field_stp_score(documentId, callbackUrl):

    try:
        docResult = putil.getDocumentResultApi(documentId,
                                                callbackUrl)
        if docResult is not None:
            docInfo = docResult["result"]["document"]["documentInfo"]
            for docInfo_ in docInfo:
                #print(docInfo_["fieldId"])
                stp_score_field = ["invoiceDate","invoiceNumber","totalAmount"]
                if docInfo_["fieldId"] in stp_score_field:
                    print(f"checking {docInfo_['fieldId']} stp confidence",docInfo_.get("confidence"))
                    if docInfo_.get("confidence") < 70:
                        print(f"{docInfo_['fieldId']} confidance not meet the stp score")
                        return False
        return True
    except:
        print("check_field_stp_score exception",
              traceback.print_exc())
        return False

def check_multi_invoices_stp(documentId, callbackUrl):

    try:
        docResult = putil.getDocumentResultApi(documentId,
                                                callbackUrl)
        if docResult is not None:
            docInfo = docResult["result"]["document"]["documentInfo"]
            for docInfo_ in docInfo:
                #print(docInfo_["fieldId"])
                if docInfo_["fieldId"]=="invoiceNumber":
                    print("checking multiple invoiceNumber flag :",docInfo_.get("multi_invoices"))
                    if docInfo_.get("multi_invoices") == 1:
                        print("multiple invoices present")
                        return False
                ## 25 Nov throttling stp if amount is greater than 200000l
                if docInfo_["fieldId"]=="totalAmount":
                    try: 
                        tot = float(docInfo_.get("fieldValue"))
                        if tot>1000000:
                            print("Total amount is greater than 10lk so pushing to manual queue")
                            return False
                    except:
                        print("check stp throtlle on total amount exception")
                        return False
                ## 25 Nov throttling stp if amount is greater than 200000l

        return True
    except:
        print("Multi invoices check exception",
              traceback.print_exc())
        return False

def getBuyerStateCode(documentId, callbackUrl):

    try:
        docResult = putil.getDocumentResultApi(documentId,
                                                callbackUrl)
        if docResult is not None:
            docInfo = docResult["result"]["document"]["documentInfo"]
            for docInfo_ in docInfo:
                if docInfo_["fieldId"]=="billingGSTIN":
                    billingGSTIN = docInfo_["fieldValue"]
                    if billingGSTIN:
                        if len(billingGSTIN) > 2:
                            billingStateCode = billingGSTIN[:2]
                            return billingStateCode
        return None
    except:
        print("check_date_field_stp_score exception",
              traceback.print_exc())
        return None

def makeDefaultPredValToNA(prediction):
    try:
        for k,v in prediction.items():
            if (k in ["vendorGSTIN","vendorPAN","billingGSTIN", "shippingGSTIN"]) and v :
                # print(k)
                if v["text"] =='':
                    v["text"] = "N/A"
                    prediction[k]= v
        return prediction 
    except:
        return prediction


#  matchingReferenceData():
import math
import traceback

# class matchingReferenceData():

def readReferenceData():
    try:
        REFERENCE_MASTER_DATA = cfg.getReferenceMasterData()
        ref_Data = pd.read_csv(REFERENCE_MASTER_DATA, encoding='unicode_escape')
        return ref_Data
    except:
        print("readReferenceData reading exception",traceback.print_exc())
        return None

def getBoundingBoxes(prediction:dict,fieldName:str)->dict:
    try:
        left = math.floor(prediction[fieldName]['left'] * prediction[fieldName]['image_widht'])
        right = math.ceil(prediction[fieldName]['right'] * prediction[fieldName]['image_widht'])
        top = math.floor(prediction[fieldName]['top'] * prediction[fieldName]['image_height'])
        bottom = math.ceil(prediction[fieldName]['bottom'] * prediction[fieldName]['image_height'])
        return {"left":left,"right":right,"top":top,"bottom":bottom}
    except:
        print("getBoundingBoxes exception",traceback.print_exc())
        return None

def getWordshape(text):
    import re
    try:
        t1 = re.sub('[A-Z]', 'X',text)
        t2 = re.sub('[a-z]', 'x', t1)
        return re.sub('[0-9]', 'd', t2)
    except:
        print("getWordshape exception.",traceback.print_exc())
        return None

def matchInvoiceNumber(ref_data,prediction:dict)->dict:
    prediction['invoiceNumber']['reference_data']=0
    try:
        if (ref_data.shape[0])>0:
            vendor_inv_ref_data = ref_data[ref_data['field_name']=='invoiceNumber']
            # vendor_inv_ref_data = vendor_inv_ref_data[vendor_inv_ref_data['status']==1]
            inv_Cordinates = getBoundingBoxes(prediction,"invoiceNumber")
            if inv_Cordinates:
                inv_left = inv_Cordinates["left"]
                inv_right = inv_Cordinates["right"]
                inv_top = inv_Cordinates["top"]
                inv_bottom = inv_Cordinates["bottom"]
                final_confidence = 0
                for index,row in vendor_inv_ref_data.iterrows():
                    if ((inv_left >= float(row['right']))|(inv_top >= float(row['bottom']))|
                        (inv_right <= float(row['left']))|(inv_bottom <=float(row["top"]))):
                        print(" Inv ref data cordinates not ovelapping")
                        if getWordshape(prediction['invoiceNumber']['text']) != row['field_shape']:
                            # prediction['invoiceNumber']['final_confidence_score']=0.4
                            if final_confidence < 0.4:
                                final_confidence = 0.4
                        elif getWordshape(prediction['invoiceNumber']['text']) == row['field_shape']:
                            if (('X' in getWordshape(prediction['invoiceNumber']['text'])) & 
                                ('d' in getWordshape(prediction['invoiceNumber']['text'])) & 
                                (len(prediction['invoiceNumber']['text'])>=5)):
                                # prediction['invoiceNumber']['final_confidence_score']=0.8
                                if final_confidence < 0.9:
                                    final_confidence = 0.9
                            else:
                                prediction['invoiceNumber']['final_confidence_score']=0.6
                                if final_confidence < 0.6:
                                    final_confidence = 0.6
                    elif getWordshape(prediction['invoiceNumber']['text']) == row['field_shape']:
                        prediction['invoiceNumber']['wordshape_confidence']=1
                        prediction['invoiceNumber']['final_confidence_score']=1
                        prediction['invoiceNumber']['reference_data']=1
                        print("matchInvoiceNumber Cordinates overlapping 1")
                        return prediction
                    elif ( (len(prediction['invoiceNumber']['text'])>=5)) :
                        if (('X' in getWordshape(prediction['invoiceNumber']['text'])) & ('d' in getWordshape(prediction['invoiceNumber']['text']))):
                            prediction['invoiceNumber']['final_confidence_score']=0.95
                            prediction['invoiceNumber']['reference_data']=1
                        else:
                            prediction['invoiceNumber']['final_confidence_score']=0.93
                            prediction['invoiceNumber']['reference_data']=1
                        return prediction
                print("final_confidence :",final_confidence)
                if final_confidence:
                    prediction['invoiceNumber']['final_confidence_score']= final_confidence
                    prediction['invoiceNumber']['reference_data']=-1
                return prediction
        else:
            prediction['invoiceNumber']['reference_data']=0
            print("there is no reference data")
            return prediction
    except:
        print("matchInvoiceNumber exception.",traceback.print_exc())
        return prediction
def matchInvoiceDate(ref_data,prediction:dict)->dict:
    prediction['invoiceDate']['reference_data']=0
    try:
        if (ref_data.shape[0])>0:
            vendor_dt_ref_data = ref_data[ref_data['field_name']=='invoiceDate']
            vendor_dt_ref_data = vendor_dt_ref_data[vendor_dt_ref_data['status']==1]
            inv_Cordinates = getBoundingBoxes(prediction,"invoiceDate")
            if inv_Cordinates:
                inv_left = inv_Cordinates["left"]
                inv_right = inv_Cordinates["right"]
                inv_top = inv_Cordinates["top"]
                inv_bottom = inv_Cordinates["bottom"]
                final_confidence = 0
                for index,row in vendor_dt_ref_data.iterrows():
                    if ((inv_left >= float(row['right']))|(inv_top >= float(row['bottom']))|
                        (inv_right <= float(row['left']))|(inv_bottom <=float(row["top"]))):
                        if final_confidence < 0.4:
                            final_confidence = 0.4
                    elif getWordshape(prediction['invoiceDate']['text']) == row['field_shape']:
                        prediction['invoiceDate']['wordshape_confidence']=1
                        prediction['invoiceDate']['final_confidence_score']=1
                        prediction['invoiceDate']['reference_data']=1
                        print('invoiceDate cordintes overlapping')
                        return prediction
                    else:
                        #prediction['invoiceDate']['wordshape_confidence']=0.75
                        prediction['invoiceDate']['final_confidence_score']=0.65
                        prediction['invoiceDate']['reference_data']=-1
                        return prediction
                print("inv date final conf ",final_confidence)
                if final_confidence:
                    prediction['invoiceDate']['final_confidence_score']= final_confidence
                    prediction['invoiceDate']['reference_data']=-1
                return prediction
        else:
            prediction['invoiceDate']['reference_data']=0
            print("there is no invoiceDate reference data")
        return prediction
    except:
        print('there is a exeption',traceback.print_exc())
        return prediction

def matchReferenceData(prediction:dict)->dict:
    print("Matching Reference Data")
    ref_data = readReferenceData()
    vendorGstin = prediction.get("vendorGSTIN").get("text")
    if (vendorGstin) and (vendorGstin != ''):
        ref_data = ref_data[ref_data["vendor_id"]==vendorGstin]
        prediction = matchInvoiceNumber(ref_data,prediction)
        prediction = matchInvoiceDate(ref_data,prediction)
        print("invoiceDate :",prediction.get("invoiceDate"))
        print("invoiceNumber :",prediction.get("invoiceNumber"))
    else:
        print("vendorGSTIN not extracted")
    return prediction
@putil.timing
def add_reference_data_flag(prediction:dict,ref_data):
    '''
    flags added for : invoceNumner, invoiceDate
    isReferenceDataPresent = 1 -> ref. data is preset for the vendor GSTIN
    isReferenceDataPresent = 0 - > ref data is not present for the vendor GSTIN
    isReferenceDataPresent = -1 -> in case there is no gstin(N?A) for Vendors
    isNewLayout = 1 -> bounding box is not overlapping with existing ref data for the vendor GSTIN
    isNewLayout = 0 - > bounding box match present in the ref data for the vendor GSTIN
    isNewLayout = -1 -> in case there is no gstin(N?A) for Vendors

    '''
    print("Adding Ref.  date flags")
    field_list = ["invoiceNumber","invoiceDate"]
    def update_flag(field:list,prediction:dict,key:str,flag:int)->dict:
        try:
            for f in field:
                if (prediction.get(f)):
                    prediction[f][key]=flag
            return prediction
        except:
            print(traceback.print_exc())
            return prediction
    try:
        vendorGstin = prediction.get("vendorGSTIN").get("text")
        if (vendorGstin) and ((vendorGstin != '') and (vendorGstin != 'N/A')):
            filt_ref_data = ref_data[ref_data["vendor_id"]==vendorGstin]
            if (len(filt_ref_data)>0):
                prediction  = update_flag(field_list,prediction,"isReferenceDataPresent",1)
                for f in field_list:
                    # Default bb is new = True (1)
                    prediction  = update_flag([f],prediction,"isNewLayout",1)                       
                    bb = getBoundingBoxes(prediction,f)
                    if bb:
                        left,top,right,bottom = bb["left"],bb["top"],bb["right"],bb["bottom"]
                        for row in filt_ref_data.itertuples():
                            if not((left >= float(row.right))|(top >= float(row.bottom))|
                                (right <= float(row.left))|(bottom <=float(row.top))):
                                prediction  = update_flag([f],prediction,"isNewLayout",0)    
            else:
                prediction  = update_flag(field_list,prediction,"isReferenceDataPresent",0) 
                prediction  = update_flag(field_list,prediction,"isNewLayout",0) 
        else:
            prediction  = update_flag(field_list,prediction,"isReferenceDataPresent",-1)
            prediction  = update_flag(field_list,prediction,"isNewLayout",-1)
        return prediction
    except:
        print(traceback.print_exc())    
        prediction  = update_flag(field_list,prediction,"isReferenceDataPresent",-1)
        prediction  = update_flag(field_list,prediction,"isNewLayout",-1)
        return prediction
         

def check_pattern(text, pattern ='[0-9]{4}'):
    match = re.search(pattern, text)
    #print("match :",match)
    if match:
        return True
    else:
        return False

def extraact_pattern(text, pattern ='[0-9]{13}'):
    return re.findall(pattern, text)

def validate_with_extracted_po(text,po_num):
    return re.findall(text,po_num)

def get_po_token(df,pattern:str):
    try:
        df["is_po"] = df["text"].apply(lambda x: check_pattern(x, pattern))
        fdf = df.loc[df["is_po"]==True]
        print(fdf.shape)
        if len(fdf) > 0:
            fdf = fdf.reset_index(drop = True) 
            print("s",fdf["token_id"][0])
            return fdf["token_id"][0]
        else:
            return
    except:
        print("find po_pattern exception :",traceback.print_exc())
        return 
@putil.timing
def checklenpoNumner(final_prediction):
    final_prediction_copy = copy.deepcopy(final_prediction)
    try:
        if final_prediction.get("poNumber")!=None and final_prediction.get("poNumber").get("text")!= None:
            if len(extraact_pattern(str(final_prediction.get("poNumber").get("text"))))==0:
                final_prediction["poNumber"]["final_confidence_score"] = 0.4            
        return final_prediction
    except:
        print('exception in po Number', traceback.print_exc())
        return final_prediction_copy
@putil.timing
def validate_poNumber(df : pd.DataFrame, prediction : dict ,metadata_po:str,docMetaData) -> dict:
    try:
        if docMetaData.get("result")!= None and docMetaData.get("result").get("document") != None and docMetaData.get("result").get("document").get("docType") == "Discrepancy Note":
            print("No validation of PoNumber for discr Note")
            return prediction
        print('Meta PO num:',metadata_po)
        if len(extraact_pattern(str(metadata_po)))>0:
            po = extraact_pattern(str(metadata_po))
            po = po[0]
            print('Meta after removing punctuation:',po)
        else :
            print("Meta date poNum is not 13 digit")
            if (metadata_po == '') or (metadata_po is None):
                prediction['poNumber']['final_confidence_score'] = 0.8
                return prediction
            prediction.update(add_new_field(field_name="poNumber",value=''))
            return prediction
        
        if prediction['poNumber']['text'] == po:
            print("Exact match for poNumber")
            prediction['poNumber']['final_confidence_score'] = 1.0

        else :
            token = get_po_token(df,po)
            if token:
                token_df = df[df["token_id"] == token]
                print("token_df :",token_df.shape)
                for _, row in token_df.iterrows():
                    prediction.update(add_new_fields(field_name="poNumber",row=row))
                    ## 9 August 2023 Added to seprate date from po Number
                    prediction['poNumber']['text'] = po
                    print("updated poNum using meta po")
            else:
                prediction.update(add_new_field(field_name="poNumber",value=''))
        return prediction
    except:
        print('exception in po Number', traceback.print_exc())
        return prediction
# 21 May 2023 Added invoice Number from metadata    
def check_df_for_invNumber(x,metadata_inv):
    if str(metadata_inv).strip().lower() == str(x).lower():
        return True
    return False
    
@putil.timing
def extract_vendor_specific_fields(final_prediction, DF,flag_invoice_number):
    final_prediction_copy = copy.deepcopy(final_prediction)
    flag_invoice = flag_invoice_number
    try:
        ## 28 August 2023, Extract Invoice Number for ITC Limited
        PAN_LIST_ITC_LIMITED = ["AAACI5950L"]
        if (final_prediction.get("vendorGSTIN")!= None) and (final_prediction.get("vendorGSTIN").get("text") != None):
            if str(final_prediction.get("vendorGSTIN").get("text"))[2:12] in PAN_LIST_ITC_LIMITED:
                flag_invoice = False
                print("gstin present")
                keywords_invoice_number = ["invoice no", "gst serial no", "sl. no."]
                keyword_pattern = "|".join(keywords_invoice_number)
                DF["DF_word_left"] = (DF["W4Lf"].astype(str)) + " " + (DF["W3Lf"].astype(str)) + " " + (DF["W2Lf"].astype(str)) + " " + (DF["W1Lf"].astype(str))
                # DF.to_csv(r"C:\Users\Admin\Desktop\test2.csv")
                filt = DF[DF["DF_word_left"].str.contains(keyword_pattern, case=False)]
                if filt.shape[0]>0:
                    for index, row in filt.iterrows():
                        if (row["text"].isdigit() or row["text"].isalnum()) and not row["text"].isalpha():
                            print("sahil", row["text"])
                            # final_prediction.update(add_new_field(field_name="invoiceNumber",value=row["text"]))
                            final_prediction.update(add_new_fields("invoiceNumber",row,from_entity=True))
                            # final_prediction["invoiceNumber"]["text"] = row["text"]
                            # final_prediction["invoiceNumber"]["final_confidence_score"] = 1
                            break
                # keywords_invoice_date = ["date & time", "gst doc date", "date"]
                # keyword_pattern = "|".join(keywords_invoice_date)
                
        ## 13 November 2023, Extract Invoice Number for Britania
        PAN_LIST_BRITANIA = ['AABCB8795R','AABCB2066P']
        if (final_prediction.get("vendorGSTIN")!= None) and (final_prediction.get("vendorGSTIN").get("text") != None):
            if str(final_prediction.get("vendorGSTIN").get("text"))[2:12] in PAN_LIST_BRITANIA:
                flag_invoice = False
                print("gstin present")
                keywords_invoice_number = ["gst invoice no", "gst doc no"]
                keyword_pattern = "|".join(keywords_invoice_number)
                DF["DF_word_left"] = (DF["W4Lf"].astype(str)) + " " + (DF["W3Lf"].astype(str)) + " " + (DF["W2Lf"].astype(str)) + " " + (DF["W1Lf"].astype(str))
                # DF.to_csv(r"C:\Users\Admin\Desktop\test2.csv")
                filt = DF[DF["DF_word_left"].str.contains(keyword_pattern, case=False)]
                if filt.shape[0]>0:
                    for index, row in filt.iterrows():
                        if ((row["text"].isalnum())) and (not row["text"].isdigit()) and (not row["text"].isalpha()):
                            print("sahil", row["text"])
                            # final_prediction.update(add_new_field(field_name="invoiceNumber",value=row["text"]))
                            final_prediction.update(add_new_fields("invoiceNumber",row,from_entity=True))
                            # final_prediction["invoiceNumber"]["text"] = row["text"]
                            # final_prediction["invoiceNumber"]["final_confidence_score"] = 1
                            break
        return final_prediction, flag_invoice
    except Exception as e:
        print('exception in extract_vendor_specific_fields', traceback.print_exc())
        return final_prediction_copy,flag_invoice
    

def get_inv_token(df,metadata_inv:str):
    try:
        df["is_invoiceNumber"] = df["text"].apply(lambda x: check_df_for_invNumber(x, metadata_inv))
        fdf = df[df["is_invoiceNumber"]==True]
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
@putil.timing
def validate_invNumber(df : pd.DataFrame, prediction : dict ,metadata_inv:str,docMetaData : dict) -> dict:
    flag_invoice = False
    try:
        if docMetaData!= None and docMetaData.get("result").get("document").get("docType") == "Discrepancy Note":
            print("Taking meta for Invoice Number of discr note")
            extracted_inv_num = ""
            if prediction.get("invoiceNumber") != None and prediction.get("invoiceNumber").get("text") != None:
                extracted_inv_num = prediction["invoiceNumber"]["text"]
            if (docMetaData.get("result").get("document").get("invNumber") != None) and (docMetaData.get("result").get("document").get("invNumber") != ""):
                print("Taking metadata Inv number passed from bot")
                metadata_inv_bot = docMetaData.get("result").get("document").get("invNumber")
            if (docMetaData.get("result").get("document").get("linked_document") != None) and (docMetaData.get("result").get("document").get("linked_document").get("invoiceNumber") != None) and (docMetaData.get("result").get("document").get("linked_document").get("invoiceNumber") != ""):
                print("Taking metadata Inv number passed from Invoice")
                metadata_invoice = docMetaData.get("result").get("document").get("linked_document").get("invoiceNumber") 
            print(f"Invoice number from Invoice is {metadata_invoice}, from RPA is {metadata_inv_bot}, extracted from document is {extracted_inv_num}")
            
            if (metadata_invoice == None) or (metadata_invoice == "") or (metadata_inv_bot == None) or (metadata_inv_bot == ""):
                return prediction,flag_invoice
            ## Checking for conditions to increase the confidence
            if str(metadata_invoice).strip().lower() == str(metadata_inv_bot).strip().lower() and str(metadata_invoice).strip().lower() == str(extracted_inv_num).strip().lower():
                flag_invoice = True
        elif docMetaData!= None and docMetaData.get("result").get("document").get("docType") == "Invoice":
            print('Meta Inv num:',metadata_inv)
            extracted_inv_num = ""
            if prediction.get("invoiceNumber") != None and prediction.get("invoiceNumber").get("text") != None:
                extracted_inv_num = prediction["invoiceNumber"]["text"]  
                print("Extracted Invoice Number", extracted_inv_num)
            if (metadata_inv == None) or (metadata_inv == ""):
                return prediction,flag_invoice   
            if str(metadata_inv).strip().lower() == str(extracted_inv_num).strip().lower():
                flag_invoice = True  
            # token = get_inv_token(df,metadata_inv)
            # if token:
            #     token_df = df[df["token_id"] == token]
            #     print("token_df :",token_df.shape)
            #     for _, row in token_df.iterrows():
            #         # prediction.update(add_new_fields("invoiceNumber",row,from_entity=True))
            #         # prediction["invoiceNumber"]["text"] = str(metadata_inv).strip()
            #         # print("updated Invoice Number using meta data")
            #         flag_invoice = True
        return prediction,flag_invoice
    except:
        print('exception in validate_invNumber', traceback.print_exc())
        return prediction,flag_invoice
# 21 May 2023 Added Vendor GSTIN from metadata    
def check_df_for_vendorGSTIN(x,metadata_VGSTIN):
    if metadata_VGSTIN in str(x):
        return True
    return False
def get_VGSTIN_token(df,metadata_VGSTIN:str):
    try:
        
        df["is_VGSTIN"] = df["text"].apply(lambda x: check_df_for_vendorGSTIN(str(x).upper(), metadata_VGSTIN.upper()))
        fdf = df[df["is_VGSTIN"]==True]
        print(fdf.shape)
        if len(fdf) > 0:
            fdf = fdf.reset_index(drop = True) 
            #print("s",fdf["token_id"][0])
            return fdf["token_id"][0]
        else:
            return
    except:
        print("find get_VGSTIN_token exception :",traceback.print_exc())
        return 
def validate_VGSTIN(df : pd.DataFrame, prediction : dict ,metadata_VGSTIN:str,VENDOR_MASTERDATA) -> dict:
    try:
        print("vendor masterdata shape:",VENDOR_MASTERDATA.shape)
        print('Meta VGSTIN num:',metadata_VGSTIN)
        if (metadata_VGSTIN == None) or (metadata_VGSTIN == ""):
            return prediction
        metadata_VGSTIN = metadata_VGSTIN.upper()
        token = get_VGSTIN_token(df,metadata_VGSTIN)
        if token:
            token_df = df[df["token_id"] == token]
            print("token_df :",token_df.shape)
            for _, row in token_df.iterrows():
                print("Metadata GSTIN matched in DF")
                #vendorGSTIN = metadata_VGSTIN
                V_GSTIN_frm_buyersData = VENDOR_MASTERDATA[VENDOR_MASTERDATA["VENDOR_GSTIN"].str.upper() == metadata_VGSTIN]
                print("Match DF shape :",V_GSTIN_frm_buyersData.shape[0])
                if V_GSTIN_frm_buyersData.shape[0] == 1:
                    print("one match found for GSTIN in vendor masterdata from bot")
                    prediction.update(add_new_fields("vendorGSTIN",row,from_entity=True))
                break
        return prediction
    except:
        print('exception in validate_VGSTIN', traceback.print_exc())
        return prediction
