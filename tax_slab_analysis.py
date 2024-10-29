import pandas as pd
import copy
import traceback
import re
import math
from calculateAmountFields import check_if_cgst_1
def get_extract_amount_for_tax_slab(df):
    df["extract_amount_for_tax_slab"] = 0
    try:
        amt_list = []
        for i in df["text"]:
            #print(str(i).replace(",","").replace(".",""))
            try:
                amt = float((str(i).replace(",","")))
                amt_list.append(amt)
            except:
                amt_list.append(0)
            #str(i).replace(",","")
        df["extract_amount_for_tax_slab"] = amt_list
        return df
    except:
        print(traceback.print_exc())
        return df

# 2 May 2023 Identifying External GST Table Header words
def get_hdr_line_num_external_gst(df,header_keywords):
    table_start = 0
    try:
        pages = df["page_num"].unique()
        keyword_match = []

        for page in pages:
            filter_pages = df[(df["page_num"] == page) & (df["is_HDR"] == 0)]
            lines = filter_pages["line_num"].unique()
            total_lines = len(lines)
            #print(f"total lines {total_lines} in page {page}")
            for idx, row in filter_pages.iterrows():
                if str(row["line_text"]).lower() in header_keywords:
                    #print(row["line_num"],row["text"])
                    keyword_match.append((row["tableLineNo_New"],row["line_top"],str(row["text"]).lower()))
        #print("Matched Keyword",keyword_match)
        threshold_distance = 2
        dict_close_keyword = {}
        for i in range(len(keyword_match)):
            for j in range(len(keyword_match)):
                if i!=j:
                    if abs(keyword_match[i][0] - keyword_match[j][0]) < threshold_distance:
                        if dict_close_keyword.get(keyword_match[i][0])!= None:                        
                            lst2 = dict_close_keyword.get(keyword_match[i][0])
                            if keyword_match[j] not in lst2:
                                lst2.append(keyword_match[j])
                            dict_close_keyword[keyword_match[i][0]] = lst2
                        else:
                            lst = []
                            lst.append(keyword_match[i])
                            lst.append(keyword_match[j])
                            dict_close_keyword[keyword_match[i][0]] = lst
        print("Final Keywords in dict for external gst table are:",dict_close_keyword)
        if len(dict_close_keyword) == 1:
            print("Only 1 match found")
            table_start = list(dict_close_keyword.keys())[0]
        else:
            print("Multiple matches are present. Applying threshold and taking final tableLineno")
            threshold_keywords = 3
            dct_final_words = {}
            for key, value in dict_close_keyword.items():
                #print("sahil",key,len(value))
                if len(value) > threshold_keywords:
                    dct_final_words[key] = value
            #print(dct_final_words)
            
            for key, value in dct_final_words.items():
                for item in value:
                    if item[0] > table_start:
                        table_start = item[0]
        return table_start
    except:
        print("get_hdr_line_num_external_gst",traceback.print_exc())
        return 0

# 2 May 2023 Identifying External GST Table Footer words
def get_footer_line_num_external_gst(df,footer_keywords,table_start): 
    table_end = 0
    try: 
        filter_df = df[df["tableLineNo_New"]>(table_start+1)]
        keyword_match = []
        for idx, row in filter_df.iterrows():
            if row["line_text"].lower() in footer_keywords:
                #print(row["line_num"],row["text"])
                keyword_match.append((row["tableLineNo_New"],row["line_top"],str(row["text"]).lower()))
        print("Matched Keyword",keyword_match)
        if len(keyword_match) != 0:
            for item in keyword_match:
                table_end = item[0]
                break
        else:
            print("Footer Keyword not found")
            table_end = max(df["tableLineNo_New"].unique())
        print(f"table start :{table_start}, table ends :{table_end}")
        return table_end
    except:
        print("get_footer_line_num_external_gst exception",traceback.print_exc())
        table_end = max(df["tableLineNo_New"].unique())
        return table_end 

# 2 May 2023 Identifying External GST Table
def find_gst_table(df):
    table_start = 0
    table_end = 0 
    from TAPPconfig import getHdrKeywordsGST, getFooterKeywordsGST
    header_keywords = getHdrKeywordsGST()
    footer_keywords = getFooterKeywordsGST()
    table_start = get_hdr_line_num_external_gst(df,header_keywords)
    if table_start!=0:
        table_end = get_footer_line_num_external_gst(df,footer_keywords,table_start)
    return table_start,table_end

# 2 May 2023 Get tax slab from extrenal GST Table
def tax_slab_v4(df,cgst_present):
    import traceback
    try:
        def remove_duplicates(final_list):
            if len(final_list) == 2 and math.isclose(final_list[0]["taxable"],final_list[1]["taxable"],abs_tol=0.5):
                del final_list[0]
                return final_list
            return final_list
        
        #cgst_present = check_if_cgst_1(df)
        if cgst_present == -1:
            return []
        table_start, table_end = find_gst_table(df)
        if table_start !=0:
            filter_df = df[(df["tableLineNo_New"] > table_start) & (df["tableLineNo_New"] < table_end) &(df["line_row_new"]==0) & (df["extracted_amount"]>0)]
            #filter_df.to_csv(r"C:\Users\Admin\Desktop\filt.csv")
        else:
            #filter_df = df[(df["line_row_new"]==0) & (df["extracted_amount"]>0)]
            print("External GST Table not found")
            return []
        final_list = []
        
        if filter_df.shape[0]>0:
            cgst_list = [14,9,6,2.5]
            igst_list = [5,12,18,28]
            
            max_amount = 0
            if cgst_present == 1:
                unique_amounts = filter_df["extracted_amount"].unique()
                #print(unique_amounts)
                for amount in unique_amounts:
                    d={}
                    #print("lets assume assount as",amount)
                    found = False
                    for rate in cgst_list:
                        cgst_amount = amount*rate/100
                        #print("tax = ",cgst_amount)
                        for value in unique_amounts:
                            if math.isclose(cgst_amount,value,abs_tol=0.02):
                                if amount > 10 and cgst_amount > 5:
                                    #print("------------ tax found",cgst_amount)
                                    d["taxable"] = amount
                                    d["cgst_percentage"] = rate
                                    d["sgst_percentage"] = rate
                                    d["cgst_amount"] = value
                                    d["sgst_amount"] = value
                                    if max_amount < amount:
                                        max_amount = amount
                                    final_list.append(d)
                                    found = True
                                    break
                        if found == True:
                            break
                        
            elif cgst_present == 0:
                unique_amounts = filter_df["extracted_amount"].unique()
                #print(unique_amounts)
                for amount in unique_amounts:
                    d={}
                    #print("lets assume assount as",amount)
                    found = False
                    for rate in igst_list:
                        igst_amount = amount*rate/100
                        #print("tax = ",igst_amount,"rate:",rate)
                        for value in unique_amounts:
                            #print("value:", value)
                            if math.isclose(igst_amount,value,abs_tol=0.05):
                                if igst_amount > 9:
                                    #print("------------ tax found",igst_amount)
                                    d["taxable"] = amount
                                    d["igst_percentage"] = rate
                                    d["igst_amount"] = value
                                    if max_amount < amount:
                                        max_amount = amount
                                    final_list.append(d)
                                    found = True
                                    break
                        if found == True:
                            break        
            
            #print("Final List V1:",final_list)
            #print("max_amount",max_amount)
            final_list = remove_duplicates(final_list)
            #print("Final List V2:",final_list)
            total_sum = 0

            ## Removing total amount from list
            for item in final_list:
                #print(item)
                #print(item["taxable"])
                if item["taxable"]!= max_amount:
                    total_sum += item["taxable"]        
            if math.isclose(total_sum,max_amount,abs_tol=0.02):
                #print("Total sum equals max amount")
                for index,item in enumerate(final_list):
                    if item["taxable"] == max_amount:
                        #print("Index=",index)
                        del final_list[index]
                        break
            
            #print("Final List V3:",final_list)       
            #call(final_list,cgst_present)    
        return final_list
    except:
        print("tax_slab_v4 exception",traceback.print_exc())
        return [] 
# 3 May 2023 Added code for extracting slabs for external GST Table similar to Line Items
def tax_slab_gst_table(df,cgst_present):
    import traceback
    try:
        def remove_duplicates(final_list):
            if len(final_list) == 2 and math.isclose(final_list[0]["taxable"],final_list[1]["taxable"],abs_tol=0.5):
                del final_list[0]
                return final_list
            return final_list
        
        #cgst_present = check_if_cgst_1(df)
        if cgst_present == -1:
            return []
        df = get_extract_amount_for_tax_slab(df)
        table_start, table_end = find_gst_table(df)
        if abs(table_end - table_start) > 30:
            print("Wrong GST Table detected")
            return []
        if table_start !=0:
            filter_df2 = df[(df["tableLineNo_New"] > table_start) & (df["tableLineNo_New"] < table_end) & (df["line_row_new"]==0) & (df["extracted_amount"]>0)]
            #filter_df2.to_csv(r"C:\Users\Admin\Desktop\filt.csv")
        else:
            #filter_df = df[(df["line_row_new"]==0) & (df["extracted_amount"]>0)]
            print("External GST Table not found")
            return []
        unique_rows = list(map(int,filter_df2["tableLineNo_New"].unique()))
        print(unique_rows)
        final_list = []
        max_amount = 0
        for line_num in unique_rows:
            if line_num != 0:
                filter_df = filter_df2[filter_df2["tableLineNo_New"] == line_num]
                if filter_df.shape[0]>0:
                    cgst_list = [14,9,6,2.5]
                    igst_list = [5,12,18,28]

                    if cgst_present == 1:
                        unique_amounts = filter_df["extracted_amount"].unique()
                        #print(unique_amounts)
                        for amount in unique_amounts:
                            d={}
                            #print("lets assume assount as",amount)
                            found = False
                            for rate in cgst_list:
                                cgst_amount = amount*rate/100
                                #print("tax = ",cgst_amount)
                                for value in unique_amounts:
                                    if math.isclose(cgst_amount,value,abs_tol=0.02):
                                        if amount > 10 and cgst_amount > 5:
                                            #print("------------ tax found",cgst_amount)
                                            d["taxable"] = amount
                                            d["cgst_percentage"] = rate
                                            d["sgst_percentage"] = rate
                                            d["cgst_amount"] = value
                                            d["sgst_amount"] = value
                                            if max_amount < amount:
                                                max_amount = amount
                                            final_list.append(d)
                                            found = True
                                            break
                                if found == True:
                                    break
                                
                    elif cgst_present == 0:
                        unique_amounts = filter_df["extracted_amount"].unique()
                        #print(unique_amounts)
                        for amount in unique_amounts:
                            d={}
                            #print("lets assume assount as",amount)
                            found = False
                            for rate in igst_list:
                                igst_amount = amount*rate/100
                                #print("tax = ",igst_amount,"rate:",rate)
                                for value in unique_amounts:
                                    #print("value:", value)
                                    if math.isclose(igst_amount,value,abs_tol=0.05):
                                        if igst_amount > 9:
                                            #print("------------ tax found",igst_amount)
                                            d["taxable"] = amount
                                            d["igst_percentage"] = rate
                                            d["igst_amount"] = value
                                            if max_amount < amount:
                                                max_amount = amount
                                            final_list.append(d)
                                            found = True
                                            break
                                if found == True:
                                    break        
                    
                    #print("Final List V1:",final_list)
                    #print("max_amount",max_amount)
        final_list = remove_duplicates(final_list)
        #print("Final List V2:",final_list)
        total_sum = 0

        ## Removing total amount from list
        for item in final_list:
            #print(item)
            #print(item["taxable"])
            if item["taxable"]!= max_amount:
                total_sum += item["taxable"] 
        #print("total_calculated_sum", total_sum)       
        if math.isclose(total_sum,max_amount,abs_tol=0.5):
            #print("Total sum equals max amount")
            for index,item in enumerate(final_list):
                if item["taxable"] == max_amount:
                    #print("Index=",index)
                    del final_list[index]
                    break
        
        #print("Final List V3:",final_list)       
        #call(final_list,cgst_present)    
        return final_list
    except:
        print("tax_slab_gst_table exception",traceback.print_exc())
        return [] 

# 20 April 2023 Added only for discr note 
def tax_slab_before_LI_discr_combined(df,final_list,cgst_present):
    try:
        def remove_duplicates(final_list):
            if len(final_list) == 2 and math.isclose(final_list[0]["taxable"],final_list[1]["taxable"],abs_tol=0.5):
                del final_list[0]
                return final_list
            return final_list
        if cgst_present == -1:
            return []
        if len(final_list) == 0:
            filter_df = df[(df["line_row_new"]==0) & df["extracted_amount"]>0]    
            final_list = []
            if filter_df.shape[0]>0:
                igst_list = [5,12,18,28]
                cgst_list = [5,12,18,28]
                max_amount = 0
                if cgst_present == 1:
                    cgst_list = get_slabs_value_discr_combined(df,cgst_list)
                    print("CGST List is:", cgst_list)
                    unique_amounts = filter_df["extracted_amount"].unique()
                    if len(unique_amounts)!=0:
                        unique_amounts[::-1].sort()
                    # print(unique_amounts)
                    for amount in unique_amounts:
                        d={}
                        # print("lets assume assount as",amount)
                        found = False
                        for rate in cgst_list:
                            cgst_amount = amount*rate/100
                            if cgst_amount > 0.10:
                                # print("tax = ",cgst_amount)
                                for value in unique_amounts:
                                    if math.isclose(value, (cgst_amount+amount), abs_tol=0.5):
                                        d["taxable"] = amount
                                        d["cgst_percentage"] = round(rate/2,2)
                                        d["sgst_percentage"] = round(rate/2,2)
                                        d["cgst_amount"] = round(cgst_amount/2,2)
                                        d["sgst_amount"] = round(cgst_amount/2,2)
                                        if max_amount < amount:
                                            max_amount = amount
                                        final_list.append(d)
                                        found = True
                                        break
                            if found == True:
                                # print("found")
                                break
                        
                elif cgst_present == 0:
                    igst_list = get_slabs_value_discr_combined(df,igst_list)
                    print("IGST List is:", igst_list)
                    unique_amounts = filter_df["extracted_amount"].unique()
                    if len(unique_amounts)!=0:
                        unique_amounts[::-1].sort()
                    #print(unique_amounts)
                    for amount in unique_amounts:
                        d={}
                        #print("lets assume assount as",amount)
                        found = False
                        for rate in igst_list:
                            igst_amount = amount*rate/100
                            #print("tax = ",igst_amount,"rate:",rate)
                            for value in unique_amounts:
                                #print("value:", value)
                                if math.isclose(igst_amount,value,abs_tol=0.5):
                                        #print("------------ tax found",igst_amount)
                                        d["taxable"] = amount
                                        d["igst_percentage"] = rate
                                        d["igst_amount"] = round(value,2)
                                        if max_amount < amount:
                                            max_amount = amount
                                        final_list.append(d)
                                        found = True
                                        break
                            if found == True:
                                break        
                
                #print("Final List V1:",final_list)
                #print("max_amount",max_amount)
                final_list = remove_duplicates(final_list)
                #print("Final List V2:",final_list)
                total_sum = 0

                ## Removing total amount from list
                for item in final_list:
                    #print(item)
                    #print(item["taxable"])
                    if item["taxable"]!= max_amount:
                        total_sum += item["taxable"]        
                if math.isclose(total_sum,max_amount,abs_tol=0.02):
                    #print("Total sum equals max amount")
                    for index,item in enumerate(final_list):
                        if item["taxable"] == max_amount:
                            #print("Index=",index)
                            del final_list[index]
                            break
                
            #print("Final List V3:",final_list)       
            #call(final_list,cgst_present)    
        return final_list
    except:
        print("tax_slab_before_LI_discr_combined exception",traceback.print_exc())
        return []

def tax_slab_before_LI(df,cgst_present):
    try:
        import traceback
        def remove_duplicates(final_list):
            if len(final_list) == 2 and math.isclose(final_list[0]["taxable"],final_list[1]["taxable"],abs_tol=0.5):
                del final_list[0]
                return final_list
            return final_list
        
        #cgst_present = check_if_cgst_1(df)
        if cgst_present == -1:
            return []
        maximum_inLI = max(list(df["line_row_new"].unique()))
        if maximum_inLI !=0:
            b = df.loc[df["line_row_new"] == maximum_inLI,["token_id"]]
            tid = max(list(b["token_id"].unique()))
            #print("sahil",maximum_inLI, tid)
            filter_df = df[(df["line_row_new"]==0) & (df["extracted_amount"]>0) & (df["token_id"]>tid)]
        else:
            filter_df = df[(df["line_row_new"]==0) & (df["extracted_amount"]>0)]
        # filter_df.to_csv(r"C:\Users\Admin\Desktop\filt1.csv")
        final_list = []
        
        if filter_df.shape[0]>0:
            cgst_list = [14,9,6,2.5]
            igst_list = [5,12,18,28]
            
            max_amount = 0
            if cgst_present == 1:
                unique_amounts = filter_df["extracted_amount"].unique()
                #print(unique_amounts)
                for amount in unique_amounts:
                    d={}
                    #print("lets assume assount as",amount)
                    found = False
                    for rate in cgst_list:
                        cgst_amount = amount*rate/100
                        #print("tax = ",cgst_amount)
                        for value in unique_amounts:
                            if math.isclose(cgst_amount,value,abs_tol=0.02):
                                if amount > 10 and cgst_amount > 5:
                                    #print("------------ tax found",cgst_amount)
                                    d["taxable"] = amount
                                    d["cgst_percentage"] = rate
                                    d["sgst_percentage"] = rate
                                    d["cgst_amount"] = value
                                    d["sgst_amount"] = value
                                    if max_amount < amount:
                                        max_amount = amount
                                    final_list.append(d)
                                    found = True
                                    break
                        if found == True:
                            break
                        
            elif cgst_present == 0:
                unique_amounts = filter_df["extracted_amount"].unique()
                #print(unique_amounts)
                for amount in unique_amounts:
                    d={}
                    #print("lets assume assount as",amount)
                    found = False
                    for rate in igst_list:
                        igst_amount = amount*rate/100
                        #print("tax = ",igst_amount,"rate:",rate)
                        for value in unique_amounts:
                            #print("value:", value)
                            if math.isclose(igst_amount,value,abs_tol=0.05):
                                if igst_amount > 9:
                                    #print("------------ tax found",igst_amount)
                                    d["taxable"] = amount
                                    d["igst_percentage"] = rate
                                    d["igst_amount"] = value
                                    if max_amount < amount:
                                        max_amount = amount
                                    final_list.append(d)
                                    found = True
                                    break
                        if found == True:
                            break        
            
            #print("Final List V1:",final_list)
            #print("max_amount",max_amount)
            final_list = remove_duplicates(final_list)
            #print("Final List V2:",final_list)
            total_sum = 0

            ## Removing total amount from list
            for item in final_list:
                #print(item)
                #print(item["taxable"])
                if item["taxable"]!= max_amount:
                    total_sum += item["taxable"]        
            if math.isclose(total_sum,max_amount,abs_tol=0.02):
                #print("Total sum equals max amount")
                for index,item in enumerate(final_list):
                    if item["taxable"] == max_amount:
                        #print("Index=",index)
                        del final_list[index]
                        break
            
            #print("Final List V3:",final_list)       
            #call(final_list,cgst_present)    
        return final_list
    except:
        print("tax_slab_before_LI exception",traceback.print_exc())
        return []

## 19 sept 2023 Seprated Tax identification for Discr Note
def tax_slab_before_LI_discr(df,cgst_present):
    try:
        import traceback
        def remove_duplicates(final_list):
            if len(final_list) == 2 and math.isclose(final_list[0]["taxable"],final_list[1]["taxable"],abs_tol=0.5):
                del final_list[0]
                return final_list
            return final_list
        #cgst_present = check_if_cgst_1(df)
        if cgst_present == -1:
            return []
        maximum_inLI = max(list(df["line_row_new"].unique()))
        if maximum_inLI !=0:
            b = df.loc[df["line_row_new"] == maximum_inLI,["token_id"]]
            tid = max(list(b["token_id"].unique()))
            #print("sahil",maximum_inLI, tid)
            filter_df = df[(df["line_row_new"]==0) & (df["extracted_amount"]>0) & (df["token_id"]>tid)]
        else:
            filter_df = df[(df["line_row_new"]==0) & (df["extracted_amount"]>0)]
        #filter_df.to_csv(r"C:\Users\Admin\Desktop\filt1.csv")
        final_list = []
        
        if filter_df.shape[0]>0:
            cgst_list = [14,9,6,2.5]
            igst_list = [5,12,18,28]
            
            max_amount = 0
            if cgst_present == 1:
                cgst_list = get_slabs_value_discr(df,cgst_list)
                print("CGST List is:", cgst_list)
                unique_amounts = filter_df["extracted_amount"].unique()
                if len(unique_amounts)!=0:
                        unique_amounts[::-1].sort()
                #print(unique_amounts)
                for amount in unique_amounts:
                    d={}
                    #print("lets assume assount as",amount)
                    found = False
                    for rate in cgst_list:
                        cgst_amount = amount*rate/100
                        #print("tax = ",cgst_amount)
                        if cgst_amount > 0.10:
                            for value in unique_amounts:
                                if math.isclose(cgst_amount,value,abs_tol=0.15):
                                        #print("------------ tax found",cgst_amount)
                                        d["taxable"] = amount
                                        d["cgst_percentage"] = rate
                                        d["sgst_percentage"] = rate
                                        d["cgst_amount"] = round(value,2)
                                        d["sgst_amount"] = round(value,2)
                                        if max_amount < amount:
                                            max_amount = amount
                                        final_list.append(d)
                                        found = True
                                        break
                        if found == True:
                            break
                        
            elif cgst_present == 0:
                igst_list = get_slabs_value_discr(df,igst_list)
                print("IGST List is:", igst_list)
                unique_amounts = filter_df["extracted_amount"].unique()
                if len(unique_amounts)!=0:
                        unique_amounts[::-1].sort()
                #print(unique_amounts)
                for amount in unique_amounts:
                    d={}
                    #print("lets assume assount as",amount)
                    found = False
                    for rate in igst_list:
                        igst_amount = amount*rate/100
                        #print("tax = ",igst_amount,"rate:",rate)
                        for value in unique_amounts:
                            #print("value:", value)
                            if math.isclose(igst_amount,value,abs_tol=0.15):
                                    #print("------------ tax found",igst_amount)
                                    d["taxable"] = amount
                                    d["igst_percentage"] = rate
                                    d["igst_amount"] = round(value,2)
                                    if max_amount < amount:
                                        max_amount = amount
                                    final_list.append(d)
                                    found = True
                                    break
                        if found == True:
                            break        
            
            #print("Final List V1:",final_list)
            #print("max_amount",max_amount)
            final_list = remove_duplicates(final_list)
            #print("Final List V2:",final_list)
            total_sum = 0

            ## Removing total amount from list
            for item in final_list:
                #print(item)
                #print(item["taxable"])
                if item["taxable"]!= max_amount:
                    total_sum += item["taxable"]        
            if math.isclose(total_sum,max_amount,abs_tol=0.02):
                #print("Total sum equals max amount")
                for index,item in enumerate(final_list):
                    if item["taxable"] == max_amount:
                        #print("Index=",index)
                        del final_list[index]
                        break
            
            #print("Final List V3:",final_list)       
            #call(final_list,cgst_present)    
        return final_list
    except:
        print("tax_slab_before_LI_discr exception",traceback.print_exc())
        return []


# 20 April 2023 Only added for discr note Different format is present so there is some change in logic
def tax_slab_line_item_discr_combined(df,final_list,cgst_present):
    try:
        def remove_duplicates(final_list):
            if len(final_list) == 2 and final_list[0]["taxable"] == final_list[1]["taxable"]:
                del final_list[0]
                return final_list
            return final_list
        df = get_extract_amount_for_tax_slab(df)
        # df.to_csv(r"C:\Users\sahil.aggarwal\Desktop\Untitled.csv")
        if len(final_list) == 0:
            unique_rows = list(map(int,df["line_row_new"].unique()))
            #print(unique_rows)
            final_list = []
            max_amount = 0
            cgst_list = [5,12,18,28]
            #cgst_list_double = [5,12,18,28]
            igst_list = [5,12,18,28]
            if cgst_present == 1:
                cgst_list = get_slabs_value_discr_combined(df,cgst_list)
                print("CGST List is:", cgst_list)
            elif cgst_present == 0:
                igst_list = get_slabs_value_discr_combined(df,igst_list)
                print("IGST List is:", igst_list)    
            for line_num in unique_rows:
                if line_num != 0:
                    filter_df = df[(df["line_row_new"] == line_num) & (df["extract_amount_for_tax_slab"]>0) & (~df["tbl_col_hdr"].str.contains('sr|qty|no|remark'))]
                    #print(filter_df)
                    #filter_df.to_csv(r"C:\Users\Admin\Desktop\filt.csv")
                    if cgst_present == 1:
                        unique_amounts = filter_df["extract_amount_for_tax_slab"].unique()
                        if len(unique_amounts)!=0:
                            unique_amounts[::-1].sort()
                        found = False
                        for amount in unique_amounts:
                            d={}
                            # print("lets assume assount as",amount)
                            for rate in cgst_list:
                                cgst_amount = amount*rate/100
                                # print("tax = ",rate,(cgst_amount+amount))
                                if cgst_amount > 0.10:
                                    for value in unique_amounts:
                                        
                                        if (value != amount)  and (math.isclose((cgst_amount+amount),value,abs_tol=0.5)):
                                                # print("------------ tax found",cgst_amount)
                                                d["taxable"] = amount
                                                d["cgst_percentage"] = round(rate/2,2)
                                                d["sgst_percentage"] = round(rate/2,2)
                                                d["cgst_amount"] = round(cgst_amount/2,2)
                                                d["sgst_amount"] = round(cgst_amount/2,2)
                                                if max_amount < amount:
                                                    max_amount = amount
                                                final_list.append(d)
                                                found = True
                                                break
                                if found == True:
                                    break
                            if found == True:
                                break
                        # if found == True:
                        #     break
                    elif cgst_present == 0:
                                                
                        unique_amounts = filter_df["extract_amount_for_tax_slab"].unique()
                        if len(unique_amounts)!=0:
                            unique_amounts[::-1].sort()
                        #print(unique_amounts,line_num)
                        #print("------------------")
                        found = False
                        for amount in unique_amounts:
                            d={}
                            #print("lets assume assount as",amount, "line num", line_num)
                            for rate in igst_list:
                                igst_amount = amount*rate/100
                                #print("tax = ",igst_amount)
                                for value in unique_amounts:
                                    if (value != amount)  and (math.isclose((igst_amount+amount),value,abs_tol=0.5)):
                                            #print("------------ tax found",igst_amount)
                                            d["taxable"] = amount
                                            d["igst_percentage"] = rate
                                            d["igst_amount"] = round(igst_amount,2)
                                            if max_amount < amount:
                                                max_amount = amount
                                            final_list.append(d)
                                            found = True
                                            break
                                if found == True:
                                    break
                            if found == True:
                                break
            #print("Final List V1:",final_list)
            #print("max_amount",max_amount)
            
            final_list = remove_duplicates(final_list)
            #print("Final List V2:",final_list)
            total_sum = 0
            for item in final_list:
                #print(item)
                #print(item["taxable"])
                #if item["taxable"]!= max_amount:
                if not math.isclose(item["taxable"],max_amount,abs_tol=0.05):
                    total_sum += item["taxable"]
            #print("Total Sum",total_sum)
            if math.isclose(total_sum,max_amount,abs_tol=0.05):
                #print("Total sum equals max amount")
                for index,item in enumerate(final_list):
                    if math.isclose(item["taxable"],max_amount,abs_tol=0.05):
                        #print("Index=",index)
                        del final_list[index]
                        break
            #print("Final List V3:",final_list)              
        return final_list
    except:
        print("tax_slab_line_item_discr_combined exception",traceback.print_exc())
        return []

def tax_slab_line_item(df,cgst_present):
    try:
        def remove_duplicates(final_list):
            if len(final_list) == 2 and final_list[0]["taxable"] == final_list[1]["taxable"]:
                del final_list[0]
                return final_list
            return final_list
        df = get_extract_amount_for_tax_slab(df)
        #df.to_csv(r"C:\Users\Admin\Desktop\Untitled.csv")
        unique_rows = list(map(int,df["line_row_new"].unique()))
        #print(unique_rows)
        final_list = []
        max_amount = 0
        for line_num in unique_rows:
            
            if line_num != 0:
                filter_df = df[(df["line_row_new"] == line_num) & (df["extract_amount_for_tax_slab"]>0)]
                #print(filter_df)
                #filter_df.to_csv(r"C:\Users\Admin\Desktop\filt.csv")
                cgst_list = [14,9,6,2.5]
                #cgst_list_double = [5,12,18,28]
                igst_list = [5,12,18,28]
                
                if cgst_present == 1:
                    unique_amounts = filter_df["extract_amount_for_tax_slab"].unique()
                    if len(unique_amounts)!=0:
                        unique_amounts[::-1].sort()
                    #print(unique_amounts)
                    found = False
                    for amount in unique_amounts:
                        d={}
                        #print("lets assume assount as",amount)
                        for rate in cgst_list:
                            cgst_amount = amount*rate/100
                            #print("tax = ",rate,cgst_amount)
                            for value in unique_amounts:
                                if math.isclose(cgst_amount,value,abs_tol=0.2):
                                    if amount > 10 and cgst_amount > 5:
                                        #print("------------ tax found",cgst_amount)
                                        d["taxable"] = amount
                                        d["cgst_percentage"] = rate
                                        d["sgst_percentage"] = rate
                                        d["cgst_amount"] = value
                                        d["sgst_amount"] = value
                                        if max_amount < amount:
                                            max_amount = amount
                                        final_list.append(d)
                                        found = True
                                        break
                            if found == True:
                                break
                        if found == True:
                            break
                elif cgst_present == 0:
                    unique_amounts = filter_df["extract_amount_for_tax_slab"].unique()
                    if len(unique_amounts)!=0:
                        unique_amounts[::-1].sort()
                    #print(unique_amounts,line_num)
                    #print("------------------")
                    found = False
                    for amount in unique_amounts:
                        d={}
                        #print("lets assume assount as",amount, "line num", line_num)
                        for rate in igst_list:
                            igst_amount = amount*rate/100
                            #print("tax = ",taxable)
                            for value in unique_amounts:
                                if math.isclose(igst_amount,value,abs_tol=0.2):
                                    if igst_amount > 10 :
                                        #print("------------ tax found",igst_amount)
                                        d["taxable"] = amount
                                        d["igst_percentage"] = rate
                                        d["igst_amount"] = value
                                        if max_amount < amount:
                                            max_amount = amount
                                        final_list.append(d)
                                        found = True
                                        break
                            if found == True:
                                break
                        if found == True:
                            break
        #print("Final List V1:",final_list)
        #print("max_amount",max_amount)

        final_list = remove_duplicates(final_list)
        #print("Final List V2:",final_list)
        total_sum = 0
        for item in final_list:
            #print(item)
            #print(item["taxable"])
            #if item["taxable"]!= max_amount:
            if not math.isclose(item["taxable"],max_amount,abs_tol=0.05):
                total_sum += item["taxable"]
        #print("Total Sum",total_sum)
        if math.isclose(total_sum,max_amount,abs_tol=0.05):
            #print("Total sum equals max amount")
            for index,item in enumerate(final_list):
                if math.isclose(item["taxable"],max_amount,abs_tol=0.05):
                    #print("Index=",index)
                    del final_list[index]
                    break
        #print("Final List V3:",final_list)              
        return final_list
    except:
        print("tax_slab_line_item exception",traceback.print_exc())
        return []

def get_slabs_value_discr(filter_df,tax_list):
    try:
        gst_label_list = []
        for item in filter_df["tbl_col_hdr"].unique():
            if "gst" in item.lower():
                gst_label_list.append(item)
        s = set()
        filter_rows = filter_df[filter_df["tbl_col_hdr"].isin(gst_label_list)]
        for index, row in filter_rows.iterrows():
            try:
                tax = float(row["text"])
                if (tax in tax_list) or (tax == 0):
                    s.add(tax)
                ## Sometimes in LI combined slabs are present
                if (tax/2 in tax_list):
                    s.add(tax/2)
            except:
                pass    
        # Removing unnecessary taxes which are caught
        s_list = list(s)
        for i in s_list:
            if (i != 0) and (i not in tax_list):
                s.remove(i)
                
        if len(s)>0:
            return list(s)
        else:
            return tax_list
    except Exception as e:
        print("Exception in get_slabs_value_discr",e, traceback.print_exc())
        return tax_list

def get_slabs_value_discr_combined(filter_df,tax_list):
    try:
        gst_label_list = []
        for item in filter_df["tbl_col_hdr"].unique():
            if "gst" in item:
                gst_label_list.append(item)
        s = set()
        filter_rows = filter_df[filter_df["tbl_col_hdr"].isin(gst_label_list)]
        for index, row in filter_rows.iterrows():
            try:
                tax = float(row["text"])
                # if (tax != 0) and (tax in tax_list) or (tax == 0) or ((tax/2) in tax_list):
                ## 15 march 2024 modified if condition
                if ((tax == 0)) or (tax in tax_list):
                    s.add(tax)
            except:
                pass
        # Removing unnecessary taxes which are caught
        s_list = list(s)
        for i in s_list:
            # if (i != 0) and (i not in tax_list) and ((i/2) not in tax_list):
            ## 15 march 2024 modified if condition
            if (i != 0) and (i not in tax_list):
                s.remove(i)
                
        if len(s)>0:
            return list(s)
        else:
            return tax_list
    except Exception as e:
        print("Exception in get_slabs_value_discr_combined",e, traceback.print_exc())
        return tax_list
                
# 27 sept 2023 Different template for LI Present 
def tax_slab_line_item_discr_v2(df,cgst_present):
    try:
        def remove_duplicates(final_list):
            if len(final_list) == 2 and final_list[0]["taxable"] == final_list[1]["taxable"]:
                del final_list[0]
                return final_list
            return final_list
        df = get_extract_amount_for_tax_slab(df)
        #df.to_csv(r"C:\Users\Admin\Desktop\Untitled.csv")
        unique_rows = list(map(int,df["line_row_new"].unique()))
        #print(unique_rows)
        final_list = []
        max_amount = 0
        cgst_list = [14,9,6,2.5]
        igst_list = [5,12,18,28]
        if cgst_present == 1:
            cgst_list = get_slabs_value_discr(df,cgst_list)
            print("CGST List is:", cgst_list)
        elif cgst_present == 0:
            igst_list = get_slabs_value_discr(df,igst_list)
            print("IGST List is:", igst_list)
        for line_num in unique_rows: 
            if line_num != 0:
                filter_df = df[(df["line_row_new"] == line_num) & (df["extract_amount_for_tax_slab"]>0) & (~df["tbl_col_hdr"].str.contains('sr|qty|no|remark'))]
                if filter_df.shape[0]>0:
                    if cgst_present == 1:
                        unique_amounts = filter_df["extract_amount_for_tax_slab"].unique()
                        if len(unique_amounts)!=0:
                            unique_amounts[::-1].sort()
                        #print(unique_amounts)
                        found = False
                        for amount in unique_amounts:
                            d={}
                            #print("lets assume assount as",amount)
                            for rate in cgst_list:
                                cgst_amount = amount*rate/100
                                #print("tax = ",rate,cgst_amount)
                                if cgst_amount > 0.10:
                                    for value in unique_amounts:
                                        if (value != amount)  and math.isclose((cgst_amount+amount),value,abs_tol=0.2):
                                                #print("------------ tax found",cgst_amount)
                                                d["taxable"] = amount
                                                d["cgst_percentage"] = rate
                                                d["sgst_percentage"] = rate
                                                d["cgst_amount"] = round(cgst_amount,2)
                                                d["sgst_amount"] = round(cgst_amount,2)
                                                if max_amount < amount:
                                                    max_amount = amount
                                                final_list.append(d)
                                                found = True
                                                break
                                if found == True:
                                    break
                            if found == True:
                                break
                    elif cgst_present == 0:
                        unique_amounts = filter_df["extract_amount_for_tax_slab"].unique()
                        if len(unique_amounts)!=0:
                            unique_amounts[::-1].sort()
                        #print(unique_amounts,line_num)
                        #print("------------------")
                        found = False
                        for amount in unique_amounts:
                            d={}
                            #print("lets assume assount as",amount, "line num", line_num)
                            for rate in igst_list:
                                igst_amount = amount*rate/100
                                #print("tax = ",taxable)
                                for value in unique_amounts:
                                    if (value != amount)  and math.isclose((igst_amount+amount),value,abs_tol=0.2):
                                            #print("------------ tax found",igst_amount)
                                            d["taxable"] = amount
                                            d["igst_percentage"] = rate
                                            d["igst_amount"] = round(igst_amount,2)
                                            if max_amount < amount:
                                                max_amount = amount
                                            final_list.append(d)
                                            found = True
                                            break
                                if found == True:
                                    break
                            if found == True:
                                break
        #print("Final List V1:",final_list)
        #print("max_amount",max_amount)

        final_list = remove_duplicates(final_list)
        #print("Final List V2:",final_list)
        total_sum = 0
        for item in final_list:
            #print(item)
            #print(item["taxable"])
            #if item["taxable"]!= max_amount:
            if not math.isclose(item["taxable"],max_amount,abs_tol=0.05):
                total_sum += item["taxable"]
        #print("Total Sum",total_sum)
        if math.isclose(total_sum,max_amount,abs_tol=0.05):
            #print("Total sum equals max amount")
            for index,item in enumerate(final_list):
                if math.isclose(item["taxable"],max_amount,abs_tol=0.05):
                    #print("Index=",index)
                    del final_list[index]
                    break
        #print("Final List V3:",final_list)              
        return final_list
    except:
        print("tax_slab_line_item_discr exception",traceback.print_exc())
        return []

# 19 sept 2023 Seprated GST Extarction for discr 
def tax_slab_line_item_discr(df,cgst_present):
    try:
        def remove_duplicates(final_list):
            if len(final_list) == 2 and final_list[0]["taxable"] == final_list[1]["taxable"]:
                del final_list[0]
                return final_list
            return final_list
        df = get_extract_amount_for_tax_slab(df)
        #df.to_csv(r"C:\Users\Admin\Desktop\Untitled.csv")
        unique_rows = list(map(int,df["line_row_new"].unique()))
        # print("sahil21", unique_rows)
        final_list = []
        max_amount = 0
        cgst_list = [14,9,6,2.5]
        igst_list = [5,12,18,28]
        if cgst_present == 1:
            cgst_list = get_slabs_value_discr(df,cgst_list)
            print("CGST List is:", cgst_list)
        elif cgst_present == 0:
            igst_list = get_slabs_value_discr(df,igst_list)
            print("IGST List is:", igst_list)
        for line_num in unique_rows: 
            if line_num != 0:
                filter_df = df[(df["line_row_new"] == line_num) & (df["extract_amount_for_tax_slab"]>0) & (~df["tbl_col_hdr"].str.contains('sr|qty|no|remark'))] 
                if filter_df.shape[0]>0:
                    if cgst_present == 1:
                        unique_amounts = filter_df["extract_amount_for_tax_slab"].unique()
                        if len(unique_amounts)!=0:
                            unique_amounts[::-1].sort()
                        # print("line num is ", line_num, unique_amounts)
                        found = False
                        for amount in unique_amounts:
                            d={}
                            # print("lets assume assount as",amount)
                            for rate in cgst_list:
                                cgst_amount = amount*rate/100
                                # print("tax = ",rate,cgst_amount)
                                if cgst_amount > 0.10:
                                    for value in unique_amounts:
                                        if math.isclose(cgst_amount,value,abs_tol=0.2):
                                                print("------------ tax found",cgst_amount)
                                                d["taxable"] = amount
                                                d["cgst_percentage"] = rate
                                                d["sgst_percentage"] = rate
                                                d["cgst_amount"] = round(value,2)
                                                d["sgst_amount"] = round(value,2)
                                                if max_amount < amount:
                                                    max_amount = amount
                                                final_list.append(d)
                                                found = True
                                                break
                                if found == True:
                                    break
                            if found == True:
                                break
                    elif cgst_present == 0:
                        unique_amounts = filter_df["extract_amount_for_tax_slab"].unique()
                        if len(unique_amounts)!=0:
                            unique_amounts[::-1].sort()
                        #print(unique_amounts,line_num)
                        #print("------------------")
                        found = False
                        for amount in unique_amounts:
                            d={}
                            #print("lets assume assount as",amount, "line num", line_num)
                            for rate in igst_list:
                                igst_amount = amount*rate/100
                                #print("tax = ",taxable)
                                for value in unique_amounts:
                                    if math.isclose(igst_amount,value,abs_tol=0.2):
                                            #print("------------ tax found",igst_amount)
                                            d["taxable"] = amount
                                            d["igst_percentage"] = rate
                                            d["igst_amount"] = round(value,2)
                                            if max_amount < amount:
                                                max_amount = amount
                                            final_list.append(d)
                                            found = True
                                            break
                                if found == True:
                                    break
                            if found == True:
                                break
        #print("Final List V1:",final_list)
        #print("max_amount",max_amount)

        final_list = remove_duplicates(final_list)
        #print("Final List V2:",final_list)
        total_sum = 0
        for item in final_list:
            #print(item)
            #print(item["taxable"])
            #if item["taxable"]!= max_amount:
            if not math.isclose(item["taxable"],max_amount,abs_tol=0.05):
                total_sum += item["taxable"]
        #print("Total Sum",total_sum)
        if math.isclose(total_sum,max_amount,abs_tol=0.05):
            #print("Total sum equals max amount")
            for index,item in enumerate(final_list):
                if math.isclose(item["taxable"],max_amount,abs_tol=0.05):
                    #print("Index=",index)
                    del final_list[index]
                    break
        #print("Final List V3:",final_list)              
        return final_list
    except:
        print("tax_slab_line_item_discr exception",traceback.print_exc())
        return []

def tax_slab_v2(df):
    
    def get_taxes_amounts_from_table_anchor_column(df):
        final_list = []
        cgst_present = check_if_cgst_1(df)
        if cgst_present == -1:
            print("Can't identify tax structure")
            return None
        elif cgst_present==1:
            print("CGST Prsesent")
        elif cgst_present == 0:
            print("IGST Present")
        try:
            df["is_digit"] = df["text"].str.replace(".","").str.replace(",","").str.replace("%","").str.isdigit()
            fitler_df = df[(df["is_digit"]==True) & df["line_row_new"]==0]
            #fitler_df.to_csv(r"C:\Users\Admin\Desktop\filt2.csv")
            #print("filter shape",fitler_df.shape)
            #pred_labels = {}
            #pred_prob = {}
            for index,row in fitler_df.iterrows():
                # below 3 lines for exact match
                # lst_left_processed_ngbr = extract_list_from_string(string = row.left_processed_ngbr.lower())
                # for item in lst_left_processed_ngbr:
                #     if (item == "total") or (item == "totals") :
                cgst_percentage = 0
                sgst_percentage = 0
                igst_percentage = 0
                cgst_amount = 0
                sgst_amount = 0
                igst_amount = 0
                taxable = 0
                total = 0
                
                row_above_processed = str(row["above_processed_ngbr"].lower())
                row_left_processed = str(row["left_processed_ngbr"].lower())
                #print(type(row_above_processed),row_above_processed)
                if ("taxable" in row_above_processed) and ("total" not in row_left_processed) and ("rate" not in row_above_processed):
                    #print(type(row_above_processed),row_above_processed)
                    idx = row.token_id
                    
                    #print("\nAbove processed :",row.above_processed_ngbr.lower())
                    #print(type(row["extracted_amount"]))
                    if (isinstance(row["extracted_amount"],float) or isinstance(row["extracted_amount"],int)) and row["extracted_amount"]!=0:
                        row_dict={}
                        taxable = row.text
                        print("taxable",taxable)
                        right_ngbrs = ["W1Rg","W2Rg","W3Rg","W4Rg","W5Rg","W6Rg","W7Rg"]
                        for word in right_ngbrs:
                            filt2 = df[df["token_id"]==idx+int(word[1])].reset_index()
                            #print("word",row[word])
                            #print("abpn",str(filt2["above_processed_ngbr"]).lower())
                            filt2_above_processed_ngbr = str(filt2.loc[0,"above_processed_ngbr"]).lower()
                            filt2_extracted_amount = filt2.loc[0,"extracted_amount"]
                            if str(filt2.loc[0,"text"]).find("%")!=-1:
                                print("Contains Percentage",filt2_above_processed_ngbr)
                                #return filt2
                                if "rate" in filt2_above_processed_ngbr:
                                    rate = filt2.loc[0,"text"]
                                    if ("cgst" in filt2_above_processed_ngbr or "central" in filt2_above_processed_ngbr) and (cgst_present != 0):
                                        cgst_present = 1
                                        cgst_percentage = rate
                                        sgst_percentage = rate
                                        print("Found CGST Percentage",cgst_percentage)
                                    elif ("sgst" in filt2_above_processed_ngbr or "state" in filt2_above_processed_ngbr) and (cgst_present != 0):
                                        cgst_present = 1
                                        cgst_percentage = rate
                                        sgst_percentage = rate
                                        print("Found SGST Percentage",sgst_percentage)
                                    elif ("integrated" in filt2_above_processed_ngbr) and (cgst_present != 1):
                                        cgst_present = 0
                                        igst_percentage = rate
                                        print("Found IGST Percentage",igst_percentage)
                                    else:
                                        #rate = filt2.text
                                        if cgst_present == 1:
                                            cgst_percentage = rate
                                            sgst_percentage = rate
                                            print("CGST Rate Present",rate) 
                                        elif cgst_present == 0:
                                            igst_percentage = rate
                                            print("IGST Rate Present",rate)
                                        else:
                                            cgst_percentage = rate
                                            sgst_percentage = rate
                                            igst_percentage = rate
                                            print("Rate Present",rate)
                            if "amount" in filt2_above_processed_ngbr:
                                amount = filt2.loc[0,"text"]
                                #print(str(list(filt2["above_processed_ngbr"])).lower())
                                if ("cgst" in filt2_above_processed_ngbr or "central" in filt2_above_processed_ngbr) and (cgst_present != 0):
                                    if (isinstance(filt2_extracted_amount,float) or isinstance(filt2_extracted_amount,int)) and (filt2_extracted_amount!=0):
                                        cgst_present = 1
                                        cgst_amount = amount
                                        sgst_amount = amount
                                        print("Found cgst amount",cgst_amount)
                                elif ("sgst" in filt2_above_processed_ngbr or "state" in filt2_above_processed_ngbr) and (cgst_present != 0):
                                    if (isinstance(filt2_extracted_amount,float) or isinstance(filt2_extracted_amount,int)) and (filt2_extracted_amount!=0):
                                        cgst_present = 1
                                        cgst_amount = amount
                                        sgst_amount = amount
                                        print("Found sgst amount",sgst_amount)
                                elif ("integrated" in filt2_above_processed_ngbr) and (cgst_present != 1):
                                        if (isinstance(filt2_extracted_amount,float) or isinstance(filt2_extracted_amount,int)) and (filt2_extracted_amount!=0):
                                            cgst_present = 0
                                            igst_amount = amount
                                            print("Found IGST amount",igst_amount)
                            if "total" in filt2_above_processed_ngbr :
                                total_amount = filt2.loc[0,"text"]
                                try:
                                    if (isinstance(filt2_extracted_amount,float) or isinstance(filt2_extracted_amount,int)) and filt2_extracted_amount!=0:
                                        print("Found Total",total_amount)
                                        total = total_amount
                                        break
                                except:
                                    pass
                    
                        if cgst_present == 1:
                            row_dict["token_id"]=idx
                            row_dict["taxable"]=float(str(taxable).replace(",",""))
                            row_dict["cgst_percentage"]=float(str(cgst_percentage).replace("%",""))
                            row_dict["sgst_percentage"]=float(str(sgst_percentage).replace("%",""))
                            row_dict["cgst_amount"]=float(str(cgst_amount).replace(",",""))
                            row_dict["sgst_amount"]=float(str(sgst_amount).replace(",",""))
                            row_dict["total_tax_amount"]=float(str(total).replace(",",""))
                            if (cgst_percentage == 0) and (sgst_percentage == 0) and (cgst_amount == 0) and (sgst_amount == 0):
                                pass
                            else: 
                                final_list.append(row_dict)
                        elif cgst_present == 0:
                            row_dict["token_id"]=idx
                            row_dict["taxable"]=float(str(taxable).replace(",",""))
                            row_dict["igst_percentage"]=float(str(igst_percentage).replace("%",""))
                            row_dict["igst_amount"]=float(str(igst_amount).replace(",",""))
                            row_dict["total_tax_amount"]=float(str(total).replace(",",""))
                            if (igst_percentage == 0) and (igst_amount == 0):
                                pass
                            else:
                                final_list.append(row_dict)
                        
            print("List V1", final_list)
            if cgst_present == 1:
                cgst_list = [2.5,5,6,12,9,18,14,28]
                
                for i,item in enumerate(final_list):
                    cal_list = []
                    if item["cgst_percentage"] == 0 and item["cgst_amount"]!=0:
                        calculated_percentage = (float(item["cgst_amount"])/(float(item["taxable"])))*100
                        for val in cgst_list:
                            cal_list.append(abs(val-calculated_percentage))
                        calculated_percentage = cgst_list[cal_list.index(min(cal_list))]
                        if calculated_percentage in [5,12,18,28]:
                            calculated_percentage/=2
                        final_list[i]["cgst_percentage"] = calculated_percentage
                        final_list[i]["sgst_percentage"] = calculated_percentage
                    if item["cgst_percentage"] != 0 and item["cgst_amount"]==0:
                        final_list[i]["cgst_amount"] = float(item["taxable"])*float(item["cgst_percentage"])/100
                        final_list[i]["sgst_amount"] = final_list[i]["cgst_amount"]
                    if item["total_tax_amount"] == 0 and item["cgst_amount"] !=0:
                        final_list[i]["total_tax_amount"] = item["cgst_amount"] + item["sgst_amount"]
            if cgst_present ==0:
                igst_list = [5,12,18,28]
                
                for i,item in enumerate(final_list):
                    cal_list = []
                    if item["igst_percentage"] == 0 and item["igst_amount"]!=0:
                        calculated_percentage = (float(item["igst_amount"])/(float(item["taxable"])))*100
                        for val in igst_list:
                            cal_list.append(abs(val-calculated_percentage))
                        calculated_percentage = igst_list[cal_list.index(min(cal_list))]
                        final_list[i]["igst_percentage"] = calculated_percentage
                    if item["igst_percentage"] != 0 and item["igst_amount"]==0:
                        final_list[i]["igst_amount"] = float(item["taxable"])*float(item["igst_percentage"])/100
                    if item["total_tax_amount"] == 0 and item["igst_amount"] !=0:
                        final_list[i]["total_tax_amount"] = item["igst_amount"]
                        
            print("List V2", final_list)
            print("cgst present",cgst_present)
            return final_list,cgst_present
        except :
            print(traceback.print_exc())
            return final_list,cgst_present
    final_list,cgst_present = get_taxes_amounts_from_table_anchor_column(df)
    return final_list,cgst_present
def amountslablist(DF):
    try:
        Amountslab=[]
        a=list(DF['tableLineNo'].unique())
        for line in a:
            #print(line)
            threshold = 1
            gst_regex = r'5%|12%|18%|28%|2.5|6.0|9.0|14.0'
            dff =  DF[ (DF["tableLineNo"] == line)]
            if (dff.shape[0]>0):
                linse =' '.join(dff['text'].to_list())
                slab = re.findall(gst_regex, linse)
                #print(linse,slab)
                if len(slab)>0:
                    GST_slab = float((slab[0]).replace("%","").replace(" ","").replace(",","."))/100
                    #print(slab[0])
                    #GST_slab = float(slab[0])/100
                    dff1=dff[dff["extracted_amount"]>0]
                    A_values = list(dff1["extracted_amount"].unique())
                    #print(GST_slab,A_values,"ad")
                    A_values.sort(reverse=True)

                    flag=False
                    for value in A_values:
                        GST_amount = round(value * GST_slab, 2)
                        #print(GST_amount)
                        for etax in A_values:
                            if abs(etax - GST_amount) <= threshold:
                                if etax not in [2.5,5,6,12,9,18,14,28] and value>28 and etax>2.5:
                                    Amountslab.append([GST_slab,value,etax])
                                    #print("ok")
                                    flag=True
                        if flag:
                            break
        return Amountslab
    except:
        print("amountslablist",traceback.print_exc())
        return []

def convert_to_float_or_string(s):
    try:
        return float(s)/100
    except ValueError:
        return s

def amountslablistwithoutgstamountinline(DF):
    try:
        Amountslab=[]
        a=list(DF['tableLineNo'].unique())
        for line in a:
            #print(line)
            threshold = 2
            gst_regex = r'5%|12%|18%|28%|2.5|6.0|9.0|14.0'
            dff =  DF[ (DF["tableLineNo"] == line)]
            if (dff.shape[0]>0):
                linse =' '.join(dff['text'].to_list())
                slab = re.findall(gst_regex, linse)
                
                #GST_slab = convert_to_float_or_string((slab[0]).replace("%","").replace(" ","").replace(",",".").replace("-","."))
                if len(slab)>0 :
                    #print(linse,slab)
                    for si in list(set(slab)):
                        GST_slab = convert_to_float_or_string((si).replace("%","").replace(" ",".").replace(",",".").replace("-","."))
                        #GST_slab = float((slab[0]).replace("%","").replace(" ","").replace(",",".").replace("-","."))/100
                        #print(si,GST_slab)
                        #GST_slab = float(slab[0])/100
                        if isinstance(GST_slab, float) :
                            dff1=dff[dff["extracted_amount"]>0]
                            A_values = list(dff1["extracted_amount"].unique())
                            #print(GST_slab,A_values,"ad")
                            A_values.sort(reverse=True)
                            #print(A_values)
                            flag=False
                            for value in A_values:
                                if GST_slab in [0.06,0.025,0.09,0.14]:
                                    GST_amount = round(value * (GST_slab*2), 2)
                                else:
                                    GST_amount = round(value * (GST_slab), 2)
                                #print(GST_amount)
                                for etax in A_values:
                                    if abs(value + GST_amount-etax) <= threshold:
                                        if GST_amount not in [2.5,5,6,12,9,18,14,28] and value>28 and GST_amount>2.5:
                                            Amountslab.append([GST_slab,value,(GST_amount/2)])
                                            #print("ok",[GST_slab,value,GST_amount])
                                            flag=True
                                if flag:
                                    break
        #print("abcdf",Amountslab)
        return Amountslab
    except:
        print("amountslablist",traceback.print_exc())
        return []
def gstslablistdictionary(DF):
    try:
        Amountslab = amountslablist(DF)
        if len(Amountslab)==0:
            Amountslab=amountslablistwithoutgstamountinline(DF)
        #print(Amountslab)
        new_list = list(set(tuple(i) for i in Amountslab))
        Amountslab_new = [list(j) for j in new_list]
        #print(Amountslab_new)
        cgst = ['taxable', 'cgst_percentage', 'sgst_percentage', 'cgst_amount', 'sgst_amount']
        igst = ["taxable", "igst_percentage", "igst_amount"]
        result=[]
        for slabs in Amountslab_new:
            sub_result={}
            if ((slabs[0]==0.025) or (slabs[0]==0.06) or (slabs[0]==0.09) or (slabs[0]==0.14)):
                sub_result = {cgst[0]: slabs[1],
                    cgst[1]: slabs[0]*100,
                    cgst[2]: slabs[0]*100,
                    cgst[3]: slabs[2],
                    cgst[4]: slabs[2]
                }
                result.append(sub_result)
            elif ((slabs[0]==0.05) or (slabs[0]==0.12) or (slabs[0]==0.18) or (slabs[0]==0.28)):
                sub_result = {igst[0]: slabs[1],
                    igst[1]: slabs[0]*100,
                    igst[2]: slabs[2]
                }
                result.append(sub_result)
        return result
    except:
        print("exception in gstslablistdictionary")
        return[]

def calculateandassignslab(DF,final_prediction):
    final_prediction_copy =copy.deepcopy(final_prediction)
    try:
        Amountslab = amountslablist(DF)
        if len(Amountslab)==0:
            #Amountslab=amountslablistwithoutgstamountinline(DF)
            pass
        print(Amountslab)
        new_list = list(set(tuple(i) for i in Amountslab))
        Amountslab_new = [list(j) for j in new_list]
        print(Amountslab_new)
        gst2p5=gst6=gst9=gst14=gst5=gst12=gst18=gst28=subt5=subt12=subt18=subt28=0
        for i in Amountslab_new:
            if (i[0]==0.025):
                subt5=subt5 + i[1]
                gst2p5=gst2p5 + i[2]
                continue
            if (i[0]==0.06):
                subt12=subt12 + i[1]
                gst6=gst6 + i[2]
                continue
            if (i[0]==0.09):
                subt18=subt18 + i[1]
                gst9=gst9 + i[2]
                continue
            if (i[0]==0.14):
                subt28=subt28 + i[1]
                gst14=gst14 + i[2]
                continue
            if (i[0]==0.05):
                subt5=subt5 + i[1]
                gst5=gst5 + i[2]
                continue
            if (i[0]==0.12):
                subt12=subt12 + i[1]
                gst12=gst12 + i[2]
                continue
            if (i[0]==0.18):
                subt18=subt18 + i[1]
                gst18=gst18 + i[2]
                continue
            if (i[0]==0.28):
                subt28=subt28 + i[1]
                gst28=gst28 + i[2]
                continue
        print(gst2p5,gst6,gst9,gst14,gst5,gst12,gst18,gst28,subt5,subt12,subt18,subt28)
        if len(str(final_prediction['CGSTAmount_2.5%']['text']))==0:
            final_prediction['CGSTAmount_2.5%']['text'] = round(gst2p5,2)
        if len(str(final_prediction['CGSTAmount_6%']['text']))==0 :
            final_prediction['CGSTAmount_6%']['text'] = round(gst6,2)
        if len(str(final_prediction['CGSTAmount_9%']['text']))==0 :
            final_prediction['CGSTAmount_9%']['text'] = round(gst9,2)
        if len(str(final_prediction['CGSTAmount_14%']['text']))==0:
            final_prediction['CGSTAmount_14%']['text'] = round(gst14,2)
        if len(str(final_prediction['SGSTAmount_2.5%']['text']))==0:
            final_prediction['SGSTAmount_2.5%']['text'] = round(gst2p5,2)
        if len(str(final_prediction['SGSTAmount_6%']['text']))==0:
            final_prediction['SGSTAmount_6%']['text'] = round(gst6,2)
        if len(str(final_prediction['SGSTAmount_9%']['text']))==0:
            final_prediction['SGSTAmount_9%']['text'] = round(gst9,2)
        if len(str(final_prediction['SGSTAmount_14%']['text']))==0:
            final_prediction['SGSTAmount_14%']['text'] = round(gst14,2)
        if len(str(final_prediction['IGSTAmount_5%']['text']))==0:
            final_prediction['IGSTAmount_5%']['text'] = round(gst5,2)
        if len(str(final_prediction['IGSTAmount_12%']['text']))==0:
            final_prediction['IGSTAmount_12%']['text'] = round(gst12,2)
        if len(str(final_prediction['IGSTAmount_18%']['text']))==0:
            final_prediction['IGSTAmount_18%']['text'] = round(gst18,2)
        if len(str(final_prediction['IGSTAmount_28%']['text']))==0:
            final_prediction['IGSTAmount_28%']['text'] = round(gst28,2)
        if len(str(final_prediction['subTotal_5%']['text']))==0:
            final_prediction['subTotal_5%']['text'] = round(subt5,2)
        if len(str(final_prediction['subTotal_12%']['text']))==0:
            final_prediction['subTotal_12%']['text'] = round(subt12,2)
        if len(str(final_prediction['subTotal_18%']['text']))==0:
            final_prediction['subTotal_18%']['text'] = round(subt18,2)
        if len(str(final_prediction['subTotal_28%']['text']))==0:
            final_prediction['subTotal_28%']['text'] = round(subt28,2)
        return final_prediction
    except:
        print("calculateandassignslab",traceback.print_exc())
        return final_prediction_copy
if __name__ == "__main__":
    file_paths = {"AKSHAYAKALPA FARMS AND FOODS PRIVATE LIMITED":r"C:\Users\Admin\Downloads\ecd75cb0-a940-11ed-af45-1f967958f293_pred.csv",
                "ARAVI HOME & KITCHEN STORE":r"C:\Users\Admin\Downloads\23a7f4ca-a941-11ed-af45-1f967958f293_pred.csv",
                "BANWARI LAL ENTERPRISES - LKO":r"C:\Users\Admin\Downloads\38c2073c-a942-11ed-af45-1f967958f293_pred.csv",
                "Batter Chatter Private Limited":r"C:\Users\Admin\Downloads\6c6c6638-ac60-11ed-af58-1f967958f293_pred.csv",
                "BENGAL BEVERAGES PVT LTD-01":r"C:\Users\Admin\Downloads\e0a83e94-a942-11ed-af45-1f967958f293_pred.csv",
                "BENGALURU-NAKSHATRA AGENCIES":r"C:\Users\Admin\Downloads\171de406-a943-11ed-af45-1f967958f293_pred.csv",
                "BLR-RIMOX MARKETING":r"C:\Users\Admin\Downloads\b9322bda-a943-11ed-af45-1f967958f293_pred.csv",
                "BRITANNIA INDUSTRIES LIMITED (BISCUITS)":r"C:\Users\Admin\Downloads\25702e0a-a944-11ed-af45-1f967958f293_pred.csv",
                "BRITANNIA INDUSTRIES LIMITED (DIARY)":r"C:\Users\Admin\Downloads\a3fd9c12-a944-11ed-af45-1f967958f293_pred.csv",
                "CONGRUENCE TRADE & SERVICES PRIVATE LIMITED":r"C:\Users\Admin\Downloads\f9208ea6-a945-11ed-af45-1f967958f293_pred.csv"}

    df = pd.read_csv(file_paths["AKSHAYAKALPA FARMS AND FOODS PRIVATE LIMITED"])
    final_list,cgst_present = tax_slab_before_LI(df)
