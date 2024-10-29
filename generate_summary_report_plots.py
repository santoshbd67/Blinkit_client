import pandas as pd
import traceback
pd.options.mode.chained_assignment = None

import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
from random import shuffle
import numpy as np
sns.set_theme(style="whitegrid", palette="muted")

# import matplotlib.backends.backend_pdf
# pdf = matplotlib.backends.backend_pdf.PdfPages("output_14.pdf")

sla_states = ["Maharashtra", "Karnataka", "MISSING"]
extracted_fields = ['invoiceNumber', 'invoiceDate', 'billingGSTIN',
       'shippingGSTIN', 'vendorGSTIN', 'vendorName', 'billingName',
       'shippingName', 'subTotal', 'totalAmount', 'CGSTAmount', 'SGSTAmount',
       'IGSTAmount', 'additionalCessAmount', 'discountAmount',
       'vendorPAN', 'totalGSTAmount', 'CessAmount']

accuracy_fileds = ["invoiceDate", "totalAmount"]


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
    DF.loc[DF['Status'].isin(["REVIEW", "REVIEW_COMPLETED", "RPA_PROCESSED", "RPA_FAILED", "DELETED"]),
         'Status_'] = DF["Status"]
    
    return DF


def plot_graphs(DF, name_extender):
    """
    """
    num_rows = 22
    num_cols = 2
    def my_fmt(x):
        return '{:.2f}%\n({:.0f})'.format(x, total*x/100)
    
    colors = ['#C39BD3', '#F1C40F','#27AE60','#99A3A4', '#D98880', '#85C1E9']
    
    title = 'Graphical Report'

    if name_extender is not None:
        title = 'Graphical Report_' + str(name_extender)
    
    figure = plt.figure(figsize=(num_cols*7, num_rows*6))

    try:
        TEMP = DF.groupby(["Submit Date"])[["Document ID"]].count().reset_index().sort_values(["Submit Date"])
        plt.subplot(num_rows, num_cols, (1,2))
        sns.barplot(x="Submit Date", y="Document ID", data=TEMP, color="b")
        plt.ylabel("Document Count")
        plt.xticks(rotation=90)
        plt.title("Document Submitted by Date")
        
        TEMP = DF.groupby(["Submit Date"])[["Document ID"]].count().reset_index().sort_values(["Submit Date"])
        TEMP["Total Sum"] = TEMP["Document ID"].cumsum()
        ax = plt.subplot(num_rows, num_cols, (3,4))
        ax.plot(TEMP["Submit Date"], TEMP["Total Sum"], 'o-', color="blue")
        plt.ylabel("Cumulative Document Count")
        plt.xticks(rotation=90)
        plt.title("Cumulative Document Count by Date")
    except Exception as e:
        print("plot_graphs",traceback.print_exc())
        pass


    try:
        TEMP = DF.groupby(["Submit Day"])[["Document ID"]].count().reset_index()
        plt.subplot(num_rows, num_cols, 5)
        sns.barplot(x="Submit Day", y="Document ID", data=TEMP, color="m")
        plt.ylabel("Total Document Count")
        plt.title("Total Document Submitted by Day of Week")

        TEMP = DF.groupby(["Submit Hour"])[["Document ID"]].count().reset_index()
        total_days = len(list(DF["Submit Date"].unique()))
        TEMP["Average Count"] = TEMP["Document ID"]/total_days
        TEMP["Average Count"] = TEMP["Average Count"].astype(int)
        ax = plt.subplot(num_rows, num_cols, 6)
        sns.barplot(x="Submit Hour", y="Document ID", data=TEMP, color="g")
        plt.ylabel("Total Document Count")
        plt.title("Total Document Submitted by Hour of Day")
    except Exception as e:
        print("plot_graphs",traceback.print_exc())
        pass


    try:
        # Plot 1: Status Wise Distribution
        TEMP = DF.groupby(["Status_"])[["Document ID"]].count().reset_index()
        data = list(TEMP["Document ID"])
        labels = list(TEMP["Status_"])
        total = sum(data)
        
        ax = plt.subplot(num_rows, num_cols, 7)
        ax.set_title("Status wise Distribution: Overall")
        shuffle(colors)
        plt.pie(data, labels = labels, autopct=my_fmt, startangle=90,
                colors=colors, textprops={'fontsize': 12})

        #draw circle
        centre_circle = plt.Circle((0,0), 0.50,fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)

    except Exception as e:
        print("plot_graphs",traceback.print_exc())
        pass


    try:
        # Plot 1: Status Wise Distribution
        TEMP = DF.groupby(["STP System"])[["Document ID"]].count().reset_index()
        data = list(TEMP["Document ID"])
        labels = list(TEMP["STP System"])
        total = sum(data)
        
        ax = plt.subplot(num_rows, num_cols, 8)
        ax.set_title("STP: Overall")
        shuffle(colors)
        plt.pie(data, labels = labels, autopct=my_fmt, startangle=90,
                colors=colors, textprops={'fontsize': 12})

        #draw circle
        centre_circle = plt.Circle((0,0), 0.50,fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
    except Exception as e:
        print("plot_graphs",traceback.print_exc())
        pass


    try:
        ### Draw STP and ACE % Line
        E = form_date_wise_sla_track_df(DF)


        ax = plt.subplot(num_rows, num_cols, (9,10))
        ax.set_title("STP and ACE % by Date")
        sns.lineplot(x="Submit Date", y="STP %", data=E, marker='o',
                    linewidth = 2, label="STP %")
        sns.lineplot(x="Submit Date", y="STP % (Cumulative)", data=E, marker='o',
                    linewidth = 2, label="STP % (Cumulative)")
        sns.lineplot(x="Submit Date", y="ACE %", data=E, marker='o',
                    linewidth = 2, label="ACE %")
        sns.lineplot(x="Submit Date", y="ACE % (Cumulative)", data=E, marker='o',
                    linewidth = 2, label="ACE % (Cumulative)")

        # Plot trend lines
        ax.axhline(y=E.iloc[-1]["STP % (Cumulative)"], ls='--', label="Mean STP %", linewidth=2, color='r',
           xmin=0.01, xmax=0.99)
        ax.axhline(y=E.iloc[-1]["ACE % (Cumulative)"], ls='--', label="Mean ACE %", linewidth=2, color='green',
           xmin=0.01, xmax=0.99)
        
        
        plt.ylabel("Percentage")
        plt.legend()
        plt.xticks(rotation=90)
    except Exception as e:
        print("plot_graphs",traceback.print_exc())
        pass


    try:
        A = DF.loc[~DF["Total Review Time"].isna()]
        A = A.loc[A["User"].str.contains("taoautomation.com")]
        TEMP1 = A.groupby(["Submit Date"])[["Document ID"]].count().reset_index().sort_values(["Submit Date"])
        TEMP2 = A.groupby(["Submit Date"])[["Total Review Time"]].sum().reset_index().sort_values(["Submit Date"])

        TEMP3 = pd.merge(TEMP1, TEMP2, on=["Submit Date"], how="outer")
        TEMP3["Average Review Time"] = TEMP3["Total Review Time"]/TEMP3["Document ID"] 

        TEMP1["source"] = "Total Documents"
        TEMP1.rename(columns={'Document ID': 'Value'}, inplace=True)
        
        
        TEMP2["source"] = "Total Review Time"
        TEMP2.rename(columns={'Total Review Time': 'Value'}, inplace=True)

        TEMP = pd.concat([TEMP1, TEMP2],ignore_index=True)

        # Plot
        ax1 = plt.subplot(num_rows, num_cols, (11,12))

        palette ={"Total Documents": "#8E44AD", "Total Review Time": "#52BE80"}
        sns.lineplot(x= "Submit Date", y="Value", hue="source", data=TEMP, ax=ax1, palette=palette)

        # Create a second y-axis with the scaled ticks
        ax1.set_ylabel('Documents Reviewed/Total Review Time')
        ax2 = ax1.twinx()

        legend1, = ax2.plot(TEMP3["Submit Date"], TEMP3["Average Review Time"], 'o-', color="red", label="Average Review Time")

        first_legend = ax2.legend(handles =[legend1], loc ='upper center') 
        ax2.add_artist(first_legend)

        ax1.set_xticklabels(ax1.get_xticklabels(), rotation = 90)
        ax1.set_title("Document Count/Total Review Time by Submit Date")
    except Exception as e:
        print("plot_graphs",traceback.print_exc())
        pass


    try:
        A = DF.loc[~DF["Total Review Time"].isna()]
        TEMP1 = A.groupby(["Submit Date"])[["Document ID"]].count().reset_index().sort_values(["Submit Date"])
        TEMP1.rename(columns={'Document ID': 'Manually Reviewed'}, inplace=True)

        B = DF.loc[(DF["Status"] != "FAILED")]
        TEMP2 = B.groupby(["Submit Date"])[["Document ID"]].count().reset_index().sort_values(["Submit Date"])
        TEMP2.rename(columns={'Document ID': 'Total Document'}, inplace=True)


        TEMP3 = pd.merge(TEMP1, TEMP2, on=["Submit Date"], how="outer")

        TEMP3["Fraction Manually Reviewed"] = TEMP3["Manually Reviewed"]/TEMP3["Total Document"] 

        # Plot
        ax = plt.subplot(num_rows, num_cols, (13,14))

        legend1, = ax.plot(TEMP3["Submit Date"], TEMP3["Fraction Manually Reviewed"], 'o-', color="blue",
            label="Fraction: Manually Reviewed")

        first_legend = ax.legend(handles =[legend1], loc ='upper right') 
        ax.add_artist(first_legend)

        ax.set_xticklabels(ax1.get_xticklabels(), rotation = 90)
        ax.set_title("Fraction of Manually Reviewed Documents")
    except Exception as e:
        print("plot_graphs",traceback.print_exc())
        pass


    try:
        A = DF.loc[~DF["Total Review Time"].isna()]
        A = A.loc[A["User"].str.contains("taoautomation.com")]
        A["Review Hour"] = A["Review Completion/Deletion Time"].dt.hour

        B = A.groupby(["Submit Date",  "Review Hour"])["Document ID"].count().reset_index().sort_values(["Submit Date"])

        l = list(B["Submit Date"].unique())
        l.sort(reverse=True)
        picked_dates = l[0:15]

        C = B.loc[B["Submit Date"].isin(picked_dates)]
        D = C.pivot(index="Submit Date", columns="Review Hour", values="Document ID")
        D.fillna(0, inplace=True)
        D = D.astype(int)

        # Plot
        ax = plt.subplot(num_rows, num_cols, (15,16))
        sns.boxplot(x=B["Review Hour"], y=B["Document ID"])
        plt.ylabel("Document Count")
        plt.xticks(rotation=90)
        ax.set_title("Reviewed Documents by Hour of Day")


        ax = plt.subplot(num_rows, num_cols, (17,18))
        sns.heatmap(D, annot=True, cmap="Greens", fmt="d")
        ax.set_title("Reviewed Documents by Date and Hour of Day")
    except Exception as e:
        print("plot_graphs",traceback.print_exc())
        pass


    try:
        A = DF.loc[~DF["Total Review Time"].isna()]
        TEMP1 = A.groupby(["User"])[["Document ID"]].count().reset_index()
        TEMP1["source"] = "Total Documents"
        TEMP1.rename(columns={'Document ID': 'Value'}, inplace=True)
        
        TEMP2 = A.groupby(["User"])[["Total Review Time"]].mean().reset_index()
        TEMP2["source"] = "Average Review Time"
        TEMP2.rename(columns={'Total Review Time': 'Value'}, inplace=True)

        TEMP = pd.concat([TEMP1, TEMP2],ignore_index=True)

        # Scale the data, just a simple example of how you might determine the scaling
        mask = TEMP["source"].isin(["Average Review Time"])
        scale = int(TEMP[~mask]["Value"].mean()
                    /TEMP[mask]["Value"].mean())
        TEMP.loc[mask, 'Value'] = TEMP.loc[mask, 'Value']*scale

        # Plot
        palette ={"Total Documents": "#8E44AD", "Average Review Time": "#52BE80"}
        ax1 = plt.subplot(num_rows, num_cols, (19,20))
        sns.barplot(x= "User", y="Value", hue="source", data=TEMP, ax=ax1, palette=palette)

        # Create a second y-axis with the scaled ticks
        ax1.set_ylabel('Total Document Reviewed')
        ax2 = ax1.twinx()

        # Ensure ticks occur at the same positions, then modify labels
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_yticklabels(np.round(ax1.get_yticks()/scale,1))
        ax2.set_ylabel('Average Review Time')

        ax1.set_xticklabels(ax1.get_xticklabels(), rotation = 90)
        ax1.set_title("Reviewer Performance: Total Reviewd Documents & Average Review Time")
    except Exception as e:
        print("plot_graphs",traceback.print_exc())
        pass


    try:
        start_cell = 21
        A = DF.loc[~DF["Total Review Time"].isna()]
        A = A.loc[A["User"].str.contains("taoautomation.com")]
        A["Review Hour"] = A["Review Completion/Deletion Time"].dt.hour

        DDD = A.groupby(["User"])["Document ID"].count().reset_index().sort_values(["Document ID"], ascending=False)
        unique_reviewers = list(DDD["User"].unique())
        picked_reviewers = unique_reviewers[0:12]
        print(picked_reviewers)

        for reviewer_ in picked_reviewers:
            B = A.loc[A["User"] == reviewer_]
            C = B.groupby(["Submit Date",  "Review Hour"])["Document ID"].count().reset_index().sort_values(["Submit Date"])

            l = list(B["Submit Date"].unique())
            l.sort(reverse=True)
            picked_dates = l[0:15]

            D = C.loc[C["Submit Date"].isin(picked_dates)]
            E = D.pivot(index="Submit Date", columns="Review Hour", values="Document ID")
            E.fillna(0, inplace=True)
            E = E.astype(int)

            # Plot
            ax = plt.subplot(num_rows, num_cols, (start_cell, start_cell+1))
            sns.heatmap(E, annot=True, cmap="Blues", fmt="d")
            ax.set_title("Reviewer Date Wise Stats:" + str(reviewer_))
            start_cell += 2
        
    except Exception as e:
        print("plot_graphs",traceback.print_exc())
        pass


    plt.tight_layout()
    figure.savefig("Reports/" + title + ".jpg")


def stp_impact_non_stp_ace(DF):
       """
       """
       total_sla_count = DF.loc[DF["SLA_flag"] == 1].shape[0]
       A = DF.loc[DF["SLA_flag"] == 1].groupby(["VENDOR NAME",
                                            "Billing State"])[["Document ID"]].count().reset_index()
       B = DF.loc[(DF["SLA_flag"] == 1) & (DF["STP System"] == True)].groupby(["VENDOR NAME",
                                            "Billing State"])[["Document ID"]].count().reset_index()

       C = DF.loc[(DF["SLA_flag"] == 1) & (DF["ACE"] == "YES")].groupby(["VENDOR NAME",
                                            "Billing State"])[["Document ID"]].count().reset_index()

       M = DF.loc[(DF["SLA_flag"] == 1) & 
                  (DF["Status"] == "REVIEW_COMPLETED")].groupby(["VENDOR NAME",
                                            "Billing State"])[["Document ID"]].count().reset_index()


       A.rename(columns={'Document ID': 'Count'}, inplace=True)

       B.rename(columns={'Document ID': 'STP Count'}, inplace=True)

       C.rename(columns={'Document ID': 'ACE Count'}, inplace=True)

       M.rename(columns={'Document ID': 'RC Count'}, inplace=True)
       # C.sort_values(["RC Count"], ascending=False, inplace=True)

       D = pd.merge(A, B, on=["VENDOR NAME", "Billing State"], how="outer")
       D = pd.merge(D, C, on=["VENDOR NAME", "Billing State"], how="outer")
       D = pd.merge(D, M, on=["VENDOR NAME", "Billing State"], how="outer")

       D.sort_values(["Count"], ascending=False, inplace=True)
       D.fillna(0, inplace=True)
       D["Pending REVIEW Count"] = D["Count"] - D["RC Count"]
       D["Percentage Overall Count"] = (D["Count"]/total_sla_count)*100
       TEMP = D.loc[D["Percentage Overall Count"] > 1]
       del TEMP["Percentage Overall Count"]

       TEMP["Extrapolated ACE Count"] = (TEMP["ACE Count"]*TEMP["Count"]/TEMP["RC Count"])
       TEMP["Extrapolated ACE Count"] = ((TEMP["ACE Count"] - TEMP["STP Count"])
                                         /(TEMP["RC Count"] - TEMP["STP Count"])*TEMP["Pending REVIEW Count"])+TEMP["ACE Count"]
       TEMP["Extrapolated ACE Count"].fillna(0, inplace=True)
       TEMP["Extrapolated ACE Count"] = TEMP["Extrapolated ACE Count"].astype(int)
       TEMP["STP Impact"] = (((TEMP["Extrapolated ACE Count"] - 
                              TEMP["STP Count"])*100)/total_sla_count).round(2)

       return TEMP


def form_date_wise_sla_track_df(DF, state = ""):
    """
    """
    SLA = DF.copy()
    if state != "":
        SLA = SLA.loc[SLA["Billing State"] == state]
    
    A = SLA.groupby(["Submit Date"])[["Document ID"]].count().reset_index()
    A.rename(columns={'Document ID': 'Total Count'}, inplace=True)
    B = SLA.loc[SLA["STP System"] == True].groupby(["Submit Date"])[["Document ID"]].count().reset_index()
    B.rename(columns={'Document ID': 'STP Count'}, inplace=True)
    C = pd.merge(A, B, on="Submit Date", how="outer")
    C["Cumulative Total Count"] = C["Total Count"].cumsum()
    C["Cumulative STP Count"] = C["STP Count"].cumsum()
    C["STP %"] = ((C["STP Count"]/C["Total Count"])*100).round(2)
    C["STP % (Cumulative)"] = ((C["Cumulative STP Count"]
                                /C["Cumulative Total Count"])*100).round(2)
    
    A = SLA.loc[SLA["ACE"] != "Not Applicable"].groupby(["Submit Date"])[["Document ID"]].count().reset_index()
    B = SLA.loc[SLA["ACE"] == "YES"].groupby(["Submit Date"])[["Document ID"]].count().reset_index()
    A.rename(columns={'Document ID': 'Total Count ACE Applicable'}, inplace=True)
    B.rename(columns={'Document ID': 'ACE Count'}, inplace=True)
    D = pd.merge(A, B, on="Submit Date", how="outer")
    D["Total Count Cum ACE Applicable"] = D["Total Count ACE Applicable"].cumsum()
    D["Cumulative ACE Count"] = D["ACE Count"].cumsum()
    D["ACE %"] = ((D["ACE Count"]/D["Total Count ACE Applicable"])*100).round(2)
    D["ACE % (Cumulative)"] = ((D["Cumulative ACE Count"]
                                /D["Total Count Cum ACE Applicable"])*100).round(2)
    
    E = pd.merge(C, D, on="Submit Date", how="outer")
    
    return E


def get_top_vendors(DF):
       """
       """
       # Top 15 Vendors
       A = DF.groupby(["VENDOR NAME"])[["Document ID"]].count().reset_index()
       A.rename(columns={'Document ID': 'Count'}, inplace=True)
       A.sort_values(["Count"], ascending=False, inplace=True)
       top_15_vendors = list(A.head(15)["VENDOR NAME"])

       B = DF.loc[DF["VENDOR NAME"].isin(top_15_vendors)].groupby(["Submit Date",
                                                            "VENDOR NAME"])[["Document ID"]].count().reset_index()
       C = B.pivot(index = "Submit Date", columns = "VENDOR NAME", values = "Document ID")
       C = C[top_15_vendors]
       C.fillna(0, inplace=True)
       C[top_15_vendors] = C[top_15_vendors].astype(int)
       return C


def form_date_wise_filed_level_accuracy(DF):
    """
    """
    SLA = DF.copy()
    SLA = SLA.loc[(SLA["Status"] == "REVIEW_COMPLETED") | (SLA["Status"].str.contains("RPA"))]

    
    A = SLA.groupby(["Submit Date"])[["Document ID"]].count().reset_index()
    A.rename(columns={'Document ID': 'Total Count'}, inplace=True)

    for field in accuracy_fileds:
        TEMP = SLA.loc[SLA[field] == "OK"]
        B = TEMP.groupby(["Submit Date"])[[field]].count().reset_index()
        A = pd.merge(A, B, on="Submit Date", how="outer")
        # print(A)

    for field in accuracy_fileds:
        A[field] = ((A[field]/A["Total Count"])*100).round(2)

    # print(A)
    
    return A

    
def form_reviewer_stats(DF):
    """
    """
    SLA = DF.copy()
    SLA = SLA.loc[(SLA["Status"] == "REVIEW_COMPLETED") | (SLA["Status"].str.contains("RPA"))]

    SLA = SLA.loc[SLA["User"] != "DUMMY_USER"]

    TEMP = SLA.groupby(["User", "Submit Date"]).agg({'Document ID':['count'], 
                         'Total Review Time':'mean'})

    return TEMP



def generate_summary_report(DF, name_extender):
    """
    """
    summary_report_file_name = 'Reports/Summary_Report.xlsx'

    if name_extender is not None:
        summary_report_file_name = 'Reports/Summary_Report_' + str(name_extender) + '.xlsx'

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
    worksheet.write_string(start_row, start_col, "OVERALL STATS", section_header_format)
    start_row = start_row + 1
    # Total Count
    worksheet.write_string(start_row, start_col, "TOTAL DOCUMENTS", table_header_format)
    start_col = 1
    worksheet.write_string(start_row, start_col, str(DF.shape[0]), table_header_format)
    start_col = 0
    # Page Count
    start_row = start_row + 1
    worksheet.write_string(start_row, start_col, "TOTAL PAGES", table_header_format)
    start_col = 1
    worksheet.write_string(start_row, start_col, str(int(DF["Pages"].sum())), table_header_format)
    start_col = 0
    # Unique Vendors
    start_row = start_row + 1
    worksheet.write_string(start_row, start_col, "UNIQUE VENDORS", table_header_format)
    start_col = 1
    worksheet.write_string(start_row, start_col, str(int(len(list(DF["VENDOR NAME"].unique())))), table_header_format)
    start_col = 0
    # STP Count
    start_row = start_row + 1
    worksheet.write_string(start_row, start_col, "STP Count", table_header_format)
    start_col = 1
    worksheet.write_string(start_row, start_col, str(int(DF.loc[DF["STP System"] == 1].shape[0])), table_header_format)
    start_col = 0
    # ACE Count
    start_row = start_row + 1
    worksheet.write_string(start_row, start_col, "ACE Count", table_header_format)
    start_col = 1
    worksheet.write_string(start_row, start_col, str(int(DF.loc[DF["ACE"] == "YES"].shape[0])), table_header_format)
    

    unique_doc_type = list(DF["Document Type"].unique())
    for d in unique_doc_type:
        TEMP = DF.loc[DF["Document Type"] == d]
        start_col = 0
        start_row = start_row + 2
        worksheet.write_string(start_row, start_col, "OVERALL STATS:" + str(d), section_header_format)
        # Total Count
        start_row = start_row + 1
        worksheet.write_string(start_row, start_col, "TOTAL DOCUMENTS", table_header_format)
        start_col = 1
        worksheet.write_string(start_row, start_col, str(TEMP.shape[0]), table_header_format)
        start_col = 0
        # Page Count
        start_row = start_row + 1
        worksheet.write_string(start_row, start_col, "TOTAL PAGES", table_header_format)
        start_col = 1
        worksheet.write_string(start_row, start_col, str(int(TEMP["Pages"].sum())), table_header_format)
        start_col = 0
        # Unique Vendors
        start_row = start_row + 1
        worksheet.write_string(start_row, start_col, "UNIQUE VENDORS", table_header_format)
        start_col = 1
        worksheet.write_string(start_row, start_col, str(int(len(list(TEMP["VENDOR NAME"].unique())))), table_header_format)
        start_col = 0
        # STP Count
        start_row = start_row + 1
        worksheet.write_string(start_row, start_col, "STP Count", table_header_format)
        start_col = 1
        worksheet.write_string(start_row, start_col, str(int(TEMP.loc[TEMP["STP System"] == 1].shape[0])), table_header_format)
        start_col = 0
        # ACE Count
        start_row = start_row + 1
        worksheet.write_string(start_row, start_col, "ACE Count", table_header_format)
        start_col = 1
        worksheet.write_string(start_row, start_col, str(int(TEMP.loc[TEMP["ACE"] == "YES"].shape[0])), table_header_format)
        start_col = 0
    

    start_row = start_row + 2
    
    # Generate Status Wise Count
    TEMP = DF.groupby(["Status"])[["Document ID"]].count().reset_index()
    TEMP.loc[len(TEMP.index)] = ['Total', TEMP["Document ID"].sum()]
    TEMP.rename(columns={'Document ID': 'Count'}, inplace=True)
    TEMP.name = "Status Wise Document Count"
    
    worksheet.write_string(start_row, start_col, TEMP.name, table_header_format)
    start_row = start_row + 1
    TEMP.to_excel(writer,sheet_name='Result',
                            startrow=start_row , startcol=start_col, index=False)
    start_row = start_row + TEMP.shape[0] + 4
    
    # Generate Date Wise Count
    TEMP = DF.groupby(["Submit Date"])[["Document ID"]].count().reset_index()
    TEMP["Cumulative Count"] = TEMP["Document ID"].cumsum()
    TEMP.rename(columns={'Document ID': 'Count'}, inplace=True)
    A = DF.loc[DF["STP System"] == 1].groupby(["Submit Date"])[["Document ID"]].count().reset_index()
    A.rename(columns={'Document ID': 'STP Count'}, inplace=True)
    TEMP = pd.merge(TEMP, A, on=["Submit Date"], how="outer")
    A = DF.loc[DF["ACE"] == "YES"].groupby(["Submit Date"])[["Document ID"]].count().reset_index()
    A.rename(columns={'Document ID': 'ACE Count'}, inplace=True)
    TEMP = pd.merge(TEMP, A, on=["Submit Date"], how="outer")
    list_status = list(DF["Status"].unique())
    for s in list_status:
       A = DF.loc[DF["Status"] == s]
       B = A.groupby(["Submit Date"])[["Document ID"]].count().reset_index()
       B.rename(columns={'Document ID': s}, inplace=True)
       TEMP = pd.merge(TEMP, B, on=["Submit Date"], how="outer")
       TEMP.fillna(0, inplace=True)
       TEMP[s] = TEMP[s].astype(int) 
    
    TEMP.name = "Date Wise Document Count"
    
    worksheet.write_string(start_row, start_col, TEMP.name, table_header_format)
    # Shift to next Row
    start_row = start_row + 1
    TEMP.to_excel(writer,sheet_name='Result',
                            startrow=start_row , startcol=start_col, index=False)
    start_row = start_row + TEMP.shape[0] + 4
    

    # Top 15 Vendors
    TEMP = get_top_vendors(DF)
    TEMP.name = "Top 15 Vendors: Documents uploaded by Date"
    worksheet.write_string(start_row, start_col, TEMP.name, table_header_format)
    start_row = start_row + 1
    TEMP.to_excel(writer,sheet_name='Result',
                            startrow=start_row, startcol=start_col, index=True)
    start_row = start_row + TEMP.shape[0] + 4
    

    # Generate Date Wise STP Numbers: Overall
    TEMP = form_date_wise_sla_track_df(DF)
    
    TEMP.name = "Date Wise STP and ACE Stats"
    worksheet.write_string(start_row, start_col, TEMP.name, table_header_format)
    start_row = start_row + 1
    worksheet.write_string(start_row, start_col, "ACE % is calculated with REVIEW_COMPLETED as base")
    start_row = start_row + 1
    TEMP.to_excel(writer,sheet_name='Result',
                            startrow=start_row, startcol=start_col, index=False)
    start_row = start_row + TEMP.shape[0] + 4


    # Populate Date Wise Field Level Accuracy
    TEMP = form_date_wise_filed_level_accuracy(DF)
    
    TEMP.name = "Date Wise Field Level Accuracy"
    worksheet.write_string(start_row, start_col, TEMP.name, table_header_format)
    start_row = start_row + 1
    worksheet.write_string(start_row, start_col, "Field Level Accuracy is calculated as REVIEW_COMPLETED/RPA Documents as base")
    start_row = start_row + 1
    TEMP.to_excel(writer,sheet_name='Result',
                            startrow=start_row, startcol=start_col, index=False)
    start_row = start_row + TEMP.shape[0] + 4


    # Identify prominent Vendors
    TEMP = identify_actionable_vendors(DF, "NON_STP_ACE")
    A = DF.groupby(["VENDOR NAME"])[["Document ID"]].count().reset_index()
    A.rename(columns={'Document ID': 'Total Doc Count'}, inplace=True)
    TEMP = pd.merge(A, TEMP, on=["VENDOR NAME"], how="right")
    TEMP.name = "NON STP but ACE"
    
    worksheet.write_string(start_row, start_col, TEMP.name, table_header_format)
    start_row = start_row + 1
    TEMP.to_excel(writer,sheet_name='Result',
                            startrow=start_row, startcol=start_col, index=False)
    start_row = start_row + TEMP.shape[0] + 2
    
    TEMP = identify_actionable_vendors(DF, "NON_STP_NON_ACE")
    A = DF.groupby(["VENDOR NAME"])[["Document ID"]].count().reset_index()
    A.rename(columns={'Document ID': 'Total Doc Count'}, inplace=True)
    TEMP = pd.merge(A, TEMP, on=["VENDOR NAME"], how="right")
    TEMP.name = "NON STP and NON ACE"
    
    worksheet.write_string(start_row, start_col, TEMP.name, table_header_format)
    start_row = start_row + 1
    TEMP.to_excel(writer,sheet_name='Result',
                            startrow=start_row, startcol=start_col, index=False)
    start_row = start_row + TEMP.shape[0] + 2
    
    TEMP = identify_actionable_vendors(DF, "NON_STP_NA_ACE")
    A = DF.groupby(["VENDOR NAME"])[["Document ID"]].count().reset_index()
    A.rename(columns={'Document ID': 'Total Doc Count'}, inplace=True)
    TEMP = pd.merge(A, TEMP, on=["VENDOR NAME"], how="right")
    TEMP.name = "NON STP and ACE NA (Documents Not Reviewed)"
    
    worksheet.write_string(start_row, start_col, TEMP.name, table_header_format)
    start_row = start_row + 1
    TEMP.to_excel(writer,sheet_name='Result',
                            startrow=start_row, startcol=start_col, index=False)
    start_row = start_row + TEMP.shape[0] + 4


    # Populate Reviewer Stats
    TEMP = form_reviewer_stats(DF)
    
    TEMP.name = "Reviewer Stats"
    worksheet.write_string(start_row, start_col, TEMP.name, table_header_format)
    start_row = start_row + 1
    TEMP.to_excel(writer,sheet_name='Result',
                            startrow=start_row, startcol=start_col, index=True)
    start_row = start_row + TEMP.shape[0] + 4

    image_title = "Reports/Graphical Report.jpg"
    if name_extender is not None:
        image_title = 'Reports/Graphical Report_' + str(name_extender) + '.jpg'

    worksheet.insert_image('S1', image_title)
    writer.close()


def identify_actionable_vendors(DF, type_, state="Overall"):
    """
    """
    TEMP = None
    if type_ == "NON_STP_ACE":
        TEMP = DF.loc[(DF["ACE"] == "YES") & (DF["STP System"] == False)]
        if state != "Overall":
            TEMP = TEMP.loc[TEMP["Billing State"] == state]
    elif type_ == "NON_STP_NON_ACE":
        TEMP = DF.loc[(DF["ACE"] == "NO") & (DF["STP System"] == False)]
        if state != "Overall":
            TEMP = TEMP.loc[TEMP["Billing State"] == state]
    elif type_ == "NON_STP_NA_ACE":
        TEMP = DF.loc[(DF["ACE"] == "Not Applicable") & (DF["STP System"] == False)]
        if state != "Overall":
            TEMP = TEMP.loc[TEMP["Billing State"] == state]
    
    if (TEMP is not None) & (type_ != "NON_STP_NON_ACE"):
        A = TEMP.groupby(["VENDOR NAME"])[["Document ID"]].count().reset_index()
        A.rename(columns={'Document ID': 'Count'}, inplace=True)
        A.sort_values(["Count"], ascending=False, inplace=True)
        A = A.loc[A["Count"] >= 1]
        A["Comment"] = ""
        A["Action"] = ""
        return A
    if (TEMP is not None) & (type_ == "NON_STP_NON_ACE"):
        A = TEMP.groupby(["VENDOR NAME"])[["Document ID"]].count().reset_index()
        A.rename(columns={'Document ID': 'Count'}, inplace=True)
        A.sort_values(["Count"], ascending=False, inplace=True)
        A = A.loc[A["Count"] >= 1]
        A["Comment"] = ""
        A["Missed"] = ""
        A["Incorrect"] = ""
        A["Action"] = ""
        for idx, row in A.iterrows():
            v = row["VENDOR NAME"]
            B = TEMP.loc[TEMP["VENDOR NAME"] == v]
            l_missed = []
            l_incorrect = []
            for c in extracted_fields:
                try:
                    C = B.groupby([c])[["Document ID"]].count().reset_index()
                    C.rename(columns={'Document ID': 'Count'}, inplace=True)
                    C = C.loc[C[c] != "OK"]
                    if C.shape[0] > 0:
                        if "Missed" == C.iloc[0][c]:
                            l_missed.append({c: C.iloc[0]["Count"]})
                        elif "Incorrect" == C.iloc[0][c]:
                            l_incorrect.append({c: C.iloc[0]["Count"]})
                except Exception as e:
                    pass
                    continue
            if len(l_missed) > 0:
                A.at[idx, 'Missed'] = l_missed
            if len(l_incorrect) > 0:
                A.at[idx, 'Incorrect'] = l_incorrect
    
        return A
    
    return None

def generate_report(DF, start_date = None):
    """
    """
    print("Generating Reports for Start Date:", start_date)
    DF = derive_columns(DF)
    if start_date is not None:
        DF = DF.loc[DF["Submit Date"] >= start_date]
    try:
        plot_graphs(DF, start_date)
        generate_summary_report(DF, start_date)
    except Exception as e:
        print("generate_report",traceback.print_exc())
        pass
