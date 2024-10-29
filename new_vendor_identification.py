import pandas as pd
import TAPPconfig as cfg
pd.options.mode.chained_assignment = None

vendor_master_path = cfg.getVendorMasterData()
buyer_master_path = cfg.getBuyerMasterData()


def read_refernce_data(refrence_file_path):
	"""
	"""
	REFERENCE_DATA = pd.read_csv(refrence_file_path)
	return REFERENCE_DATA


def identify_new_vendors(DF, fetch_date):
	"""
	"""
	print("Identifying New Vendors for:", fetch_date)
	VENDOR_MASTER = read_refernce_data(vendor_master_path)
	ADDRESS_MASTER = read_refernce_data(buyer_master_path)
	# print(VENDOR_MASTER)
	refrence_vendors = list(VENDOR_MASTER["VENDOR_GSTIN"].unique())

	# Process Documents to identify new Vendors
	UNIQUE_VENDORS = DF[["VENDOR GSTIN", "VENDOR NAME"]].drop_duplicates()
	UNIQUE_VENDORS = UNIQUE_VENDORS[~UNIQUE_VENDORS["VENDOR GSTIN"].isna()]

	UNIQUE_VENDORS["vendor_data_exist"] = 0

	for idx, rows in UNIQUE_VENDORS.iterrows():
		vendor_gstin = rows["VENDOR GSTIN"]
		if vendor_gstin in refrence_vendors:
			UNIQUE_VENDORS.at[idx,'vendor_data_exist'] = 1

	NEW_VENDORS = UNIQUE_VENDORS.loc[UNIQUE_VENDORS["vendor_data_exist"] == 0]
	del NEW_VENDORS["vendor_data_exist"]

	NEW_VENDORS.to_csv("Reports/VendorMasterMissing_" + str(fetch_date) + ".csv", index=False)

	# Process Documents to identify new Buyer/Shipper GSTINs
	refrence_vendors = list(ADDRESS_MASTER["GSTIN"].unique())
	UNIQUE_ADDRESSES_1 = DF[["Billing GSTIN", "Billing Name"]].drop_duplicates()
	UNIQUE_ADDRESSES_1.rename(columns={'Billing GSTIN': 'GSTIN', 'Billing Name': 'NAME'}, inplace=True)

	UNIQUE_ADDRESSES_2 = DF[["Shipping GSTIN", "Shipping Name"]].drop_duplicates()
	UNIQUE_ADDRESSES_2.rename(columns={'Shipping GSTIN': 'GSTIN', 'Shipping Name': 'NAME'}, inplace=True)

	UNIQUE_ADDRESSES_3 = pd.concat([UNIQUE_ADDRESSES_1, UNIQUE_ADDRESSES_1],ignore_index=True)
	UNIQUE_ADDRESSES_3 = UNIQUE_ADDRESSES_3[["GSTIN", "NAME"]].drop_duplicates()
	UNIQUE_ADDRESSES_3 = UNIQUE_ADDRESSES_3.loc[~UNIQUE_ADDRESSES_3["GSTIN"].isna()]

	UNIQUE_ADDRESSES_3["data_exist"] = 0

	for idx, rows in UNIQUE_ADDRESSES_3.iterrows():
		gstin = rows["GSTIN"]
		if gstin in refrence_vendors:
			UNIQUE_ADDRESSES_3.at[idx,'data_exist'] = 1

	NEW_VENDORS = UNIQUE_ADDRESSES_3.loc[UNIQUE_ADDRESSES_3["data_exist"] == 0]
	del NEW_VENDORS["data_exist"]

	NEW_VENDORS.to_csv("Reports/BuyerMasterMissing_" + str(fetch_date) + ".csv", index=False)

	# print(NEW_VENDORS)
