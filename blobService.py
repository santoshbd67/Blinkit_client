from azure.storage.blob import BlobServiceClient, generate_account_sas, ResourceTypes, AccountSasPermissions
from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretClient
from datetime import datetime, timedelta

import os
import traceback
import base64
import TAPPconfig as config

class AzureBlobStorage:
    def __init__(self):
        self.sub_id = config.getSubscriberId()
        self.blob_service_client = self.__get_blob_service_client()

    def __get_blob_service_client(self):
        try:
            access_key, account_name, container_name = self.__get_blob_account_details()
            if access_key and account_name:
                connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={access_key};EndpointSuffix=core.windows.net"
                return BlobServiceClient.from_connection_string(connection_string)
            return None
        except:
            print("Failed to get blob service client:", traceback.print_exc())
            return None

    def __get_blob_account_details(self):
        try:
            ## 17 July 2023 Getting access key from config
            """keyVaultName = config.getSecretVault()
            KVUri = f"https://{keyVaultName}.vault.azure.net"
            tenant_id, client_secret, client_id = self.__get_service_principal()
            credential = ClientSecretCredential(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret
            )
            client = SecretClient(vault_url=KVUri, credential=credential)
            access_key = client.get_secret('access').value
            account_name = client.get_secret('account').value"""
            container_name = self.sub_id
            from cryptography.fernet import Fernet
            # Initialize the Fernet cipher with the encryption key
            encrypted_access_key = config.get_blob_access_key()
            key = config.getKEY()
            cipher = Fernet(key)
            decrypted_access_key = cipher.decrypt(encrypted_access_key.encode())
            access_key = decrypted_access_key.decode('utf-8')
            account_name = config.get_account_name()
            return access_key, account_name, container_name
        except:
            print("Failed to get blob account details:", traceback.print_exc())
            return None, None, None

    def __get_service_principal(self):
        encodedString = config.getServicePrinciple()
        base64_string_byte = encodedString.encode("UTF-8")
        original_string_bytes = base64.b64decode(base64_string_byte)
        original_string = original_string_bytes.decode("UTF-8")
        sp_list = original_string.split(';')
        tenant_id, client_secret, client_id = sp_list[0], sp_list[1], sp_list[2]
        return tenant_id, client_secret, client_id

    def __getMacAddress():
        """
        Returns
        -------
        get_mac : st
            MAC ID.
        """
        from uuid import getnode as get_mac
        return get_mac()

    
    def generate_sas_token(self, activity:str, IP:str=None)-> tuple:
        try:
            access_key, account_name, container_name = self.__get_blob_account_details()
            time_mins = 10
            if access_key and account_name:
                account_url = f"https://{account_name}.blob.core.windows.net"
                permissions = {
                    "upload": AccountSasPermissions(write=True),
                    "download": AccountSasPermissions(read=True),
                    "delete": AccountSasPermissions(delete=True)
                }
                permission = permissions.get(activity.lower())
                if permission:
                    sas_token = generate_account_sas(
                        account_name=account_name,
                        account_key=access_key,
                        resource_types=ResourceTypes(object=True),
                        ip=IP,
                        permission=permission,
                        expiry=datetime.utcnow() + timedelta(minutes=time_mins)
                    )
                else:
                    sas_token = None
                    account_url = None
                return account_url, sas_token, container_name
            return None, None, None
        except Exception as e:
            print("Failure while SAS token generation:", e)
            return None, None, None

    def download_file(self, blob_uri:str, local_path:str=None, IP:str=None)-> bool:
        try:
            account_url, credential, container_name = self.__generate_sas_token("download", IP)
            print(account_url,credential,container_name)
            if account_url is None:
                return False
            # blob_uri_with_sas = f"{account_url}/{container_name}/{blob_uri}?{credential}"
            blob_client = self.blob_service_client.get_blob_client(container=container_name,blob=blob_uri)
            if local_path:
                blob_name = os.path.join(local_path,os.path.basename(blob_uri))
            else:
                blob_name = os.path.join(os.getcwd(),os.path.basename(blob_uri))
            with open(blob_name,"wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            return True
        except:
            print("Failed to download file:", traceback.print_exc())
            return False

    def upload_file(self, file_path:str, IP: str=None)-> tuple:
        try:
            account_url, credential, container_name = self.__generate_sas_token("upload", IP)
            if account_url is None:
                return False, None
            # container_url_with_sas = f"{account_url}/{container_name}?{credential}"
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=os.path.basename(file_path)
            )
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            blob_path = f"{container_name}/{os.path.basename(file_path)}"
            return True, blob_path
        except:
            print("Failed to upload file:", traceback.print_exc())
            return False, None


if __name__ == "__main__":
    print("lets start")
    blob = AzureBlobStorage()
    blob_name = "GRN-UPNCRPC1_01-VS-3212-2324-04062023-000001163_DISCR.pdf"
    print("downloading ",blob_name)
    dwn = blob.download_file(blob_name,local_path)
    print("download :",dwn)
    if os.path.exists("./"+blob_name):
        upd = blob.upload_file("./"+blob_name)
        print("upload :",upd)
