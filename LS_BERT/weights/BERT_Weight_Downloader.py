from __future__ import print_function
import warnings
warnings.filterwarnings('ignore','.*conversion.*')
import os
import zipfile
import shutil
import requests
import numpy as np
import argparse

import warnings
from sys import stdout
from os import makedirs
from os.path import dirname
from os.path import exists

################################
# Weights with Google Drive IDs#
################################

weights = {
    'chinese_roberta_wwm_ext_pytorch': '1eHM3l4fMo6DsQYGmey7UZGiTmQquHw25',
}

##########################
# Google Drive Downloader#
##########################
class GoogleDriveDownloader:
    """
    Minimal class to download shared files from Google Drive.
    """

    CHUNK_SIZE = 32768
    DOWNLOAD_URL = 'https://docs.google.com/uc?export=download'

    @staticmethod
    def download_file_from_google_drive(file_id, dest_path, overwrite=False, showsize=True):
        """
        Downloads a shared file from google drive into a given folder.

        Parameters
        ----------
        file_id: str
            the file identifier.
            You can obtain it from the sharable link.
        dest_path: str
            the destination where to save the downloaded file.
            Must be a path (for example: './downloaded_file.txt')
        overwrite: bool
            optional, if True forces re-download and overwrite.
        showsize: bool
            optional, if True print the current download size.
        Returns
        -------
        None
        """

        destination_directory = dirname(dest_path)
        if not exists(destination_directory):
            makedirs(destination_directory)

        if not exists(dest_path) or overwrite:

            session = requests.Session()
            stdout.flush()
            response = session.get(GoogleDriveDownloader.DOWNLOAD_URL, params={'id': file_id}, stream=True)

            token = GoogleDriveDownloader._get_confirm_token(response)
            if token:
                params = {'id': file_id, 'confirm': token}
                response = session.get(GoogleDriveDownloader.DOWNLOAD_URL, params=params, stream=True)

            current_download_size = [0]
            GoogleDriveDownloader._save_response_content(response, dest_path, showsize, current_download_size)
            print('Done.')

    @staticmethod
    def _get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    @staticmethod
    def _save_response_content(response, destination, showsize, current_size):
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(GoogleDriveDownloader.CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    if showsize:
                        print('\r' + GoogleDriveDownloader.sizeof_fmt(current_size[0]), end=' ')
                        stdout.flush()
                        current_size[0] += GoogleDriveDownloader.CHUNK_SIZE

    # From https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
    @staticmethod
    def sizeof_fmt(num, suffix='B'):
        for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
            if abs(num) < 1024.0:
                return '{:.1f} {}{}'.format(num, unit, suffix)
            num /= 1024.0
        return '{:.1f} {}{}'.format(num, 'Yi', suffix)

###########################
# ReID Dataset Downloader#
###########################

def BERT_Weight_Downloader(save_dir, weight_name):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_dir_exist = os.path.join(save_dir , weight_name)

    if not os.path.exists(save_dir_exist):
        temp_dir = os.path.join(save_dir , 'temp')

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        destination = os.path.join(temp_dir , weight_name)

        id = weights[weight_name]

        print("Downloading %s" % weight_name)
        # gdrive_downloader(destination, id)
        GoogleDriveDownloader.download_file_from_google_drive(file_id=id,dest_path=destination,showsize=True)

        zip_ref = zipfile.ZipFile(destination)
        print("Extracting %s" % weight_name)
        zip_ref.extractall(os.path.join(save_dir,weight_name))
        zip_ref.close()
        shutil.rmtree(temp_dir)
        print("Done")
    else:
        print("Weight Check Success: %s exists!" %weight_name)

#For United Testing and External Use
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset Name and Dataset Directory')
    parser.add_argument(dest="save_dir", action="store", default="./weight/RetinaFace/",help="")
    parser.add_argument(dest="weight_name", action="store",default="RetinaFace_Resnet50", type=str,help="")
    args = parser.parse_args() 
    BERT_Weight_Downloader(args.save_dir,args.weight_name)