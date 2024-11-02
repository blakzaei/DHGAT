#-- Import -------------------------------------------------------------------------------------
import fasttext
import fasttext.util

from IPython import display
import os
import shutil

from utils.util import create_folder
#-----------------------------------------------------------------------------------------------

def download():
    ft_dir = create_folder('data/fasttext_model')
    model_file_path = os.path.join(ft_dir, 'cc.en.300.bin')

    if not os.path.isfile(model_file_path):
        print("FastText Model not found, downloading...")

        #fasttext.util.download_model('en', if_exists='ignore')

        # downloaded_file_path = 'cc.en.300.bin'
        # shutil.move(downloaded_file_path, model_file_path)
        # downloaded_file_path = 'cc.en.300.bin.gz'
        # shutil.move(downloaded_file_path, model_file_path)

        display.clear_output()
        print('FastText downloaded successfully :)\n')

    else:
        print("Model already exists, skipping download :)\n")
#-----------------------------------------------------------------------------------------------