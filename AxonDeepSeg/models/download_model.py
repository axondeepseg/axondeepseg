from AxonDeepSeg.ads_utils import download_data
import shutil

def main():
    url_TEM_model = "https://osf.io/2hcfv/?action=download&version=2"  # URL of TEM model hosted on OSF storage with the appropriate versioning on OSF
    url_SEM_model = "https://osf.io/rdqgb/?action=download&version=2"  # URL of SEM model hosted on OSF storage with the appropriate versioning on OSF

    if (not download_data(url_TEM_model) and not download_data(url_SEM_model)) ==1:
        print('Data downloaded and unzipped succesfully.')
    else:
        print('ERROR: Data was not succesfully downloaded and unzipped- please check your link and filename and try again.')
    shutil.move("default_SEM_model_v1", "AxonDeepSeg/models/default_SEM_model_v1")
    shutil.move("default_TEM_model_v1", "AxonDeepSeg/models/default_TEM_model_v1")

# Calling the script
if __name__ == '__main__':
    main()
