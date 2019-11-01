from AxonDeepSeg.ads_utils import download_data

def main():
    url_TEM_model = "https://osf.io/2hcfv/?action=download"  # URL of TEM model hosted on OSF storage
    url_SEM_model = "https://osf.io/rdqgb/?action=download"  # URL of SEM model hosted on OSF storage

    if (not download_data(url_TEM_model) and not download_data(url_SEM_model)) ==1:
        print('Data downloaded and unzipped succesfully.')
    else:
        print('ERROR: Data was not succesfully downloaded and unzipped- please check your link and filename and try again.')



# Calling the script
if __name__ == '__main__':
    main()
