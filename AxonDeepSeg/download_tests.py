from AxonDeepSeg.ads_utils import download_data, convert_path
from pathlib import Path
import shutil
import requests

def download_tests(destination = None):
    if destination is None:
        test_files_destination = Path("test/__test_files__")
    else:
        destination = convert_path(destination)
        test_files_destination = destination / "__test_files__"


    # retrieve latest release identifier from github API
    response_tests_files = requests.get("https://api.github.com/repos/axondeepseg/data-testing/releases/latest")
    folder_name_test_files = Path("data-testing-"+str(response_tests_files.json()['tag_name']))

    url_tests = "https://github.com/axondeepseg/data-testing/archive/refs/tags/"+str(response_tests_files.json()['tag_name'])+".zip"   # URL of TEM model hosted on OSF storage with the appropriate versioning on OSF

    if (
        not download_data(url_tests)
    ) == 1:
        print("Test data downloaded and unzipped successfully.")
    else:
        print(
            "ERROR: Test data was not successfully downloaded and unzipped- please check your link and filename and try again."
        )

    if test_files_destination.exists():
        print('SEM model folder already existed - deleting old one.')
        shutil.rmtree(str(folder_name_test_files))

    shutil.move(folder_name_test_files.joinpath("__test_files__"), str(test_files_destination))


def main(argv=None):
    download_tests()
