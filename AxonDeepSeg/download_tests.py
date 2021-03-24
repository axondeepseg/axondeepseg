from AxonDeepSeg.ads_utils import download_data, convert_path
from pathlib import Path
import shutil


def download_tests(destination = None):
    if destination is None:
        test_files_destination = Path("test/__test_files__")
    else:
        destination = convert_path(destination)
        test_files_destination = destination / "__test_files__"

    url_tests = "https://osf.io/vqt94/?action=download"  # URL of TEM model hosted on OSF storage with the appropriate versioning on OSF

    if (
        not download_data(url_tests)
    ) == 1:
        print("Test data downloaded and unzipped succesfully.")
    else:
        print(
            "ERROR: Test data was not succesfully downloaded and unzipped- please check your link and filename and try again."
        )

    if test_files_destination.exists():
        print('SEM model folder already existed - deleting old one.')
        shutil.rmtree(str(test_files_destination))

    shutil.move("__test_files__", str(test_files_destination))


def main(argv=None):
    download_tests()
