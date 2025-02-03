from ads_base.ads_utils import download_data, convert_path
from pathlib import Path
import shutil
import ads_base

def download_tests(destination=None):
    package_dir = Path(ads_base.__file__).parent
    if destination is None:
        test_files_destination =package_dir / "test" / "__test_files__"
    else:
        destination = convert_path(destination)
        test_files_destination = destination / "__test_files__"

    url_tests = "https://github.com/axondeepseg/data-testing/archive/refs/tags/r20250117.zip"  
    files_before = list(Path.cwd().iterdir())

    if (
        not download_data(url_tests)
    ) == 1:
        print("Test data downloaded and unzipped successfully.")
    else:
        print(
            "ERROR: Test data was not successfully downloaded and unzipped- please check your link and filename and try again."
        )

    files_after = list(Path.cwd().iterdir())

    # retrieving unknown test files names
    test_folder = list(set(files_after)-set(files_before))
    folder_name_test_files = ''.join([str(x) for x in test_folder if 'data-testing' in str(x)])

    if test_files_destination.exists():
        print('Test files folder already existed - deleting old one.')
        shutil.rmtree(str(test_files_destination))

    shutil.move(Path(folder_name_test_files).joinpath("__test_files__"), str(test_files_destination))

    # remove temporary folder
    shutil.rmtree(folder_name_test_files)

def main(argv=None):
    download_tests()
