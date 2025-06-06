import AxonDeepSeg
from AxonDeepSeg.ads_utils import download_data, convert_path
from pathlib import Path
from loguru import logger
import shutil
import argparse


def download_tests(destination=None, overwrite=True):
    '''
    Download test data for AxonDeepSeg.
    
    Parameters
    ----------
    destination : str
        Directory to download the tests to. Default: test/
    '''
    # Get AxonDeepSeg installation path
    package_dir = Path(AxonDeepSeg.__file__).parent
    if destination is None:
        test_files_destination =  package_dir.parent / "test" / "__test_files__"
        print('Downloading test files to default location: {}'.format(test_files_destination))
    else:
        destination = convert_path(destination)
        test_files_destination = destination / "__test_files__"

    if test_files_destination.exists() and overwrite == False:
        logger.info("Overwrite set to False - not deleting old test files.")
        return test_files_destination

    url_tests = "https://github.com/axondeepseg/data-testing/archive/refs/tags/r20250523.zip"
    
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
    output_dir=test_files_destination.resolve()


    if test_files_destination.exists():
        print('Test files folder already existed - deleting old one.')
        shutil.rmtree(str(test_files_destination))

    shutil.move(Path(folder_name_test_files).joinpath("__test_files__"), str(test_files_destination))

    # remove temporary folder
    shutil.rmtree(folder_name_test_files)

    return output_dir

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-d", "--dir",
        required=False,
        help="Directory to download the tests to. Default: test/",
        default = None,
    )
    args = vars(ap.parse_args(argv))
    print(args["dir"])
    download_tests(destination=args["dir"], overwrite=True)

if __name__ == "__main__":
    with logger.catch():
        main()
