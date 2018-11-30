import os
import sys
import configparser
from distutils.util import strtobool
import raven
from tqdm import tqdm
import tempfile
import zipfile
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util import Retry

DEFAULT_CONFIGFILE = "axondeepseg.cfg"

# raven function override - do not modify unless needed if raven version is
# changed to a version other than 6.8.0.
# See https://github.com/getsentry/raven-python/blob/master/raven/transport/threaded.py
# -- Start function override -- #
def _main_thread_terminated(self):
    self._lock.acquire()
    try:
        if not self.is_alive():
            # thread not started or already stopped - nothing to do
            return

        # wake the processing thread up
        self._queue.put_nowait(self._terminator)

        timeout = self.options['shutdown_timeout']

        # wait briefly, initially
        initial_timeout = 0.1
        if timeout < initial_timeout:
            initial_timeout = timeout

        if not self._timed_queue_join(initial_timeout):
            # if that didn't work, wait a bit longer
            # NB that size is an approximation, because other threads may
            # add or remove items
            size = self._queue.qsize()

            print(("Sentry is attempting to send %i pending error messages"
                  % size))
            print(("Waiting up to %s seconds" % timeout))

            if os.name == 'nt':
                print("Press Ctrl-Break to quit")
            else:
                print("Press Ctrl-C to quit")

            # -- Function override statement --#
            config_path = get_config_path()
            print(("Note: you can opt out of Sentry reporting by changing the "
                   "value of bugTracking to 0 in the "
                   "file {}".format(config_path)))
            # -- EO Function override statement --#

            self._timed_queue_join(timeout - initial_timeout)

        self._thread = None

    finally:
        self._lock.release()


raven.transport.threaded.AsyncWorker.main_thread_terminated = _main_thread_terminated
# -- End function override -- #

def config_setup():

    config_path = get_config_path()
    
    if 'pytest' in sys.modules:
        bugTracking = bool(0)
    else:
        print ("To improve user experience and fix bugs, the ADS development team "
               "is using a report system to automatically receive crash reports "
               "and errors from users. These reports are anonymous.")

        bugTracking = strtobool(
            input("Do you agree to help us improve ADS? [y]es/[n]o:")
            )

    if bugTracking:
        print(("Note: you can opt out of Sentry reporting by changing the "
               "value of bugTracking from 1 to 0 in the "
               "file {}".format(config_path)))

    config = configparser.ConfigParser()
    config['Global'] = {
        'bugTracking': bugTracking,
    }

    with open(config_path, 'w') as configFile:
        config.write(configFile)

    print("Configuration saved successfully !")

def get_config_path():
    """Get the full path of the AxonDeepSeg configuration file.
    :return: String with the full path to the ADS config file.
    """
    return os.path.join(
                    os.path.expanduser("~"),
                    DEFAULT_CONFIGFILE
                    )

def read_config():
    """Read the system configuration file.
    :return: a dict with the configuration parameters.
    """

    config_path = get_config_path()

    if not os.path.exists(config_path):
        raise IOError("Could not find configuration file.")

    config = configparser.ConfigParser()
    config.read(config_path)

    return config


def init_ads():
    """ Initialize ads for typical terminal usage
    :return:
    """

    config_path = get_config_path()

    if not os.path.isfile(config_path):
        config_setup()
    else:
        pass

    config = read_config()

    init_error_client(config.get('Global','bugTracking'))


def init_error_client(bugTracking):
    """ Send traceback to neuropoly servers
    :return:
    """

    if strtobool(bugTracking):

        try:

            client = raven.Client(
                        "https://e04a130541c64bc9a64939672f19ad52@sentry.io/1238683",
                        processors=(
                            'raven.processors.RemoveStackLocalsProcessor',
                            'raven.processors.SanitizePasswordsProcessor')
                            )

            traceback_to_server(client)

        except:
            print("Unexpected error: bug tracking may not be functionning.")


def traceback_to_server(client):
    """
        Send all traceback children of Exception to sentry
    """

    def excepthook(exctype, value, traceback):
        if issubclass(exctype, Exception):
            client.captureException(exc_info=(exctype, value, traceback))
        sys.__excepthook__(exctype, value, traceback)

    sys.excepthook = excepthook

def download_data(url_data, zip_filename):
    """ Downloads and extracts zip files from the web.
    :return: 0 - Success, 1 - Encountered an exception.
    """

    # Download
    try:
        print('Trying URL: %s' % url_data)
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 503, 504])
        session = requests.Session()
        session.mount('https://', HTTPAdapter(max_retries=retry))
        response = session.get(url_data, stream=True)
        tmp_path = os.path.join(tempfile.mkdtemp(), zip_filename)
        with open(tmp_path, 'wb') as tmp_file:
            total = int(response.headers.get('content-length', 1))
            tqdm_bar = tqdm(total=total, unit='B', unit_scale=True, desc="Downloading", ascii=True)
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp_file.write(chunk)
                    dl_chunk = len(chunk)
                    tqdm_bar.update(dl_chunk)
            tqdm_bar.close()
    except Exception as e:
        print("ERROR: %s" % e)
        return 1

    # Unzip
    print("Unzip...")
    try:
        zf = zipfile.ZipFile(tmp_path)
        zf.extractall(".")
    except (zipfile.BadZipfile):
        print('ERROR: ZIP package corrupted. Please try downloading again.')
        return 1
    print("--> Folder created: " + os.path.join(os.path.abspath(os.curdir), zip_filename.strip(".zip")))
    return 0

# Call init_ads() automatically when module is imported
init_ads()
