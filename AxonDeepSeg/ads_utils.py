import os
import sys
import raven
import configparser
from distutils.util import strtobool

DEFAULT_CONFIGFILE = ".adsconfig"


def config_setup():

    configPath = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        DEFAULT_CONFIGFILE
        )

    print ("To improve user experience and fix bugs, the ADS development team "
           "is using a report system to automatically receive crash reports "
           "and errors from users. These reports are anonymous.")

    bugTracking = strtobool(
        raw_input("Do you agree to help us improve ADS? [y]es/[n]o:")
        )

    if bugTracking:
        print ("Note: you can opt out of Sentry reporting by changing the "
               "value of bugTracking from 1 to 0 in the "
               "file {}".format(configPath))

    config = configparser.ConfigParser()
    config['Global'] = {
        'bugTracking': bugTracking
    }

    with open(configPath, 'w') as configFile:
        config.write(configFile)

    print("Configuration saved successfully !")


def read_config():
    """Read the system configuration file.
    :return: a dict with the configuration parameters.
    """
    configPath = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        DEFAULT_CONFIGFILE
        )

    if not os.path.exists(configPath):
        raise IOError("Could not find configuration file.")

    config = configparser.ConfigParser()
    config.read(configPath)
    return config['Global']


def init_ads():
    """ Initialize ads for typical terminal usage
    :return:
    """

    configPath = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        DEFAULT_CONFIGFILE
        )

    if not os.path.isfile(configPath):
        config_setup()
    else:
        pass

    config = read_config()
    init_error_client(config['bugTracking'])


def init_error_client(bugTracking):
    """ Send traceback to neuropoly servers
    :return:
    """

    if bugTracking:

        try:

            client = raven.Client(
                        "https://e04a130541c64bc9a64939672f19ad52@sentry.io/1238683",
                        processors=(
                            'raven.processors.RemoveStackLocalsProcessor',
                            'raven.processors.SanitizePasswordsProcessor')
                            )

            traceback_to_server(client)

            old_exitfunc = sys.exitfunc

            def exitfunc():
                sent_something = False
                try:
                    # implementation-specific
                    import atexit
                    for handler, args, kw in atexit._exithandlers:

                        if handler.__module__.startswith("raven."):
                            sent_something = True

                except:
                    pass
                old_exitfunc()

                print ("Note: you can opt out of Sentry reporting by "
                       "setting \"bugTracking = False\" in the function "
                       "init_ads() of ads_utils.py")

            sys.exitfunc = exitfunc

        except:
            print "Unexpected error: bug tracking may not be functionning."


def traceback_to_server(client):
    """
        Send all traceback children of Exception to sentry
    """

    def excepthook(exctype, value, traceback):
        if issubclass(exctype, Exception):
            client.captureException(exc_info=(exctype, value, traceback))
        sys.__excepthook__(exctype, value, traceback)

    sys.excepthook = excepthook

# Call init_ads() automatically when module is imported
init_ads()
