import os
import sys
import raven


def init_ads():
    """ Initialize ads for typical terminal usage
    :return:
    """
    init_error_client()


def get_dsn_filepath():
    ''' Get full path directory to Sentry DSN file.
    :return: dsnFilePath.
    '''

    # Generate n-th generation path directory name
    uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])

    dsnFilePath = os.path.join(
        uppath(os.path.realpath(__file__), 2),  # Go to parent of module dir
        'bin',
        'ads_sentry'
        )

    return dsnFilePath


def init_error_client():
    """ Send traceback to neuropoly servers
    :return:
    """

    dsnFilePath = get_dsn_filepath()

    if os.path.isfile(dsnFilePath):

        with open(dsnFilePath) as f:
            sentryDSN = f.readline()

        try:

            client = raven.Client(
                        sentryDSN,
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
                if sent_something:
                    print ("Note: you can opt out of Sentry reporting by "
                           "deleting the file (axondeepseg)/bin/ads_sentry")

            sys.exitfunc = exitfunc
        except raven.exceptions.InvalidDsn:
            # This could happen if sct staff change the dsn
            print 'Sentry DSN not valid anymore, not reporting errors'


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
