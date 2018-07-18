import os
import sys
import raven


def init_ads():
    """ Initialize ads for typical terminal usage
    :return:
    """
    bugTracking = True

    init_error_client(bugTracking)


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
