from loguru import logger

# Imports are done inside each wrapper so a deprecated command only loads the
# module it needs (segment pulls in torch/nnunet, which is slow to import).

def _warn(old_cmd: str, new_cmd: str):
    logger.warning(
        f"The '{old_cmd}' command is deprecated and will be removed in v6. "
        f"Please use '{new_cmd}' instead."
    )

def axondeepseg():
    _warn("axondeepseg", "ads_segment")
    from AxonDeepSeg.segment import main
    main()

def axondeepseg_morphometrics():
    _warn("axondeepseg_morphometrics", "ads_morphometrics")
    from AxonDeepSeg.morphometrics.launch_morphometrics_computation import main
    main()

def axondeepseg_aggregate():
    _warn("axondeepseg_aggregate", "ads_aggregate")
    from AxonDeepSeg.morphometrics.aggregate import main
    main()

def axondeepseg_filter():
    _warn("axondeepseg_filter", "ads_filter")
    from AxonDeepSeg.morphometrics.filter_morphometrics import main
    main()

def axondeepseg_count():
    _warn("axondeepseg_count", "ads_count")
    from AxonDeepSeg.morphometrics.count_axons import main
    main()

def axondeepseg_test():
    _warn("axondeepseg_test", "ads_test")
    from AxonDeepSeg.integrity_test import main
    main()

def download_model():
    _warn("download_model", "ads_download_model")
    from AxonDeepSeg.download_model import main
    main()

def download_tests():
    _warn("download_tests", "ads_download_tests")
    from AxonDeepSeg.download_tests import main
    main()
