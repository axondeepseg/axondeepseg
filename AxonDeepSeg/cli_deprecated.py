from loguru import logger
import AxonDeepSeg.segment
import AxonDeepSeg.integrity_test
import AxonDeepSeg.morphometrics.launch_morphometrics_computation
import AxonDeepSeg.morphometrics.aggregate
import AxonDeepSeg.morphometrics.filter_morphometrics
import AxonDeepSeg.morphometrics.count_axons

def _warn(old_cmd: str, new_cmd: str):
    logger.warning(
        f"The '{old_cmd}' command is deprecated and will be removed in a future release. "
        f"Please use '{new_cmd}' instead."
    )

def axondeepseg():
    _warn("axondeepseg", "ads_segment")
    AxonDeepSeg.segment.main()

def axondeepseg_morphometrics():
    _warn("axondeepseg_morphometrics", "ads_morphometrics")
    AxonDeepSeg.morphometrics.launch_morphometrics_computation.main()

def axondeepseg_aggregate():
    _warn("axondeepseg_aggregate", "ads_aggregate")
    AxonDeepSeg.morphometrics.aggregate.main()

def axondeepseg_filter():
    _warn("axondeepseg_filter", "ads_filter")
    AxonDeepSeg.morphometrics.filter_morphometrics.main()

def axondeepseg_count():
    _warn("axondeepseg_count", "ads_count")
    AxonDeepSeg.morphometrics.count_axons.main()

def axondeepseg_test():
    _warn("axondeepseg_test", "ads_test")
    AxonDeepSeg.integrity_test.main()
