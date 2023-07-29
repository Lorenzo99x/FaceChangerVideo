import logging
import sys
import re

def isValidIPV4(ip):
    ipv4_pattern = r"^(25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\." \
                   r"(25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\." \
                   r"(25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\." \
                   r"(25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)$"
    valid = re.match(ipv4_pattern, ip)
    return bool(valid)

def isValidPort(port):
    number_port = int(port)

    if 1 <= port <= 65535:
        return True
    else:
        return False

def releaseCW(cap, writer):
    cap.release()
    writer.release()
def checkError(var, error, string):
    if var == error:
        logger.error(string)
        sys.exit(1)

def checkCritical(var, error, string):
    if var == error:
        logger.critical(string)
        sys.exit(2)

def checkInfo(var, error, string):
    if var == error:
        logger.info(string)
        return True
    return False

def checkWarning(var, error, string):
    if var == error:
        logger.warning(string)
        return True
    return False

def checkDebug(var, error, string):
    if var == error:
        logger.debug(string)
        return True
    return False


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("debugger.log")
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

