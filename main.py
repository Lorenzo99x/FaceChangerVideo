import ErrorManager as erm

import cv2
import face_recognition
import numpy
import ErrorManager

def _readFromIPCamera(ip, port):
    IP = ip
    PORT = port
    url = f"http://{IP}:{PORT}/video"
    cap = cv2.VideoCapture(url)
    return cap

def _getFrame(cap):
    ret, frame = cap.read()
    if not ret:
        return ret,frame
    return ret, frame

def _getRGBFrame(cap, frame):
    ret, frame = cap.read()
    if not ret:
        print("error reading frame")
        return ret, frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return ret, frame

def _detectFace(rgb_frame, bgr_image):

    face_locations = face_recognition.face_locations(rgb_frame)
    if len(face_locations) > 0:
        for top, right, bottom, left in face_locations:
            cv2.rectangle(bgr_image, (left, top), (right, bottom), (0, 0, 255), 2)
            print("face detected")
    else:
        print("face doesn't detected")
    return bgr_image


def _waitEnter(cap,out):
    if cv2.waitKey(1) == 13:
        erm.releaseCW(cap, out)
        return True
    return False

def initVideoWriter(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    path = input("Insert path to save the video")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    return out

if __name__=="__main__":

    face_locations = []
    ip = input("Insert the IP")
    erm.checkCritical(ip, False, "Ip not valid")
    port = input("Insert port")
    erm.checkCritical(port, False, "Port not valid")
    cap = _readFromIPCamera(ip, port)

    erm.checkCritical(cap, None, "VideoCapture not found")

    out = initVideoWriter(cap)
    erm.checkError(out, None, "Cannot open video writer")

    while cap.isOpened():
        ret = _getFrame(cap) #we can have some error to single frame
        erm.checkWarning(ret[0], False, "Doesn't get BGR frame")
        ret_rgb = _getRGBFrame(cap, ret[1])
        erm.checkWarning(ret_rgb[0], False, "Doesn't get RGB frame")

        if numpy.size(ret[1], axis=None) == 0 or numpy.size(ret_rgb[1], axis=None) == 0:
            erm.releaseCW(cap, out)
            break

        img_detected = _detectFace(ret_rgb[1], ret[1])
        cv2.imshow('frame', img_detected)
        out.write(img_detected)
        _waitEnter(cap, out)

    erm.checkInfo(1,1, "Program shutting down successfully")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
