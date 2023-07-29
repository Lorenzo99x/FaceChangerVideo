import cv2
import face_recognition
import numpy


def _readFromIPCamera():
    IP = "192.168.1.34"
    PORT = "8080"
    url = f"http://{IP}:{PORT}/video"
    cap = cv2.VideoCapture(url)
    return cap

def _getFrame(cap):
    ret, frame = cap.read()
    if not ret:
        print("error reading frame")
        return frame
    return frame

def _getRGBFrame(cap):
    ret, frame = cap.read()
    if not ret:
        print("error reading frame")
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print("rgb_frame preso")
    return rgb_frame

def _detectFace(rgb_frame, bgr_image):

    face_locations = face_recognition.face_locations(rgb_frame)
    if len(face_locations) > 0:
        for top, right, bottom, left in face_locations:
            cv2.rectangle(bgr_frame, (left, top), (right, bottom), (0, 0, 255), 2)
            print("face detected")
            cv2.imshow('facciabell', bgr_frame)
    else:
        print("face doesn't detected")
    return bgr_frame


def _waitEnter(cap,out):
    if cv2.waitKey(1) == 13:
        print("exiting")
        cap.release()
        out.release()
        return True
    return False


if __name__=="__main__":

    face_locations = []

    cap = _readFromIPCamera()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("C:/Users/loren/Downloads/test.mp4", fourcc, fps, (width, height))

    while cap.isOpened():
        print("dentro while")
        bgr_frame = _getFrame(cap)
        rgb_frame = _getRGBFrame(cap)
        if numpy.size(bgr_frame, axis=None) == 0 or numpy.size(rgb_frame, axis=None) == 0:
            cap.release()
            out.release()
            break

        img_detected = _detectFace(rgb_frame, bgr_frame)
        cv2.imshow('frame', img_detected)
        out.write(img_detected)
        _waitEnter(cap, out)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

