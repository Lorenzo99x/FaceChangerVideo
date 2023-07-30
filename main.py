import numpy as np

import ErrorManager as erm
import cv2
import face_recognition
import dlib
from imutils import face_utils
import numpy


predictor = dlib.shape_predictor("C:/Users/loren/PycharmProjects/OpenCVCartoon/shape_predictor_68_face_landmarks.dat")

def read_from_ip_camera(ip, port):
    IP = ip
    PORT = port
    url = f"http://{IP}:{PORT}/video"
    cap = cv2.VideoCapture(url)
    return cap


def get_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return ret,frame
    return ret, frame


def get_rgb_frame(cap, frame):
    ret, frame = cap.read()
    if not ret:
        print("error reading frame")
        return ret, frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return ret, frame

def get_landmark_points_nparray(landmarks):
    landmarks_points = [(p.x, p.y) for p in landmarks.parts()]
    points = np.array(landmarks_points, np.int32)
    return points

def draw_points_face(bgr_frame, face_landmarks):
    points = face_utils.shape_to_np(face_landmarks)
    for (x, y) in points:
        cv2.circle(bgr_frame, (x, y), 2, (0, 255, 0), -1)


def rect_contains(rect, point):
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

def draw_delonay_triangulation(img,face_landmarks):
    delaunay_color = (255, 255, 255)
    points_color = (0, 0, 255)
    img_orig = img.copy()
    triangleList = delonay_triangulation_get_triangles(img,  face_landmarks)
    size = img.shape
    rect = (0, 0, size[1], size[0])

    for t in triangleList:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        if rect_contains(rect,pt1) and rect_contains(rect,pt2) and rect_contains(rect,pt3):
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA , 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA , 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA , 0)


def get_triangle_indexes (img,points):
    delaunay_color = (255, 255, 255)
    points_color = (0, 0, 255)
    img_orig = img.copy()
    triangleList = delonay_triangulation_get_triangles(img, face_landmarks)
    triangle_indexes = []
    index_triangles = []
    for t in triangleList:
        pt1 = (int(points[0][0]), int(points[0][1]))
        pt2 = (int(points[1][0]), int(points[1][1]))
        pt3 = (int(points[2][0]), int(points[2][1]))

        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)
        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)
        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            triangle_indexes.append(triangle)

    return triangle_indexes

def delonay_triangulation_get_triangles(img,face_landmarks):
    landmarks_points = [(p.x, p.y) for p in face_landmarks.parts()]
    img_orig = img.copy()
    size = img.shape
    rect = (0,0,size[1],size[0])
    subdiv = cv2.Subdiv2D(rect)
    for p in landmarks_points:
        subdiv.insert(p)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)
    return triangles

def extract_points_face(rgb_frame):
    face_locations = face_recognition.face_locations(rgb_frame)
    if len(face_locations) > 0:
        top, right, bottom, left = face_locations[0]
        face_landmarks = predictor(rgb_frame,  dlib.rectangle(left, top, right, bottom))
        return face_landmarks
        '''
        for i in range(1, len(points)):
            cv2.line(bgr_frame, tuple(points[i-1]), tuple(points[i]), (0, 0, 255), 2)
        cv2.line(bgr_frame, tuple(points[-1]), tuple(points[0]), (0, 0, 255), 2)
        cv2.imshow("face_detected", bgr_frame)
        mask = np.zeros_like(bgr_frame)
        cv2.fillPoly(mask, [points], (255,255,255))
        face_only = cv2.bitwise_and(bgr_frame, mask)
        cv2.imshow("face retail", face_only)'''
    else:
        print("face doesn't detected")

def extract_face(rgb_frame, face_landmarks):
    points = [(p.x, p.y) for p in face_landmarks.parts()]
    points = np.array(points, np.int32)
    left = np.min(points[:, 0])
    top = np.min(points[:, 1])
    right = np.max(points[:, 0])
    bottom = np.max(points[:, 1])
    face_extracted = rgb_frame[top:bottom, left:right]
    mask = np.zeros_like(face_extracted)
    points -= (left, top)  # Translate points to the local coordinate system of the face region
    cv2.fillPoly(mask, [points], (255, 255, 255))  # Fill the facial landmarks with white color

    # Apply the mask to the face region to obtain only the facial landmarks
    face_with_landmarks = cv2.bitwise_and(face_extracted, mask)
    return face_extracted

def detect_face(rgb_frame, bgr_image):
    face_locations = face_recognition.face_locations(rgb_frame)
    if len(face_locations) > 0:
        for top, right, bottom, left in face_locations:
            cv2.rectangle(bgr_image, (left, top), (right, bottom), (0, 0, 255), 2)
            print("face detected")
    else:
        print("face doesn't detected")
    return bgr_image


def wait_enter(cap,out):
    if cv2.waitKey(1) == 13:
        erm.releaseCW(cap, out)
        return True
    return False

def init_video_writer(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    path = input("Insert path to save the video")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    return out

if __name__=="__main__":

    img = cv2.imread("C:/Users/loren/Downloads/fedez-20220912-newsabruzzo.jpg")
    img2 = cv2.imread("C:/Users/loren/Downloads/volto_uomo.jpg")
    ret_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ret2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    cv2.imshow("Foto", img)
    face_landmarks = extract_points_face(ret_rgb)
    landmarks_points = get_landmark_points_nparray(face_landmarks)
    #triangle_indexes = get_triangle_indexes(img, landmarks_points)
    draw_delonay_triangulation(img, face_landmarks)

    draw_points_face(img, face_landmarks)
    cv2.imshow("Points attached", img)
    face_extracted = extract_face(ret_rgb, face_landmarks)
    face_extracted = cv2.cvtColor(face_extracted, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Face extracted", face_extracted)

    face_landmarks2 = extract_points_face(ret2_rgb)
    landmarks_points2 = get_landmark_points_nparray(face_landmarks2)
    #triangle_indexes2 = get_triangle_indexes(img2, landmarks_points2)
    draw_delonay_triangulation(img2, face_landmarks2)
    '''for triangle_index in triangle_indexes:
        print(triangle_index)
        pt1 = landmarks_points[triangle_index[0]]
        pt2 = landmarks_points[triangle_index[0]]
        pt3 = landmarks_points[triangle_index[0]]
        cv2.line(img2, pt1, pt2, (0,0,255), 2)
        cv2.line(img2, pt2, pt3, (0, 0, 255), 2)
        cv2.line(img2, pt3, pt1, (0, 0, 255), 2)'''

    cv2.imshow("Second image", img2)
#primo landmarks point
    if cv2.waitKey(0) == 13:
        cv2.destroyAllWindows()
    '''
    face_locations = []
    ip = input("Insert the IP")
    erm.checkCritical(ip, False, "Ip not valid")
    port = input("Insert port")
    erm.checkCritical(port, False, "Port not valid")
    cap = read_from_ip_camera(ip, port)

    erm.checkCritical(cap, None, "VideoCapture not found")

    out = init_video_writer(cap)
    erm.checkError(out, None, "Cannot open video writer")

    while cap.isOpened():
        ret = get_frame(cap) #we can have some error to single frame
        erm.checkWarning(ret[0], False, "Doesn't get BGR frame")
        ret_rgb = get_rgb_frame(cap, ret[1])
        erm.checkWarning(ret_rgb[0], False, "Doesn't get RGB frame")

        if numpy.size(ret[1], axis=None) == 0 or numpy.size(ret_rgb[1], axis=None) == 0:
            erm.releaseCW(cap, out)
            break

        img_detected = detect_face(ret_rgb[1], ret[1])
        cv2.imshow('frame', img_detected)
        out.write(img_detected)
        wait_enter(cap, out)

    erm.checkInfo(1,1, "Program shutting down successfully")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    '''
