import cv2
import numpy as np


# init part
face_cascade = cv2.CascadeClassifier('./eye_tracking/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./eye_tracking/haarcascade_eye.xml')
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)

def brighten(image, factor):
    return cv2.add(image,np.array([factor * 1.0]))

def detect_faces(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        frame = img[y:y + h, x:x + w]
    # return frame, (x, y, w, h)
    return frame

'''
def detect_eyes(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("gray")
    cv2.imshow('image',gray_frame)
    cv2.waitKey(0)

    for factor in range(0, 100, 10):
        brightened = brighten(gray_frame, factor)
        print("after brighten")
        cv2.imshow('image',brightened)
        cv2.waitKey(0)

        eyes = cascade.detectMultiScale(brightened, 1.3, 5)  # detect eyes
        print(str(len(eyes)) + " eyes detected")
    width = np.size(img, 1)  # get face frame width
    height = np.size(img, 0)  # get face frame height
    left_eye = None
    right_eye = None
    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
        else:
            right_eye = img[y:y + h, x:x + w]
    return left_eye, right_eye
'''
def detect_eyes(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = cascade.detectMultiScale(gray_frame, 1.3, 5)  # detect eyes
    width = np.size(img, 1)  # get face frame width
    height = np.size(img, 0)  # get face frame height
    left_eye = None
    right_eye = None
    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            left_eye = (img[y:y + h, x:x + w], (x, y, w, h))
        else:
            right_eye = (img[y:y + h, x:x + w], (x, y))
    return left_eye, right_eye


def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)

    return img, eyebrow_h


def blob_process(img, threshold, detector):
    # print(threshold)
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    # print("Before processing")
    # cv2.imshow('image',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img = cv2.erode(img, None, iterations=1)
    # print("After erode")
    # cv2.imshow('image',img)
    # cv2.waitKey(0)
    img = cv2.dilate(img, None, iterations=1)
    # print("After dilate")
    # cv2.imshow('image',img)
    # cv2.waitKey(0)
    img = cv2.medianBlur(img, 5)
    # print("After processing")
    # cv2.imshow('image',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    keypoints = detector.detect(img)
    return keypoints


def nothing(x):
    pass

def detect_pupils():
    import time
    good = []
    for i in range(3, 151):
        time.sleep(1.0)
        frame = cv2.imread('./out/' + str(i) + '.jpg')
        # cv2.imshow('image',frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        face_frame = detect_faces(frame, face_cascade)
        if face_frame is not None:
            # cv2.imshow('image',face_frame)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            eyes = detect_eyes(face_frame, eye_cascade)
            if eyes == [None, None]:
                print("Did not find 2 eyes")
                continue
            points = []
            output = face_frame
            for eye_data in eyes:
                if eye_data is None:
                    continue
                eye, location = eye_data
                # cv2.imshow('image',eye)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                if eye is not None:
                    threshold = 50 # SET
                    eye, height = cut_eyebrows(eye)
                    # cv2.imshow('image',eye)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    keypoint = blob_process(eye, threshold, detector)
                    if len(keypoint) > 0:
                        keypoint = keypoint[0]
                    else:
                        continue
                    keypoint.pt = (keypoint.pt[0] + location[0], keypoint.pt[1] + location[1] + height)
                    keypoints = [keypoint]
                    points += keypoints
                    output = cv2.drawKeypoints(output, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            if len(points) >= 2:
                print("Two pupils found: " + str(i))
                cv2.imshow('image',output)
                cv2.waitKey(500)
                # time.sleep(0.5)
                cv2.destroyAllWindows()
                good.append(i)
        cv2.destroyAllWindows()
    print(good)

def main():
    detect_pupils()


    # threshold = 15
    # face = cv2.imread('f.jpg')
    # eyes = detect_eyes(face, eye_cascade)
    # for eye in eyes:
    #     if eye is not None:
    #         base_eye = cut_eyebrows(eye)
    #         keypoints = blob_process(base_eye, threshold, detector)
    #         eye = cv2.drawKeypoints(base_eye, keypoints, base_eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #         cv2.imshow('image',eye)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()


    # threshold = 10
    # for threshold in range(5, 25, 5):
    #     print("Threshold: " + str(threshold))
    #     frame = cv2.imread('face.jpg')
        
    #     print("Full image")
    #     cv2.imshow('image',frame)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    #     face_frame = detect_faces(frame, face_cascade)
    #     assert face_frame is not None, "face_frame is None"
    #     print("Face")
    #     cv2.imshow('image',face_frame)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     eyes = detect_eyes(face_frame, eye_cascade)
    #     for eye in eyes:
    #         if eye is not None:
    #             eye = cut_eyebrows(eye)
    #             cv2.imshow('image',eye)
    #             cv2.waitKey(0)
    #             cv2.destroyAllWindows()
    #             keypoints = blob_process(eye, threshold, detector)
    #             eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #             cv2.imshow('image',eye)
    #             cv2.waitKey(0)
    #             cv2.destroyAllWindows()

    # cv2.destroyAllWindows()

    # frame = cv2.imread('image.jpg')
    # face_frame = detect_faces(frame, face_cascade)
    # if face_frame is not None:
    #     cv2.imshow('image',face_frame)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     eyes = detect_eyes(face_frame, eye_cascade)
    #     for eye in eyes:
    #         cv2.imshow('image',eye)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
    #         if eye is not None:
    #             threshold = 60 # SET
    #             eye = cut_eyebrows(eye)
    #             cv2.imshow('image',eye)
    #             cv2.waitKey(0)
    #             cv2.destroyAllWindows()
    #             keypoints = blob_process(eye, threshold, detector)
    #             eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #             cv2.imshow('image',eye)
    #             cv2.waitKey(0)
    #             cv2.destroyAllWindows()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cap = cv2.VideoCapture(0)
    # cv2.namedWindow('image')
    # while True:
    #     threshold = 5
    #     _, frame = cap.read()
    #     face_data = detect_faces(frame, face_cascade)
    #     if face_data is not None:
    #         face_frame, face_outline = face_data
    #     else:
    #         face_outline = None
    #         face_frame = None
    #     print(face_frame)
    #     if face_frame is not None:
    #         eyes = detect_eyes(face_frame, eye_cascade)
    #         for eye in eyes:
    #             if eye is not None:
    #                 eye = cut_eyebrows(eye)
    #                 keypoints = blob_process(eye, threshold, detector)
    #                 eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #     if face_data is not None:
    #         cv2.imwrite('face.jpg', frame)
    #         cv2.rectangle(frame,(face_outline[0], face_outline[1]), \
    #                             (face_outline[0]+face_outline[2],face_outline[1]+face_outline[3]),(255,255,0),2)
    #     cv2.imshow('image', frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
