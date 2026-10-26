import cv2

from .detector import FaceDetector


def open_camera():
    camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if camera.isOpened():
        return camera

    camera.release()
    return cv2.VideoCapture(0)


def main():
    camera = open_camera()
    if not camera.isOpened():
        raise RuntimeError("Could not open camera")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = FaceDetector()

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Could not read camera frame")
                break

            display = frame.copy()
            face = detector.track_face(frame)
            if face is not None:
                x, y, width, height = face.rect
                cv2.rectangle(
                    display,
                    (x, y),
                    (x + width - 1, y + height - 1),
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Aligned Face", face.aligned)

            cv2.imshow("Camera", display)
            key = cv2.waitKey(1) & 255
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
