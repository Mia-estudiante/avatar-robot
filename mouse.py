import cv2
import numpy as np

class Selector:
    def __init__(self, window_name: str = "Select an Object"):
        self.window_name = window_name
        self.reset()
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_cb)

    def reset(self):
        self.start = None
        self.end = None
        self.done = False
        self.point = None
        self.box = None

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
            self.end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.start is not None:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.start is not None:
            self.end = (x, y)
            if self.start == self.end:
                self.point = np.array([[x, y]], dtype=np.int64)
            else:
                x1, y1 = self.start
                x2, y2 = self.end
                self.box = np.array([[[x1, y1], [x2, y2]]], dtype=np.float32)
            self.done = True

    def select(self, frame):
        img = frame.copy()
        if self.start is not None and not self.done:
            x1, y1 = self.start
            x2, y2 = self.end
            if (x1, y1) != (x2, y2):
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow(self.window_name, img)
        cv2.waitKey(1)

        if self.done:
            cv2.destroyWindow(self.window_name)

        return {"point": self.point, "box": self.box}