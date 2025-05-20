import numpy as np
import cv2

class WindowPassing:
    def __init__(self, kp=0, kd=0, ki=0, cent_threshold=20, tilt_threshold=15):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.cent_threshold = cent_threshold
        self.tilt_threshold = tilt_threshold
        self.corner_coords = []
        self.cols = 0
        self.rows = 0

    def get_frame_shape(self, frame):
        self.rows, self.cols = frame.shape[:2]

    def sort_corners_from_points(self, points):
        points = sorted(points, key=lambda p: p[1])  # sort by Y
        top = sorted(points[:2], key=lambda p: p[0])
        bottom = sorted(points[2:], key=lambda p: p[0])
        return top + bottom

    def detect_colored_corners(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # HSV range for your window corners
        lower = np.array([20, 63, 0])
        upper = np.array([52, 235, 154])
        mask = cv2.inRange(hsv, lower, upper)

        # Clean mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        corners = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < 1500:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                circularity = 4 * np.pi * area / (cv2.arcLength(cnt, True) ** 2)
                if circularity > 0.6:
                    center = (int(x), int(y))
                    corners.append(center)
                    cv2.circle(frame, center, int(radius), (255, 0, 0), 2)

        for i, pt in enumerate(corners):
            cv2.putText(frame, f"{i+1}", pt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if len(corners) == 4:
            sorted_pts = self.sort_corners_from_points(corners)
            cv2.polylines(frame, [np.array(sorted_pts)], isClosed=True, color=(0, 255, 0), thickness=2)
            return sorted_pts

        return corners

    def process_frame(self, frame):
        self.get_frame_shape(frame)
        corners = self.detect_colored_corners(frame)
        self.corner_coords = corners
        return corners
