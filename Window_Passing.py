import numpy as np
import cv2
import time

class Window_Passing:
    def __init__(self, kp, kd, ki, cent_threshold, tilt_threshold):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.cent_threshold = cent_threshold
        self.tilt_threshold = tilt_threshold
        self.pre_d = np.array([0, 0])
        self.pre_i = np.array([0, 0])

    def getting_shape(self, frame):
        self.rows = frame.shape[0]
        self.cols = frame.shape[1]
        return self.rows, self.cols

    def finding_corners(self, frame):
        blurred_img = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv_roi = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)
        lower_color = np.array([0, 102, 0], np.uint8)
        upper_color = np.array([179, 255, 163], np.uint8)
        mask = cv2.inRange(hsv_roi, lower_color, upper_color)
        _, thresh = cv2.threshold(mask, 95, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        edges = cv2.Canny(closing, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour = None
        largest_area = 0
        approx_corners = []

        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > largest_area:
                    largest_area = area
                    largest_contour = approx
                    approx_corners = approx

        if largest_contour is not None:
            sorted_corners = self.sort_corners(approx_corners)
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 3)

            corner_coords = []
            for i, corner in enumerate(sorted_corners):
                x, y = corner[0]
                corner_coords.append((x, y))
                cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)
                cv2.putText(frame, f'Corner {i+1}: ({x}, {y})', (x+15, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # cv2.imshow('Masked Image', cv2.resize(masked_img, None, fx=0.4, fy=0.4))
            cv2.imshow('Masked Image', mask)

            # cv2.imshow('Detected Corners', cv2.resize(frame1, None, fx=0.4, fy=0.4))
            cv2.imshow('Detected Corners', frame)

            cv2.waitKey(1)

            print("Coordinates of the 4 corners:", corner_coords)

            
            self.corner_coords = [(corner[0][0], corner[0][1]) for corner in sorted_corners]
        else:
            self.corner_coords = []

        return self.corner_coords

    @staticmethod
    def sort_corners(corners):
        corners = sorted(corners, key=lambda x: x[0][1])
        top_corners = sorted(corners[:2], key=lambda x: x[0][0])
        bottom_corners = sorted(corners[2:], key=lambda x: x[0][0])
        return top_corners + bottom_corners

    def tilting_correction(self, corner_coordinates):
        if abs(abs(corner_coordinates[0][1] - corner_coordinates[2][1]) - abs(corner_coordinates[1][1] - corner_coordinates[3][1])) > self.tilt_threshold:
            if abs(corner_coordinates[0][1] - corner_coordinates[2][1]) > abs(corner_coordinates[1][1] - corner_coordinates[3][1]):
                print("Yaw to the left")
                return [0., 0., -0.2, 0.]
            else:
                print("Yaw to the right")
                return [0., 0., 0.2, 0.]
                
    def offset_calculating(self, corner_coordinates):
        frame_center = np.array([self.rows // 2, self.cols // 2])
        window_center = np.array([
            (corner_coordinates[0][0] + corner_coordinates[1][0]) // 2,
            (corner_coordinates[0][1] + corner_coordinates[2][1]) // 2])
        offset_arr = window_center - frame_center
        offset = np.linalg.norm(offset_arr)
        
        return offset

    def window_pass_pid(self, distance_arr, dt):
        p_term = self.kp * distance_arr
        d_term = self.kd * (distance_arr - self.pre_d) / dt
        i_term = self.ki * (distance_arr * dt) + self.pre_i

        correction = p_term + d_term + i_term
        self.pre_d = distance_arr
        self.pre_i = i_term

        return correction

    def process_frame(self, frame):
        t1 = time.time()

        self.getting_shape(frame)
        corner_array = self.finding_corners(frame)

        if not corner_array:
            print("No rectangular window found.")
            return [0., 0., 0., 0.]

        self.tilting_correction(corner_array)
        cent_offset = self.offset_calculating(corner_array)
        print('cent offset is:', cent_offset)
        print('self.cent_threshold is:', self.cent_threshold)
        while cent_offset > self.cent_threshold:
            t2 = time.time()
            dt = max(t2 - t1, 0.01)
            correction_value = np.linalg.norm(self.window_pass_pid(cent_offset, dt))
            print('correction value is:', correction_value)
            
            cent_offset = correction_value

        print("Offset corrected. Maintaining constant pitch.")
        return [0., 0.2, 0., 0.]