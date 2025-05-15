import numpy as np
import cv2
import time
from Image_Processing import LineDetection

class LineNavigation:
    def __init__(self, akp, akd, aki, xkp, xkd, xki):
        self.rows = None
        self.cols = None
        self.first_point = None
        self.second_point = None
        self.akp, self.akd, self.aki = akp, akd, aki
        self.xkp, self.xkd, self.xki = xkp, xkd, xki
        self.pre_d_a = 0
        self.pre_i_a = 0
        self.pre_d_x = 0
        self.pre_i_x = 0
        self.dt = 0.001
        #debugged: i added the line detection part before processing it for navigation
        self.line_detector = LineDetection()  


    def get_shape(self, frame):
        self.rows, self.cols = frame.shape[:2]

    def find_points(self, frame, neighbor_num):
        n = neighbor_num
        # rows_with_one = np.nonzero(np.any(frame == 1, axis=1))[0].tolist()
        # columns_with_one = np.nonzero(np.any(frame == 1, axis=0))[0].tolist()
        
        #for debugging NaN value that i got from: first_med_white_row
        
        rows_with_one = np.nonzero(np.any(frame == 255, axis=1))[0].tolist()
        print('rows are: ', rows_with_one)
        columns_with_one = np.nonzero(np.any(frame == 255, axis=0))[0].tolist()
        print('cols are: ', columns_with_one)
        
        first_white_row_roi = rows_with_one[-n:]
        first_white_col_roi = columns_with_one[:n]
        first_med_white_row = int(np.floor(np.median(first_white_row_roi)))
        first_med_white_col = int(np.floor(np.median(first_white_col_roi)))
        
        second_white_row_roi = rows_with_one[-2 * n : -n]
        second_white_col_roi = columns_with_one[n : 2 * n]
        second_med_white_row = int(np.floor(np.median(second_white_row_roi)))
        second_med_white_col = int(np.floor(np.median(second_white_col_roi)))

        self.first_point = np.array([first_med_white_row, first_med_white_col])
        self.second_point = np.array([second_med_white_row, second_med_white_col])
        return self.first_point, self.second_point

    def calculate_angle(self, var1, var2):
        delta = var2 - var1
        if 0 < delta[1] < 1:
            return "Recognize horizontal line"  # Placeholder for horizontal line handling
        else:
            slope = -delta[0] / delta[1]
            angle = 90 - np.degrees(np.arctan(abs(slope)))
            angle = angle if slope > 0 else -angle
        if abs(angle) < 2:  
            return 0
        return angle

    def calculate_x_offset(self, frame):
        row_offset = int(np.floor((self.first_point[0] + self.second_point[0]) / 2))
        white_col_list = np.where(frame[row_offset, :] == 255)[0]
        min_col = min(white_col_list)
        max_col = max(white_col_list)
        line_center = int(np.floor((min_col + max_col) / 2))
        frame_center = int(np.floor(self.cols / 2))
        x_offset = line_center - frame_center
        return x_offset

    def pid_control(self, angle, x_offset):
        ang_pterm = self.akp * angle
        ang_dterm = self.akd * ((angle - self.pre_d_a) / self.dt)
        ang_iterm = self.aki * (angle * self.dt) + self.pre_i_a
        self.pre_d_a, self.pre_i_a = angle, angle

        x_pterm = self.xkp * x_offset
        x_dterm = self.xkd * ((x_offset - self.pre_d_x) / self.dt)
        x_iterm = self.xki * (x_offset * self.dt) + self.pre_i_x
        self.pre_d_x, self.pre_i_x = x_offset, x_offset

        final_angle = ang_pterm + ang_dterm + ang_iterm
        final_x_offset = x_pterm + x_dterm + x_iterm
        if abs(final_angle) < 2:  
            final_angle = 0
        return final_angle, final_x_offset

    def process_frame(self, frame, neighbor_num):
        self.line_detector.line_detector(frame)  
        if self.line_detector.result is None:
            return "No line detected"
        #this should be in a while!!
        frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
        self.get_shape(frame)
        self.find_points(frame, neighbor_num)
        angle = self.calculate_angle(self.first_point, self.second_point)
        x_offset = self.calculate_x_offset(frame)

        if 0 < angle < 5 and abs(x_offset) < 10:
            return "constant pitch"

        new_angle, new_x_offset = self.pid_control(angle, x_offset)
        if abs(new_angle) >= 5:
            print(new_angle)
            print(new_x_offset)
            return "Positive Yaw adjustment required"
        if abs(new_angle) <= -5:
            print(new_angle)
            print(new_x_offset)
            return "Negative Yaw adjustment required"
        if abs(new_x_offset) > 10:
            return "Roll"  