import numpy as np
import cv2
import time

class WindowDetection:
    def __init__(self):
        pass

    def finding_corners(self, frame1):
        blurred_img = cv2.GaussianBlur(frame1, (5, 5), 0)
        hsv_roi = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)
        # print('shape of hsv roi is: ', np.shape(hsv_roi))
        # print('hsv roi is: ', hsv_roi)
        lower_color = np.array([0, 102, 0], np.uint8)
        upper_color = np.array([179, 255, 163], np.uint8)
        mask = cv2.inRange(hsv_roi, lower_color, upper_color)
        masked_img = cv2.bitwise_and(frame1, frame1, mask=mask)

        _, thresh = cv2.threshold(mask, 95, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        edges = cv2.Canny(closing, 50, 150)
        cv2.imshow('edges',edges)
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
            cv2.drawContours(frame1, [largest_contour], -1, (0, 255, 0), 3)

            corner_coords = []
            for i, corner in enumerate(sorted_corners):
                x, y = corner[0]
                corner_coords.append((x, y))
                cv2.circle(frame1, (x, y), 10, (255, 0, 0), -1)
                cv2.putText(frame1, f'Corner {i+1}: ({x}, {y})', (x+15, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # cv2.imshow('Masked Image', cv2.resize(masked_img, None, fx=0.4, fy=0.4))
            cv2.imshow('Masked Image', masked_img)

            # cv2.imshow('Detected Corners', cv2.resize(frame1, None, fx=0.4, fy=0.4))
            cv2.imshow('Detected Corners', frame1)

            cv2.waitKey(1)

            print("Coordinates of the 4 corners:", corner_coords)
        else:
            print("No rectangular window found.")

    def sort_corners(self, corners):
        corners = sorted(corners, key=lambda x: x[0][1])
        top_corners = sorted(corners[:2], key=lambda x: x[0][0])
        bottom_corners = sorted(corners[2:], key=lambda x: x[0][0])
        return top_corners + bottom_corners


class LineDetection:
    def __init__(self):
        self.img_smp = None
        self.masked_img = None
        self.opened = None
        self.closed = None
        self.dilated = None
        self.result = None
        self.edges = None
        self.lines = None

    def line_detector(self, frame):
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        self.img_smp = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        self.masked_img = cv2.inRange(self.img_smp, np.array([0, 0, 0]), np.array([170, 120, 200]))

        kernel_open = np.ones((8, 8), np.int8)
        kernel_close = np.ones((3, 3), np.int8)
        kernel_dilate = np.ones((20, 20), np.uint8)

        self.opened = cv2.morphologyEx(self.masked_img, cv2.MORPH_OPEN, kernel_open)
        self.closed = cv2.morphologyEx(self.opened, cv2.MORPH_CLOSE, kernel_dilate)

        contours1, _ = cv2.findContours(self.closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours1:
            largest_contour1 = max(contours1, key=cv2.contourArea)
            mask1 = np.zeros_like(self.closed)
            cv2.drawContours(mask1, [largest_contour1], -1, 255, thickness=cv2.FILLED)
            result_after_closing = cv2.bitwise_and(mask1, self.closed, mask=mask1)

            self.dilated = cv2.dilate(result_after_closing, kernel_dilate, iterations=2)
            contours, _ = cv2.findContours(self.dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                mask = np.zeros_like(self.dilated)
                cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                self.result = cv2.bitwise_and(self.dilated, self.dilated, mask=cv2.bitwise_not(mask))

                self.edges = cv2.Canny(self.result, 40, 60, apertureSize=3)
            else:
                self.edges = None
        else:
            self.edges = None
        
        self.show_results()  

    def show_results(self):
        if self.edges is not None:
            cv2.imshow('Edges',cv2.resize(self.edges, None, fx=0.4, fy=0.4))
            cv2.imshow('Edges',self.edges)
            cv2.imshow('result',self.result)
        else:
            print("No edges detected.")

    def get_edges(self):
        return self.edges


 