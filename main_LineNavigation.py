import cv2
import time
import numpy as np
from Image_Processing import WindowDetection, LineDetection
from Simulation import UnityEnvironmentWrapper
from Line_Navigation import LineNavigation

class Main:
    def __init__(self, unity_env_path):
        self.unity_env = UnityEnvironmentWrapper(unity_env_path)
        self.front_camera_window_detector = LineNavigation(
            akp=0.1, akd=0, aki=0,
            xkp=0.1, xkd=0, xki=0) 
        self.downward_camera_line_navigation = WindowDetection()  

    def run(self):
        observation = self.unity_env.reset()

        try:
            while True:
                downward_camera_image_rgb = np.transpose(observation[0], (1, 2, 0)) 
                downward_camera_image_bgr = cv2.cvtColor(downward_camera_image_rgb, cv2.COLOR_RGB2BGR)
                print() 
                # print('type of downward cam is:', type(downward_camera_image_bgr))
                # print('shape od down is: ',np.shape(downward_camera_image_bgr))
                # print('down is:', downward_camera_image_bgr)

                front_camera_image_rgb = np.transpose(observation[1], (1, 2, 0))
                front_camera_image_bgr = cv2.cvtColor(front_camera_image_rgb, cv2.COLOR_RGB2BGR) 
                print()
                # print('type of front cam is:', type(front_camera_image_bgr))
                # print('shape of front is: ',np.shape(front_camera_image_bgr))
                # print('front is:', front_camera_image_bgr)


                cv2.imshow("Front Camera Real-Time View", front_camera_image_bgr)
                cv2.imshow("Downward Camera Real-Time View", downward_camera_image_bgr)
                
                

                # Detect corners in the front camera image
                # self.front_camera_window_detector.finding_corners(downward_camera_image_bgr)
                
                #debugged: first using LineDetection to detect line and then process it for LineNavigation
                line_detection = LineDetection()
                if front_camera_image_bgr.ndim == 2:
                    front_camera_image_bgr = cv2.cvtColor(front_camera_image_bgr, cv2.COLOR_GRAY2BGR)
                line_detection.line_detector(front_camera_image_bgr)
                
                
                # control_action = self.downward_camera_line_navigation.process_frame(front_camera_image_bgr, neighbor_num=5)
                
                if line_detection.result is None:
                    print("No line detected. Hovering")
                    action = [0., 0., 0., 0.]  # Hover/Stop
                else:
                # Use LineNavigation to process the result from LineDetection
                    control_action = self.front_camera_window_detector.process_frame(line_detection.result, neighbor_num=5)
                    print(line_detection.result)
                print(control_action)
                if control_action == "constant pitch":
                    action = [0., 0.2, 0., 0.]  # Move forward
                elif control_action == "Positive Yaw adjustment required":
                    action = [0., 0., 0.2, 0.]  # Adjust yaw
                elif control_action == "Negative Yaw adjustment required":
                    action = [0., 0., -0.2, 0.]  # Adjust yaw
                elif control_action == "Roll":
                    action = [0.2, 0., 0., 0.]  # Adjust roll
                else:
                    action = [0., 0., 0., 0.]  # Hover

                observation, reward, done, info = self.unity_env.step(action)

                if done:
                    observation = self.unity_env.reset()
                    
                time.sleep(0.05)
                
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        finally:
            self.unity_env.close()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    path = r'C:\Users\Lenovo\Desktop\ip-fira1403\Gate_Simulation\Gate_Simulation\GSim\Xerox_UAV.exe'
    app = Main(path)
    app.run()
