import cv2
import numpy as np
from Image_Processing import WindowDetection, LineDetection
from Simulation import UnityEnvironmentWrapper


class Main:
    def __init__(self, unity_env_path):
        self.unity_env = UnityEnvironmentWrapper(unity_env_path)
        self.front_camera_window_detector = LineDetection()    
        self.downward_camera_line_detector = WindowDetection()  

    def run(self):
        observation = self.unity_env.reset()

        try:
            while True:
                # Processing the downward-facing camera image
                downward_camera_image_rgb = np.transpose(observation[0], (1, 2, 0)) 
                downward_camera_image_bgr = cv2.cvtColor(downward_camera_image_rgb, cv2.COLOR_RGB2BGR) 
                
                # Processing the front-facing camera image
                front_camera_image_rgb = np.transpose(observation[1], (1, 2, 0))
                front_camera_image_bgr = cv2.cvtColor(front_camera_image_rgb, cv2.COLOR_RGB2BGR) 

                # Display real-time camera feeds
                cv2.imshow("Front Camera Real-Time View", front_camera_image_bgr)
                cv2.imshow("Downward Camera Real-Time View", downward_camera_image_bgr)

                # Detect corners in the front camera image
                self.front_camera_window_detector.line_detector(front_camera_image_bgr)
                
                # Detect lines in the downward camera image
                self.downward_camera_line_detector.finding_corners(downward_camera_image_bgr)

                # action = self.unity_env.env.action_space.sample()        ##random action
                action = [0., 0.5, 0., 0.]
                observation, reward, done, info = self.unity_env.step(action)

                if done:
                    observation = self.unity_env.reset()
                    
                # Exit if the ESC key is pressed
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        finally:
            self.unity_env.close()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    path = r'C:\Users\alisa\Desktop\IP\IP\GateSimulation\Gate_Simulation\GSim'
    app = Main(path)
    app.run()
