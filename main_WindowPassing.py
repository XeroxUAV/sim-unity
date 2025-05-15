import cv2
import time
import numpy as np
from Simulation import UnityEnvironmentWrapper
from Window_Passing import Window_Passing

class Main:
    def __init__(self, unity_env_path):
        self.unity_env = UnityEnvironmentWrapper(unity_env_path)
        time.sleep(5)
        self.window_passing = Window_Passing(
            kp=0.5, kd=0.5, ki=0.5,
            cent_threshold=20, tilt_threshold=15
        )

    def run(self):
        observation = self.unity_env.reset()
        print("Observation shape:", observation[0].shape)

        try:
            while True:
                front_camera_image_rgb = np.transpose(observation[0], (1, 2, 0))
                front_camera_image_bgr = cv2.cvtColor(front_camera_image_rgb, cv2.COLOR_RGB2BGR)

                cv2.imshow("Front Camera Real-Time View", front_camera_image_bgr)

                # self.window_passing.process_frame(front_camera_image_bgr)
                action = self.window_passing.process_frame(front_camera_image_bgr)

                if not self.window_passing.corner_coords:
                    print("No window detected. Hovering.")
                    action = [0., 0., 0., 0.]  # Hovering
                else:
                    tilt_action = self.window_passing.tilting_correction(self.window_passing.corner_coords)
                    if tilt_action:
                        action = tilt_action  # Apply yaw correction

                print(f"Final Action Sent to Unity: {action}")

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
    path = r'C:\Users\alisa\Desktop\IP\IP\GateSimulation\Gate_Simulation\GSim\Xerox_UAV.exe'
    app = Main(path)
    app.run()
