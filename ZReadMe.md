The detection of line and window is done in "Image_Processing.py" file, if you want to adjust the mask, edit the mentioned file.

Do not edit the "Simulation.py" file.

To adjust and edit line navigation code or the line following code, edit the "Line_Navigation.py" or "Window_Passing.py".

To test the line navigation code run "main_LineNavigation.py" file.

To test the window passing code run "main_WindowPassing.py" file.

To test both run "main.py" file.

*Know that the front and downward camera are inverted, it was like this when it was handed to me*

Before running any of the main make sure your VPN is activated.

Before running for the first time, do the steps below:

## 1. Create a Conda Environment
Start by creating a new Conda environment with Python 3.10.12. Run the following command in your terminal:

```bash
conda create -n IP_env python=3.10.12
conda activate IP_env
```
## 2. Install Requirements
After activating the environment, install the required packages listed in requirements.txt:

```bash
pip install -r requirements.txt
```
## 3. Update the Main File
In main.py(line 51), locate the section that requires the path to your simulation file. Update this part with the full path to the simulation file on your system.

## 4. Run the Main File
After setting the correct path in the main file, you can execute it to start the program.


