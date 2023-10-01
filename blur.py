import pyautogui
import cv2
import numpy as np
import time

# Take a screenshot
screenshot = pyautogui.screenshot()

# Convert the screenshot to a NumPy array
screenshot_np = np.array(screenshot)

# Apply Gaussian blur to the screenshot
blurred_image = cv2.GaussianBlur(screenshot_np, (21, 21), 0)

# Create a window to display the blurred image
cv2.imshow("Blurred Screen", blurred_image)

# Wait for a few seconds
time.sleep(5)

# Close the window
cv2.destroyAllWindows()
