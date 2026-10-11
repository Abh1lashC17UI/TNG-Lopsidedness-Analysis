import pyautogui
import pygetwindow as gw
from datetime import datetime
import time

SAVE_FOLDER = r"P:\ScriptUpdates" # My drive was in P:, change if yours is different
DELAY = 600  # 10 min

'''
Script to remotely observe the updates to a running code.
Lots of API queries often cause the runtime to exceed 24 hours.
'''

window_title_keyword = "Kate"   # adjust if your title is different

while True:
    # Find all windows whose title contains the keyword
    windows = [w for w in gw.getAllWindows() if window_title_keyword.lower() in w.title.lower()]

    if windows:
        win = windows[0]

        # Bring it to the front (if possible)
        try:
            win.activate()
            time.sleep(0.3)
        except:
            pass

        # bounding box
        left, top, right, bottom = win.left, win.top, win.right, win.bottom

        # screenshot window region
        img = pyautogui.screenshot(region=(left, top, right-left, bottom-top))

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        img.save(f"{SAVE_FOLDER}\\kate_{timestamp}.png")
    else:
        # fallback full screen capture
        img = pyautogui.screenshot()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        img.save(f"{SAVE_FOLDER}\\screen_{timestamp}.png")

    time.sleep(DELAY)
