import re
import time
import cv2
import gym
import numpy as np
from gym import spaces
from PIL import Image
import mss
from pynput.keyboard import Controller
from pynput.mouse import Controller as MouseController, Button
import pywinctl as pwc
import pytesseract

class CelticHeroesEnv(gym.Env):
    """
    Gym environment for Celtic Heroes via automatic window-based capture.
    Actions: MultiBinary(3) for [forward, turn_left, turn_right].
    Observations: 128×128 RGB frames cropped to center square.
    Rewards: +1 on kill, -1 on death; uses external detectors.
    Automatically locates, focuses, and resizes the BlueStacks window to 16:9,
    then captures and crops a centered square viewport each frame.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 window_title="BlueStacks",
                 fps=10):
        super().__init__()
        self.window_title = window_title
        self.fps = fps
        self.interval = 1.0 / fps

        # Multi-hot actions: [forward, turn_left, turn_right]
        self.action_space = spaces.MultiBinary(3)

        # Observation: 128×128×3 RGB
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(128, 128, 3), dtype=np.uint8
        )

        # Input and capture
        self.keyboard = Controller()
        self.sct = mss.mss()

        self.prev_kill_id = 0

        self.W = 1280
        self.H = 720

        self.mouse = MouseController()

        # Locate, focus, resize, and set capture region
        self._set_capture_region()

    def _set_capture_region(self):
        # Find BlueStacks window
        windows = pwc.getWindowsWithTitle(
            self.window_title, condition=pwc.Re.CONTAINS, flags=pwc.Re.IGNORECASE
        )
        if not windows:
            raise RuntimeError(f"No window found with title containing '{self.window_title}'")
        win = windows[0]
        # Focus window
        try:
            win.activate()
            time.sleep(0.2)
        except Exception:
            pass
        # Resize to 16:9 (e.g. 1280×720)
        try:
            win.resizeTo(self.W, self.H)
        except Exception:
            pass
        time.sleep(0.5)
        # Get client area
        left, top, right, bottom = win.getClientFrame()
        width, height = right - left, bottom - top
        self.capture_region = (left, top, width, height)

    # indendedCoords - tuple (left, top, right, bottom)
    # returns actualCoords
    def _normalize_dimensions(self, actualW, indendedCoords):
        scale = actualW / self.W
        left, top, right, bottom = indendedCoords
        return (left * scale, top * scale, right * scale, bottom * scale)

    def _grab_frame(self):
        left, top, width, height = self.capture_region
        img = self.sct.grab({
            'left': left, 'top': top,
            'width': width, 'height': height
        })
        frame = Image.frombytes('RGB', img.size, img.rgb)
        # Center-crop to square
        w0, h0 = frame.size
        side = min(w0, h0)
        left_crop = (w0 - side) // 2
        top_crop = (h0 - side) // 2

        kill_img = frame.crop(self._normalize_dimensions(w0, (100, 120, 340, 370)))
        died_img = frame.crop(self._normalize_dimensions(w0, (500, 200, 750, 270)))
        frame = frame.crop((left_crop, top_crop, left_crop + side, top_crop + side))
        # Resize to model input
        frame = frame.resize((128, 128), Image.BILINEAR)
        return np.array(frame), np.array(kill_img), np.array(died_img)

    def _parse_kill_id(self, img_rgb: np.ndarray):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        text = pytesseract.image_to_string(thresh, config="--psm 6").lower()
        gold_m = re.search(r"(\d{2})[.\s]*gold", text)
        xp_m = re.search(r"(\d{2})[.\s]*xp", text)

        gold = gold_m.group(1) if gold_m else ""
        xp = xp_m.group(1) if xp_m else ""

        # Only return an ID if at least one match found
        return f"{gold}_{xp}" if (gold or xp) else None

    def _detect_died(self, img_rgb: np.ndarray):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        text = pytesseract.image_to_string(thresh, config="--psm 6").lower()
        return 'died' in text

    def reset(self):
        left, top, w, h = self.capture_region
        dx, dy = 650, 330
        click_x = left + dx
        click_y = top + dy
        self.mouse.position = (click_x, click_y)
        time.sleep(0.1)
        self.mouse.click(Button.left)
        time.sleep(1.0)

        self.prev_kill_id = 0
        return self._grab_frame()

    def step(self, action):
        # action is [forward, left, right]
        # Map bits to keys
        if action[0]:  # forward
            self.keyboard.press('w')
        else:
            self.keyboard.release('w')
        if action[1]:  # turn left
            self.keyboard.press('q')
        else:
            self.keyboard.release('q')
        if action[2]:  # turn right
            self.keyboard.press('e')
        else:
            self.keyboard.release('e')

        time.sleep(self.interval)

        obs, killimg, diedimg = self._grab_frame()

        kill_id = self._parse_kill_id(killimg)
        if kill_id and kill_id != self.prev_kill_id:
            reward = 1.0
            self.prev_kill_id = kill_id
        else:
            reward = 0.0

        died = self._detect_died(diedimg)
        if died: reward -= 3

        done = died
        return obs, reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        self.sct.close()
        super().close()

# Example usage:
# env = CelticHeroesEnv(window_title="BlueStacks", fps=10,
#                       reward_detector=my_kill_detector,
#                       death_detector=my_death_detector)
# obs = env.reset()
# for _ in range(500):
#     action = env.action_space.sample()
#     obs, r, d, _ = env.step(action)
#     if d:
#         obs = env.reset()
