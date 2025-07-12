import re
import time
import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image
import mss
from pynput.keyboard import Controller
from pynput.mouse import Controller as MouseController, Button
import pywinctl as pwc
import pytesseract

class CelticHeroesEnv(gym.Env):
    """
    Gymnasium environment for Celtic Heroes via automatic window‐based capture.
    Actions: MultiBinary(3) for [forward, turn_left, turn_right].
    Observations: 128×128 RGB frames cropped to center square.
    Rewards: +1 on kill, -3 on death.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 window_title="BlueStacks",
                 fps=10,
                 render_mode=None):
        super().__init__(render_mode=render_mode)
        self.window_title = window_title
        self.fps = fps
        self.interval = 1.0 / fps

        # Multi-hot actions: [forward, turn_left, turn_right]
        self.action_space = spaces.MultiBinary(3)

        # Observation: 128×128×3 RGB
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(128, 128, 3), dtype=np.uint8
        )

        # Input & capture setup
        self.keyboard = Controller()
        self.mouse = MouseController()
        self.sct = mss.mss()
        self.prev_kill_id = None

        # Window target size & capture region
        self.W, self.H = 1280, 720
        self._set_capture_region()

    def _set_capture_region(self):
        windows = pwc.getWindowsWithTitle(
            self.window_title, condition=pwc.Re.CONTAINS, flags=pwc.Re.IGNORECASE
        )
        if not windows:
            raise RuntimeError(f"No window found with title containing '{self.window_title}'")
        win = windows[0]

        # Focus & resize
        try:
            win.activate(); time.sleep(0.2)
            win.resizeTo(self.W, self.H)
        except Exception:
            pass
        time.sleep(0.5)

        left, top, right, bottom = win.getClientFrame()
        self.capture_region = (
            left, top,
            right - left, bottom - top
        )

    def _normalize_dimensions(self, actualW, coords):
        scale = actualW / self.W
        return tuple(int(c * scale) for c in coords)

    def _grab_frame(self):
        left, top, w, h = self.capture_region
        img = self.sct.grab({'left': left, 'top': top, 'width': w, 'height': h})
        frame = Image.frombytes('RGB', img.size, img.rgb)

        # Crop kill & death regions, then center‐crop & resize
        w0, h0 = frame.size
        kill_img = frame.crop(self._normalize_dimensions(w0, (100, 120, 340, 370)))
        died_img = frame.crop(self._normalize_dimensions(w0, (500, 200, 750, 270)))

        side = min(w0, h0)
        lc = (w0 - side) // 2
        tc = (h0 - side) // 2
        frame = frame.crop((lc, tc, lc + side, tc + side))
        frame = frame.resize((128, 128), Image.BILINEAR)

        return np.array(frame), np.array(kill_img), np.array(died_img)

    def _parse_kill_id(self, img_rgb):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        text = pytesseract.image_to_string(thresh, config="--psm 6").lower()
        gm = re.search(r"(\d{2})[.\s]*gold", text)
        xm = re.search(r"(\d{2})[.\s]*xp", text)
        gold = gm.group(1) if gm else ""
        xp = xm.group(1) if xm else ""
        return f"{gold}_{xp}" if (gold or xp) else None

    def _detect_died(self, img_rgb):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        text = pytesseract.image_to_string(thresh, config="--psm 6").lower()
        return 'died' in text

    def reset(self, *, seed=None, options=None):
        # optional: super().reset(seed=seed) to seed RNGs if needed
        left, top, _, _ = self.capture_region
        dx, dy = 650, 330
        self.mouse.position = (left + dx, top + dy)
        time.sleep(0.1)
        self.mouse.click(Button.left)
        time.sleep(1.0)

        self.prev_kill_id = None
        obs, _, _ = self._grab_frame()
        info = {}
        return obs, info

    def step(self, action):
        # press/release keys
        if action[0]: self.keyboard.press('w')
        else:            self.keyboard.release('w')
        if action[1]: self.keyboard.press('q')
        else:            self.keyboard.release('q')
        if action[2]: self.keyboard.press('e')
        else:            self.keyboard.release('e')

        time.sleep(self.interval)
        obs, kill_img, died_img = self._grab_frame()

        # compute reward
        kill_id = self._parse_kill_id(kill_img)
        reward = 1.0 if (kill_id and kill_id != self.prev_kill_id) else 0.0
        if kill_id: self.prev_kill_id = kill_id

        if self._detect_died(died_img):
            reward -= 3
            terminated = True
        else:
            terminated = False

        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        # implement if you want to display frames
        pass

    def close(self):
        self.sct.close()
        super().close()

# Example usage with Gymnasium:
# import gymnasium as gym
# env = CelticHeroesEnv(window_title="BlueStacks", fps=10)
# obs, info = env.reset(seed=42)
# for _ in range(500):
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()
