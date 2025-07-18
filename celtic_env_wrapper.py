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
                 max_steps = 1000):
        super().__init__()
        self.window_title = window_title
        self.fps = fps
        self.interval = 1.0 / fps

        self.max_steps = max_steps
        self.current_steps = 0

        # 0=idle, 1=forward, 2=turn_left, 3=turn_right
        self.action_space = spaces.Discrete(4)

        # Observation: 64×64×3 RGB
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
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
        frame = frame.resize((64, 64), Image.BILINEAR)

        return np.array(frame), np.array(kill_img), np.array(died_img)

    def _parse_kill_id(self, img_rgb):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        text = pytesseract.image_to_string(img_rgb, config="--psm 6").lower()
        gm = re.search(r"(\d{2})[.\s]*gold", text)
        xm = re.search(r"(\d{2})[.\s]*xp", text)
        gold = gm.group(1) if gm else ""
        xp = xm.group(1) if xm else ""
        if gold: print('kill gold:', gold)
        return f"{gold}" if (gold) else None

    def _detect_died(self, img_rgb):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        text = pytesseract.image_to_string(thresh, config="--psm 6").lower()
        return 'died' in text

    def reset(self, *, seed=None, options=None):
        # optional: super().reset(seed=seed) to seed RNGs if needed
        left, top, _, _ = self.capture_region
        self.mouse.position = (left + 650, top + 430)
        time.sleep(0.1)
        self.mouse.click(Button.left)
        time.sleep(2.0)
        self.mouse.position = (left + 525, top + 438)
        time.sleep(0.1)
        self.mouse.click(Button.left)

        self.current_steps = 0
        self.prev_kill_id = None
        obs, _, _ = self._grab_frame()
        info = {}
        return obs, info

    def step(self, action):
        # first, release everything
        for k in ('w', 'q', 'e'):
            self.keyboard.release(k)

        # press only the chosen key
        if action == 1:
            self.keyboard.press('w')
        elif action == 2:
            self.keyboard.press('q')
        elif action == 3:
            self.keyboard.press('e')
        # else action==0 → idle

        time.sleep(self.interval * 1.5)

        # release again so no sticky keys
        for k in ('w', 'q', 'e'):
            self.keyboard.release(k)

        obs, kill_img, died_img = self._grab_frame()

        # compute reward
        kill_id = self._parse_kill_id(kill_img)
        # reward = 1.0 if (kill_id and kill_id != self.prev_kill_id) else 0.0
        # if kill_id: self.prev_kill_id = kill_id
        reward = 1.0 if kill_id else 0.0

        if self._detect_died(died_img):
            print('died', time.time())
            reward -= 10
            terminated = True
        else:
            terminated = False

        if reward: print('reward received')

        truncated = False
        info = kill_img

        if self.current_steps >= self.max_steps:
            print('MAX STEPS REACHED')
            terminated = True

        return obs, reward, terminated, truncated, info

    def render(self):
        # implement if you want to display frames
        pass

    def close(self):
        self.sct.close()
        super().close()
