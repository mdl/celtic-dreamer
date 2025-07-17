#!/usr/bin/env python3
"""
Record human demo_data from live BlueStacks gameplay, saving each observation as an image
with metadata encoded in the filename and storing the transition arrays to demo_data.npz.
"""
import os
import time
import numpy as np

from PIL import Image
from pynput import keyboard

from celtic_env_wrapper import CelticHeroesEnv

# ——— Config ———
FPS = 10
INTERVAL = 1.0 / FPS
DATA_DIR = "./data"
OUT_NPZ = os.path.join(DATA_DIR, "record_data.npz")
IMG_DIR = os.path.join(DATA_DIR, "record_images")
KEYS = {"w", "q", "e"}

# ——— Globals for key capture ———
keys_pressed = set()
running = True

# ——— Key event handlers ———
def on_press(key):
    global running
    try:
        k = key.char.lower()
    except AttributeError:
        return
    if k in KEYS:
        keys_pressed.add(k)
    if key == keyboard.Key.esc:
        running = False
        return False

def on_release(key):
    try:
        k = key.char.lower()
    except AttributeError:
        return
    if k in keys_pressed:
        keys_pressed.remove(k)

# ——— Recorder loop ———
def record():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    # init env
    env = CelticHeroesEnv(window_title="BlueStacks", fps=FPS)

    # storage lists
    obs_list, act_list, rew_list, next_obs_list, done_list = [], [], [], [], []

    # prime first observation
    obs, kill_img, died_img = env._grab_frame()
    prev_kill = None
    prev_died = False

    idx = 0
    t0 = time.time()

    while running:
        # 1) Human action multi-hot vector
        action = np.array([
            1 if "w" in keys_pressed else 0,
            1 if "q" in keys_pressed else 0,
            1 if "e" in keys_pressed else 0,
        ], dtype=np.int8)

        # 2) Grab next frame
        next_obs, kill_img, died_img = env._grab_frame()

        # 3) Compute reward & done
        kill_id = env._parse_kill_id(kill_img)
        kill_reward = 1 if kill_id else 0

        # if kill_id and kill_id != prev_kill:
        #     print('New kill detected', time.time())
        #     kill_reward = 1.0
        #     prev_kill = kill_id
        # else:
        #     kill_reward = 0.0

        died = env._detect_died(died_img)
        if died and not prev_died:
            death_penalty = -10.0
            prev_died = True
        elif not died:
            death_penalty = 0.0
            prev_died = False
        else:
            death_penalty = 0.0

        done = bool(died)
        reward = kill_reward + death_penalty
        if reward: print('Reward received', reward, time.time())

        # 4) Save observation image with metadata in filename
        ts = int(time.time() * 1000)
        action_str = ''.join(str(int(b)) for b in action)
        fname = f"{idx:06d}_a{action_str}_r{reward:+.2f}_d{int(done)}_{ts}.png"
        Image.fromarray(obs).save(os.path.join(IMG_DIR, fname))

        # 5) Append transition
        obs_list.append(obs)
        act_list.append(action)
        rew_list.append(reward)
        next_obs_list.append(next_obs)
        done_list.append(done)

        # 6) Advance
        obs = next_obs
        idx += 1

        # 7) If done, reset kill state and optionally wait for manual respawn
        if done:
            prev_kill = None
            time.sleep(1.0)

        # 8) Maintain FPS
        t1 = time.time()
        dt = t1 - t0
        if dt < INTERVAL:
            time.sleep(INTERVAL - dt)
        t0 = time.time()

    # ——— Save transitions to NPZ ———
    np.savez_compressed(
        OUT_NPZ,
        observations  = np.stack(obs_list),
        actions       = np.stack(act_list),
        rewards       = np.array(rew_list, np.float32),
        next_obs      = np.stack(next_obs_list),
        dones         = np.array(done_list, np.bool_)
    )
    print(f"Saved {len(obs_list)} transitions to {OUT_NPZ}")

if __name__ == "__main__":
    # start keyboard listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("Recording demo_data – press ESC to stop.")
    record()

    listener.stop()
