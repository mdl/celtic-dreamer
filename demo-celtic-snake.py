#!/usr/bin/env python3
"""
Demo script for testing the CelticHeroesEnv gym wrapper using MultiBinary(3) actions.
Moves in a simple snake pattern: forward, left turn, forward, right turn, repeating.
Captures and saves observations to ./demo_ss for visual inspection.
"""
import os
import time
import numpy as np
from PIL import Image
from celtic_env_wrapper import CelticHeroesEnv

def main():
    # Prepare output directory for screenshots
    output_dir = os.path.join(os.path.dirname(__file__), 'demo_ss')
    os.makedirs(output_dir, exist_ok=True)

    # Initialize environment (no detectors needed for this demo)
    env = CelticHeroesEnv()
    obs = env.reset()

    # Define snake pattern as multi-hot vectors: [forward, turn_left, turn_right]
    pattern = [
        # np.array([1, 0, 0], dtype=np.int8),  # forward
        # np.array([1, 1, 0], dtype=np.int8),  # turn left
        # np.array([1, 0, 0], dtype=np.int8),  # forward
        # np.array([1, 0, 1], dtype=np.int8),  # turn right
        np.array(0, dtype=np.int8),  # forward
    ]
    total_steps = 200

    print("Starting snake demo with MultiBinary actions and saving screenshots...")
    for step in range(total_steps):
        action = pattern[step % len(pattern)]
        obs, reward, done, _, kill = env.step(action)
        print(f"Step {step+1}/{total_steps}: action={action.tolist()}, reward={reward}, done={done}")

        # Save the observation image
        img = Image.fromarray(obs)
        killimg = Image.fromarray(kill)
        timestamp = int(time.time() * 1000)
        filename = f"obs_step_{step+1}_{timestamp}.png"
        killfilename = f"kill_step_{step + 1}_{timestamp}.png"
        img.save(os.path.join(output_dir, filename))
        killimg.save(os.path.join(output_dir, killfilename))

        if done:
            print("Episode ended, resetting environment")
            obs = env.reset()

        # Pause to observe movement
        time.sleep(env.interval)

    env.close()
    print(f"Demo finished. Screenshots saved in {output_dir}")


if __name__ == "__main__":
    main()
