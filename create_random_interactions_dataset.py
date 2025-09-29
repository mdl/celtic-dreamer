import torch
import argparse
import os
from tqdm import tqdm
import torch.nn.functional as F

from utils import loadConfig, seedEverything
from envs import GymPixelsProcessingWrapper, CleanGymWrapper
from celtic_env_wrapper import CelticHeroesEnv
from buffer import ReplayBuffer

device = torch.device("mps")

def create_dataset(config_file, num_episodes, output_file):
    """
    Runs a random agent in the environment to collect a dataset
    and saves it using the buffer's save method.
    """
    config = loadConfig(config_file)
    seedEverything(config.seed)

    print("Initializing environment...")
    # Use the same environment wrappers as your main script
    env = CleanGymWrapper(GymPixelsProcessingWrapper(CelticHeroesEnv()))
    observation_shape, action_size = env.observation_space.shape, env.action_space.n

    print(f"Observation shape: {observation_shape}")
    print(f"Action size: {action_size}")

    # Increase the buffer capacity for data collection if needed,
    # otherwise use the one from the config file.
    # A large capacity ensures you don't overwrite data during collection.
    if config.dreamer.buffer.capacity < 50000:
        print("Temporarily increasing buffer capacity for dataset creation.")
        config.dreamer.buffer.capacity = 50000

    # Initialize your ReplayBuffer
    replay_buffer = ReplayBuffer(observation_shape, action_size, config.dreamer.buffer, device)

    print(f"Collecting data for {num_episodes} episodes...")
    total_steps = 0
    for i in range(num_episodes):
        print(f"--- Starting Episode {i + 1}/{num_episodes} ---")
        obs = env.reset()
        done = False

        pbar = tqdm(desc=f"Episode {i + 1}")
        while not done:
            # Choose a random integer action from the environment's action space
            action_int = env.action_space.sample()

            next_obs, reward, terminated = env.step(action_int)
            done = terminated

            # One-hot encode the integer action to match the buffer's expected format
            action_one_hot = F.one_hot(torch.tensor(action_int), num_classes=action_size).numpy()

            # Add the transition to the buffer using its 'add' method signature
            replay_buffer.add(obs, action_one_hot, reward, done)

            obs = next_obs
            total_steps += 1
            pbar.update(1)
        pbar.close()
        print(f"Episode finished. Total steps so far: {total_steps}")

    print("\nData collection complete.")
    print(f"Total transitions collected: {len(replay_buffer)}")

    # Use the new save method from your ReplayBuffer class
    replay_buffer.save(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="celtic-heroes.yml", help="Path to the config file.")
    parser.add_argument("--episodes", type=int, default=80, help="Number of episodes to collect.")
    parser.add_argument("--out", type=str, default="celtic_heroes_dataset.npz", help="Path to save the dataset file.")
    args = parser.parse_args()

    # Ensure the output directory exists
    output_dir = os.path.dirname(args.out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    create_dataset(args.config, args.episodes, args.out)