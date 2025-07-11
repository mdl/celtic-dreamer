import gymnasium as gym
import numpy as np
import cv2

class GymPixelsProcessingWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        observationSpace = self.observation_space
        newObsShape = observationSpace.shape[-1:] + observationSpace.shape[:2]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=newObsShape, dtype=np.float32)

    def observation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))/255.0
        return observation

class CleanGymWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._latest_processed_observation_image = None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self._latest_processed_observation_image = obs
        return obs, reward, done

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self._latest_processed_observation_image = obs
        return obs

    def render(self):
        if self._latest_processed_observation_image is not None:
            source_image = np.copy(self._latest_processed_observation_image)
            processed_frame = source_image
            # 1. Transpose from (C, H, W) to (H, W, C)
            if processed_frame.ndim == 3 and processed_frame.shape[0] in [1, 3, 4]:
                processed_frame = processed_frame.transpose(1, 2, 0)

            # 2. Scale and convert to uint8 [0, 255] using OpenCV
            if np.issubdtype(processed_frame.dtype, np.floating):
                if not (processed_frame.min() >= -0.01 and processed_frame.max() <= 1.01):
                    print(f"Warning (from render method): Frame data is float but not in the expected [0,1] range (min: {processed_frame.min():.2f}, max: {processed_frame.max():.2f}). It will be clipped to [0,1] before scaling.")
                processed_frame_clipped = np.clip(processed_frame, 0.0, 1.0)
                video_ready_frame = cv2.convertScaleAbs(processed_frame_clipped, alpha=255.0)

            elif processed_frame.dtype == np.uint8:
                video_ready_frame = processed_frame
            else:
                print(f"Warning (from render method): Frame data is {processed_frame.dtype} (not float or uint8). Attempting to clip to [0,255] and convert to uint8. Please verify image source and range.")
                video_ready_frame = np.clip(processed_frame, 0, 255).astype(np.uint8)

            return video_ready_frame
        else:
            print("Warning: No observation image cached yet. Call reset() first.")
            return None
