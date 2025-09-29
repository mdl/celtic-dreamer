import attridict
import numpy as np
import torch

# Code comes from SimpleDreamer repo, I only changed some formatting and names, but I should really remake it.
class ReplayBuffer(object):
    def __init__(self, observation_shape, actions_size, config, device):
        self.config = config
        self.device = device
        self.capacity = int(self.config.capacity)

        self.observations        = np.empty((self.capacity, *observation_shape), dtype=np.float32)
        self.actions             = np.empty((self.capacity, actions_size), dtype=np.float32)
        self.rewards             = np.empty((self.capacity, 1), dtype=np.float32)
        self.dones               = np.empty((self.capacity, 1), dtype=np.float32)

        self.bufferIndex = 0
        self.full = False
        
    def __len__(self):
        return self.capacity if self.full else self.bufferIndex

    def add(self, observation, action, reward, done):
        self.observations[self.bufferIndex]     = observation
        self.actions[self.bufferIndex]          = action
        self.rewards[self.bufferIndex]          = reward
        self.dones[self.bufferIndex]            = done

        self.bufferIndex = (self.bufferIndex + 1) % self.capacity
        self.full = self.full or self.bufferIndex == 0

    def sample(self, batchSize, sequenceSize):
        lastFilledIndex = self.bufferIndex - sequenceSize + 1
        # A small correction to ensure we don't sample from an empty buffer
        if not self.full and lastFilledIndex <= 0:
             raise ValueError("Not enough data in the buffer to sample a full sequence.")

        sampleIndex = np.random.randint(0, self.capacity if self.full else lastFilledIndex, batchSize).reshape(-1, 1)
        sequenceLength = np.arange(sequenceSize).reshape(1, -1)

        sampleIndex = (sampleIndex + sequenceLength) % self.capacity

        observations = torch.as_tensor(self.observations[sampleIndex], device=self.device).float()
        actions  = torch.as_tensor(self.actions[sampleIndex], device=self.device)
        rewards  = torch.as_tensor(self.rewards[sampleIndex], device=self.device)
        dones    = torch.as_tensor(self.dones[sampleIndex], device=self.device)

        sample = attridict({
            "observations"      : observations,
            "actions"           : actions,
            "rewards"           : rewards,
            "dones"             : dones})
        return sample

    def save(self, filepath):
        """Saves the replay buffer data to a compressed npz file."""
        print(f"Saving buffer data to {filepath}...")
        with open(filepath, 'wb') as f:
            np.savez_compressed(f,
                                observations=self.observations,
                                actions=self.actions,
                                rewards=self.rewards,
                                dones=self.dones,
                                bufferIndex=np.array(self.bufferIndex, dtype=np.int32),
                                full=np.array(self.full, dtype=np.bool_)
                                )
        print("Buffer saved successfully!")

    def load(self, filepath):
        """Loads the replay buffer data from a file."""
        try:
            print(f"Loading buffer data from {filepath}...")
            with np.load(filepath) as data:
                # Check if loaded data exceeds current capacity
                num_to_load = len(data['observations'])
                if num_to_load > self.capacity:
                    raise ValueError(
                        f"Dataset size ({num_to_load}) > buffer capacity ({self.capacity}). Increase buffer capacity in your config.")

                observations = data['observations']
                self.observations[:num_to_load] = observations
                self.actions[:num_to_load] = data['actions']
                self.rewards[:num_to_load] = data['rewards']
                self.dones[:num_to_load] = data['dones']
                self.bufferIndex = data['bufferIndex'].item()
                self.full = data['full'].item()

            print(f"Buffer data loaded. Current size: {len(self)}")
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {filepath}")
            exit()
