import torch
import argparse
import os
from dreamer import Dreamer
from utils import loadConfig, seedEverything, plotMetrics, saveLossesToCSV, ensureParentFolders
from tqdm import tqdm

device = torch.device("mps")

def main(configFile):
    config = loadConfig(configFile)

    seedEverything(config.seed)

    runName = f"{config.environmentName}_{config.runName}_WM"  # Suffix to denote World Model pre-training
    metricsFilename = os.path.join(config.folderNames.metricsFolder, runName)
    plotFilename = os.path.join(config.folderNames.plotsFolder, runName)
    checkpointFilenameBase = os.path.join(config.folderNames.checkpointsFolder, runName)
    final_checkpoint_path = f"{checkpointFilenameBase}_final"

    ensureParentFolders(metricsFilename, plotFilename, checkpointFilenameBase)

    # We don't need the live environment for pre-training, only its properties.
    # This is a placeholder to get observation and action sizes.
    # In a more advanced setup, you might save these properties with the dataset.
    from envs import GymPixelsProcessingWrapper, CleanGymWrapper
    from celtic_env_wrapper import CelticHeroesEnv
    env = CleanGymWrapper(GymPixelsProcessingWrapper(CelticHeroesEnv()))
    observationShape, actionSize = env.observation_space.shape, env.action_space.n
    env.close()  # We don't need the live env running

    print(f"envProperties: obs {observationShape}, action size {actionSize}")

    dreamer = Dreamer(observationShape, actionSize, device, config.dreamer)
    dreamer.loadCheckpoint(final_checkpoint_path)

    # Load the static dataset created by create_dataset.py
    if hasattr(config.dreamer.buffer, 'loadFrom') and config.dreamer.buffer.loadFrom:
        print(f"Loading dataset from: {config.dreamer.buffer.loadFrom}")
        dreamer.buffer.load(config.dreamer.buffer.loadFrom)
    else:
        raise ValueError("Config file must specify a dataset to load via 'buffer.loadFrom'")

    # Pre-train for a set number of steps
    print(f"Pre-training World Model for {config.gradientStepsWorldPretrain} steps...")
    for i in tqdm(range(config.gradientStepsWorldPretrain)):
        sampledData = dreamer.buffer.sample(dreamer.config.batchSize, dreamer.config.batchLength)

        # --- ONLY TRAIN THE WORLD MODEL ---
        _, worldModelMetrics = dreamer.worldModelTraining(sampledData)

        # We don't call behaviorTraining() or environmentInteraction()
        dreamer.totalGradientSteps += 1

        # Periodically save metrics
        if i % 100 == 0 and config.saveMetrics:
            metricsBase = {"gradientSteps": dreamer.totalGradientSteps}
            saveLossesToCSV(metricsFilename, metricsBase | worldModelMetrics)

    dreamer.saveCheckpoint(final_checkpoint_path)
    print(f"\nWorld Model pre-training complete. Final checkpoint saved to: {final_checkpoint_path}.pth")

    # Plot the pre-training loss
    if config.saveMetrics:
        plotMetrics(f"{metricsFilename}", savePath=f"{plotFilename}",
                    title=f"{config.environmentName} - World Model Pre-training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="celtic-heroes.yml")
    main(parser.parse_args().config)
