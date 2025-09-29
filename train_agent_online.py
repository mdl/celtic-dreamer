import torch
import argparse
import os
from dreamer import Dreamer
from utils import loadConfig, seedEverything, plotMetrics, saveLossesToCSV, ensureParentFolders
from tqdm import tqdm
from celtic_env_wrapper import CelticHeroesEnv
from envs import GymPixelsProcessingWrapper, CleanGymWrapper

device = torch.device("mps")

def main(configFile):
    config = loadConfig(configFile)
    seedEverything(config.seed)

    runName = f"{config.environmentName}_{config.runName}_ONLINE"  # Suffix for online training

    # --- IMPORTANT ---
    # Specify the pre-trained world model checkpoint to load
    # This should match the output of the pretrain_world_model.py script
    checkpointToLoad = os.path.join(config.folderNames.checkpointsFolder,
                                    f"{config.environmentName}_{config.runName}_WM_final.pth")

    metricsFilename = os.path.join(config.folderNames.metricsFolder, runName)
    plotFilename = os.path.join(config.folderNames.plotsFolder, runName)
    checkpointFilenameBase = os.path.join(config.folderNames.checkpointsFolder, runName)
    videoFilenameBase = os.path.join(config.folderNames.videosFolder, runName)
    ensureParentFolders(metricsFilename, plotFilename, checkpointFilenameBase, videoFilenameBase)

    env = CleanGymWrapper(GymPixelsProcessingWrapper(CelticHeroesEnv()))
    observationShape, actionSize = env.observation_space.shape, env.action_space.n
    print(f"envProperties: obs {observationShape}, action size {actionSize}")

    dreamer = Dreamer(observationShape, actionSize, device, config.dreamer)

    # if hasattr(config.dreamer.buffer, 'loadFrom') and config.dreamer.buffer.loadFrom:
    #     print(f"Loading dataset from: {config.dreamer.buffer.loadFrom}")
    #     dreamer.buffer.load(config.dreamer.buffer.loadFrom)

    # Load the pre-trained world model checkpoint
    print(f"Loading pre-trained world model from: {checkpointToLoad}")
    dreamer.loadCheckpoint(checkpointToLoad)
    # Reset the gradient step counter after loading
    dreamer.totalGradientSteps = 0

    # Start the online training loop
    print("Starting online training...")
    iterationsNum = config.gradientSteps // config.replayRatio
    for _ in tqdm(range(iterationsNum)):
        for _ in range(config.replayRatio):
            # Sample from the buffer (which is initially empty and will be filled by the agent)
            # You might need to aedd a check here to ensure the buffer has enough data to sample
            if len(dreamer.buffer) > dreamer.config.batchSize * dreamer.config.batchLength:
                sampledData = dreamer.buffer.sample(dreamer.config.batchSize, dreamer.config.batchLength)
                initialStates, worldModelMetrics = dreamer.worldModelTraining(sampledData)
                behaviorMetrics = dreamer.behaviorTraining(initialStates)
                dreamer.totalGradientSteps += 1
            else:
                # If buffer is not ready, just collect experience
                break

            if dreamer.totalGradientSteps % config.checkpointInterval == 0 and config.saveCheckpoints:
                suffix = f"{dreamer.totalGradientSteps / 1000:.0f}k"
                dreamer.saveCheckpoint(f"{checkpointFilenameBase}_{suffix}")
                dreamer.buffer.save(f"{config.folderNames.bufferFolder}/{runName}_{suffix}.npz")
                evaluationScore = dreamer.environmentInteraction(env, config.numEvaluationEpisodes,
                                                                 seed=config.seed, evaluation=True, saveVideo=True,
                                                                 filename=f"{videoFilenameBase}_{suffix}")
                print(
                    f"Saved Checkpoint and Video at {suffix:>6} gradient steps. Evaluation score: {evaluationScore:>8.2f}")

        # Interact with the environment to collect new, on-policy data
        mostRecentScore = dreamer.environmentInteraction(env, config.numInteractionEpisodes, seed=config.seed)

        if config.saveMetrics and 'worldModelMetrics' in locals():
            metricsBase = {"envSteps": dreamer.totalEnvSteps, "gradientSteps": dreamer.totalGradientSteps,
                           "totalReward": mostRecentScore}
            saveLossesToCSV(metricsFilename, metricsBase | worldModelMetrics | behaviorMetrics)
            plotMetrics(f"{metricsFilename}", savePath=f"{plotFilename}",
                        title=f"{config.environmentName} - Online Training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="celtic-heroes.yml")
    main(parser.parse_args().config)
