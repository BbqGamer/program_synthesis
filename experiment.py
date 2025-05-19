import argparse

import gymnasium as gym
import stable_baselines3 as st3

import wandb
from assembly_game.processor import PROCESSOR_ACTIONS
from wandb.integration.sb3 import WandbCallback

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup an experiment")
    parser.add_argument(
        "-e",
        "--environment",
        type=str,
        help="Choose an environment where you wish to perform your experiment [min_game;TODO]",
    )
    parser.add_argument("-n", type=int, help="Size of environment")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Model name. Choose one of the following [ppo;TODO]",
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=30,
        help="Maximum number of steps our program can make before force termination",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=0.0003,
        help="Learning rate of our model",
    )
    parser.add_argument(
        "-b", "--batch-size", default=64, type=int, help="Batch size of our model"
    )
    parser.add_argument(
        "-t",
        "--timesteps",
        type=int,
        default=10000,
        help="Timesteps taken by our model",
    )
    args = parser.parse_args()

    if not (args.model or args.environment or args.n):
        parser.error("No action requested, add -e ENVIRONMENT, -n N and -m MODEL")
    if args.learning_rate > 1 or args.learning_rate < 0:
        parser.error("Learning rate should be between 0 and 1")

    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="kostekkor-pozna-university-of-technology",
        # Set the wandb project where this run will be logged.
        project="ProgSynth",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
        },
        sync_tensorboard=True,
    )
    environments = {
        "min_game": "MinGame",
    }

    models = {"ppo": st3.PPO}

    env = gym.make(
        environments[args.environment],
        max_episode_steps=args.steps,
        size=args.n,
    )
    env
    rl_model = models[args.model](
        policy="MlpPolicy",
        env=env,
        verbose=1,
        device="cpu",
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        tensorboard_log="./ppo_tensorboard/"
    )

    rl_model.learn(
        total_timesteps=args.timesteps,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_freq=1000,
            model_save_path=f"models/{run.id}",
            verbose=2,
            log='all',
        ),
    )
    wandb.finish()
    state, _ = env.reset()
    cumreward = 0
    run.finish()
    for i in range(args.steps):
        action, _ = rl_model.predict(state)
        state, reward, terminated, truncated, info = env.step(action)
        cumreward += reward
        print(PROCESSOR_ACTIONS[action], info, reward)
        if terminated or truncated:
            print("Terminated?", terminated)
            print("Truncated?:", truncated)
            print(f"Episode finished after {i + 1} timestamps")
            break
    print(f"total reward {cumreward}")
