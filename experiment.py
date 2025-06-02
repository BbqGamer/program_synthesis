import argparse

import gymnasium as gym
import numpy as np
import stable_baselines3 as st3
from stable_baselines3.common.callbacks import BaseCallback

import wandb
from assembly_game.processor import PROCESSOR_ACTIONS, actions_to_asm
from wandb.integration.sb3 import WandbCallback


class BestTrajectoryCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super(BestTrajectoryCallback, self).__init__(verbose)
        self.save_path = save_path
        self.best_reward = -np.inf
        self.current_ep_actions = []
        self.current_ep_rewards = []
        self.results = []

    def _on_step(self) -> bool:
        # Save obs and action at every step
        self.current_ep_actions.append(self.locals["actions"])

        # Info contains "episode" key at the end of an episode
        infos = self.locals["infos"]
        rewards = self.locals["rewards"]
        self.current_ep_rewards.append(rewards)

        for info in infos:
            if "episode" in info:
                self._log_metrics(info)
                ep_reward = sum(self.current_ep_rewards)
                if ep_reward > self.best_reward:
                    self.best_reward = ep_reward
                    self._save_trajectory(info)

                self.current_ep_rewards = []
                self.current_ep_actions = []
                break

        return True

    def _save_trajectory(self, info):
        print(f"New best trajectory found with reward: {self.best_reward}")
        actions = [action[0] for action in self.current_ep_actions]
        self.results.append(
            (
                self.num_timesteps,
                actions,
                self.best_reward,
            )
        )

        wandb.log(
            {
                "correct_items" = info["correct_items"],
                "correct_testcases" = info["correct_testcases"],
                "best_program_code": wandb.Html(
                    f"<pre><code>{actions_to_asm(actions)}</code></pre>"
                ),
            },
            step=self.num_timesteps,
        )

    def _on_training_end(self):
        # Save the best trajectory to a file
        with open(self.save_path, "w") as f:
            for timestep, actions, reward in self.results:
                f.write(
                    f"Timestep: {timestep}, Len: {len(actions)}, Reward: {reward}\n"
                )
                f.write(actions_to_asm(actions))
                f.write("\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup an experiment")
    parser.add_argument(
        "-e",
        "--environment",
        type=str,
        help="Choose an environment where you wish to perform your experiment [MinGame;SortGame]",
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
        default=50000,
        help="Timesteps taken by our model",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="results.txt",
        help="Path to save the results of the experiment",
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.01, help="Entropy coefficient (for PPO)"
    )
    parser.add_argument(
        "--run_name", default=None, type=str, help="Name of the run appearing in wandb"
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
            "algorithm": args.model,
            "environment": args.environment,
            "environment-size": args.n,
            "ent-coef": args.ent_coef,
        },
        sync_tensorboard=True,
        name=args.run_name,
    )
    models = {"PPO": st3.PPO}

    env = gym.make(
        args.environment,
        max_episode_steps=args.steps,
        size=args.n,
    )
    rl_model = models[args.model](
        policy="MlpPolicy",
        env=env,
        verbose=1,
        device="cpu",
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        tensorboard_log="./ppo_tensorboard/",
        ent_coef=args.ent_coef,
    )

    trajcetory_callback = BestTrajectoryCallback(verbose=1, save_path="results.txt")
    rl_model.learn(
        total_timesteps=args.timesteps,
        callback=[
            WandbCallback(
                gradient_save_freq=100,
                model_save_freq=1000,
                model_save_path=f"models/{run.id}",
                verbose=2,
                log="all",
            ),
            trajcetory_callback,
        ],
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
