import itertools
from typing import Any

import gymnasium
import numpy as np

from assembly_game.processor import Processor


class MinGame(gymnasium.Env):
    MIN = 1

    def __init__(self, **kwargs):
        size = kwargs.get("size", 2)
        if size < 2 or size > 4:
            raise ValueError("Size must be between 2 and 4")

        self.action_space = gymnasium.spaces.Discrete(Processor.get_num_actions())

        values = [self.MIN + i for i in range(size)]
        self.permutations = list(itertools.permutations(values))
        self.observation_space = gymnasium.spaces.Box(
            low=0,  # 0 is a valid value for the registers
            high=self.MIN + size,
            shape=(Processor.get_state_size() * len(self.permutations),),
            dtype=np.int8,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        # state are two possible arrangements of the registers
        # rdi and rsi are the two numbers to compare
        # rax is the result of the comparison
        self.processors = [Processor(*perm) for perm in self.permutations]
        self.previous_correct_items = 0
        state = []
        for proc in self.processors:
            state.extend(proc.get_state())

        return np.array(state), {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        for proc in self.processors:
            halted = proc.evaluate_action(action)
        state = []
        correct_items = 0
        for proc in self.processors:
            state.extend(proc.get_state())
            if proc.rax == self.MIN:
                correct_items += 1
        correctness_reward_weight = 1
        reward = correctness_reward_weight * (
            correct_items - self.previous_correct_items
        )
        self.previous_correct_items = correct_items
        all_correct_reward = 10
        total_env_size = len(self.processors)
        if correct_items == total_env_size:
            reward += all_correct_reward

        log = {f"example_{i}": str(proc) for i, proc in enumerate(self.processors)}

        return np.array(state), reward, halted, False, log
