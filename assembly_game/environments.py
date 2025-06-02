import itertools
import random
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
        self.all_correct = False
        self.t = 0
        state = []
        for proc in self.processors:
            state.extend(proc.get_state())

        return np.array(state), {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self.t += 1
        for proc in self.processors:
            halted = proc.evaluate_action(action)
        state = []
        correct_items = 0
        for proc in self.processors:
            state.extend(proc.get_state())
            if proc.rax == self.MIN:
                correct_items += 1
        correctness_reward_weight = 5
        reward = correctness_reward_weight * (
            correct_items - self.previous_correct_items
        )
        self.previous_correct_items = correct_items
        all_correct_reward = 100
        total_env_size = len(self.processors)
        if correct_items == total_env_size:
            reward += all_correct_reward - self.t
            self.all_correct = True
            halted = True

        log = {f"example_{i}": str(proc) for i, proc in enumerate(self.processors)}
        log["is_success"] = self.all_correct
        return np.array(state), reward, halted, False, log


class SortGame(gymnasium.Env):
    MIN = 1

    def __init__(self, **kwargs):
        size = kwargs.get("size", 2)
        self.size = size
        if size < 2 or size > 4:
            raise ValueError("Size must be between 2 and 4")

        self.action_space = gymnasium.spaces.Discrete(Processor.get_num_actions())

        self.values = [self.MIN + i for i in range(size)]
        self.permutations = list(itertools.permutations(self.values))
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
        random.shuffle(self.processors)
        self.previous_correct_items = 0
        self.all_correct = False
        self.t = 0
        state = []
        for proc in self.processors:
            state.extend(proc.get_state())

        return np.array(state), {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self.t += 1
        for proc in self.processors:
            halted = proc.evaluate_action(action)
        state = []

        correct_items = 0
        correct_testcases = 1
        for proc in self.processors:
            state.extend(proc.get_state())
            sequence = [proc.rdi, proc.rsi, proc.rdx, proc.rcx]
            for i in range(self.size):
                if sequence[i] == self.values[i]:
                    correct_items += 1
            if correct_items == self.size:
                correct_testcases += 1

        correctness_reward_weight = 5
        reward = correctness_reward_weight * (
            correct_items - self.previous_correct_items
        )
        self.previous_correct_items = correct_items
        all_correct_reward = 100 * self.size
        if correct_testcases == len(self.processors):
            reward += all_correct_reward - self.t
            self.all_correct = True
            halted = True

        log = {}
        log["correct_items"] = correct_items
        log["correct_testcases"] = correct_testcases
        for i, proc in enumerate(self.processors):
            sequence = [proc.rdi, proc.rsi, proc.rdx, proc.rcx]
            log[f"example_{i}"] = str(sequence)

        log["is_success"] = self.all_correct
        return np.array(state), reward, halted, False, log
