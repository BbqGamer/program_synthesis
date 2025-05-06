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


class Min4Game(gymnasium.Env):
    MIN = 1
    MID_LOW = 2
    MID_HIGH = 3
    MAX = 4
    action_space = gymnasium.spaces.Discrete(Processor.get_num_actions())
    observation_space = gymnasium.spaces.Box(
        low=MIN - MAX,
        high=MAX,
        shape=(Processor.get_state_size() * 4 * 3 * 2,),
        dtype=np.int8,
    )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        # state are two possible arrangements of the registers
        # rdi and rsi are the two numbers to compare
        # rax is the result of the comparison
        all_possible = itertools.permutations(
            [self.MIN, self.MID_LOW, self.MID_HIGH, self.MAX]
        )

        self.processors = []
        for i in all_possible:
            p = Processor(rdi=i[0], rsi=i[1], rcx=i[2], rdx=i[3])
            self.processors.append(p)
        self.t = 0
        self.previous_correct_items = 0
        state = []
        for proc in self.processors:
            state.extend(proc.get_state())

        return np.array(state), {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        for proc in self.processors:
            halted = proc.evaluate_action(action)
        self.t += 1
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


import math


class MinNGame(gymnasium.Env):
    def __init__(self, n):
        self.n = n
        self.vals = [_ for _ in range(n)]
        self.MIN = 1
        self.MAX = self.n
        self.did_achieve_goal = False
        self.action_space = gymnasium.spaces.Discrete(Processor.get_num_actions())
        self.observation_space = gymnasium.spaces.Box(
            low=self.MIN - self.MAX,
            high=self.MAX,
            shape=(Processor.get_state_size() * math.factorial(n),),
            dtype=np.int8,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        # state are two possible arrangements of the registers
        # rdi and rsi are the two numbers to compare
        # rax is the result of the comparison
        all_possible = itertools.permutations(self.vals)

        self.processors = []
        for permutation in all_possible:
            p = Processor(*permutation)
            self.processors.append(p)
        self.t = 0
        self.previous_correct_items = 0
        state = []
        for proc in self.processors:
            state.extend(proc.get_state())

        return np.array(state), {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        for proc in self.processors:
            halted = proc.evaluate_action(action)
        self.t += 1
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
            self.did_achieve_goal = True
            reward += all_correct_reward

        log = {f"example_{i}": str(proc) for i, proc in enumerate(self.processors)}
        log['is_success'] = self.did_achieve_goal
        return np.array(state), reward, halted, False, log
