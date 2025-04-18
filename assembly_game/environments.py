from typing import Any

import gymnasium
import numpy as np

from assembly_game.processor import Processor

MIN = 1
MAX = 2
TIMEOUT = 20


class Min2Game(gymnasium.Env):
    action_space = gymnasium.spaces.Discrete(Processor.get_num_actions())
    observation_space = gymnasium.spaces.Box(
        low=MIN - MAX,
        high=MAX,
        shape=(Processor.get_state_size() * 2,),
        dtype=np.int8,
    )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        # state are two possible arrangements of the registers
        # rdi and rsi are the two numbers to compare
        # rax is the result of the comparison
        self.processors = [Processor(rdi=MIN, rsi=MAX), Processor(rdi=MAX, rsi=MIN)]
        self.t = 0

        state = []
        for proc in self.processors:
            state.extend(proc.get_state())

        return np.array(state), {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        for proc in self.processors:
            halted = proc.evaluate_action(action)

        self.t += 1
        num_correct = 0
        state = []
        for proc in self.processors:
            state.extend(proc.get_state())
            if proc.rax == 1:
                num_correct += 1

        reward = 10 * num_correct - self.t

        return np.array(state), reward, halted, False, {}
