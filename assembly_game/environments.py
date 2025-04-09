from typing import Any

import gymnasium

from assembly_game.processor import Processor

MIN = 1
MAX = 2
TIMEOUT = 20


class Min2Game(gymnasium.Env):
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

        return state, {}

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

        return state, reward, halted, False, {}
