from enum import Enum
from typing import Any

import gymnasium


class Instruction(Enum):
    MOV = 0
    CMP = 1
    CMOVG = 2
    RET = 3


class Operand(Enum):
    RDI = 0
    RSI = 1
    RAX = 2


ProcState = tuple[
    int,  # rdi
    int,  # rsi
    int,  # rax
    int,  # comparison result
]


def evaluate_instruction(
    state: ProcState, inst: Instruction, op1: Operand, op2: Operand
) -> ProcState:
    if inst == Instruction.MOV:
        new_state = list(state)
        new_state[op2.value] = new_state[op1.value]
    elif inst == Instruction.CMP:
        new_state = list(state)
        new_state[3] = new_state[op2.value] - new_state[op1.value]
    elif inst == Instruction.CMOVG:
        new_state = list(state)
        if new_state[3] > 0:
            new_state[op2.value] = new_state[op1.value]
    elif inst == Instruction.RET:
        new_state = state
    else:
        raise ValueError("Invalid instruction")
    return ProcState(tuple(new_state))


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
        self.state = ((MIN, MAX, 0, 0), (MAX, MIN, 0, 0))
        self.t = 0
        return self.state, {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        inst, op1, op2 = action
        self.state = [evaluate_instruction(s, inst, op1, op2) for s in self.state]
        self.t += 1

        num_correct = len(
            list(filter(lambda s: s[Operand.RAX.value] == MIN, self.state))
        )

        done = self.t >= TIMEOUT or inst == Instruction.RET

        reward = 10 * num_correct - self.t
        return self.state, reward, done, False, {}
