from enum import Enum


class Instruction(Enum):
    MOV = 0
    CMP = 1
    CMOVG = 2
    RET = 3


class Operand(Enum):
    RDI = 0
    RSI = 1
    RAX = 2


NUM_REGISTERS = 4
PROCESSOR_ACTIONS = [
    (inst, op1, op2)
    for inst in [Instruction.MOV, Instruction.CMP, Instruction.CMOVG]
    for op1 in Operand
    for op2 in Operand
] + [(Instruction.RET,)]


class Processor:
    rdi: int
    rsi: int
    rax: int
    cmp_res: int

    def __init__(self, rdi: int = 0, rsi: int = 0) -> None:
        self.rdi = rdi
        self.rsi = rsi
        self.rax = 0
        self.cmp_res = 0

    def _set_reg(self, reg: Operand, value: int) -> None:
        if reg == Operand.RDI:
            self.rdi = value
        elif reg == Operand.RSI:
            self.rsi = value
        elif reg == Operand.RAX:
            self.rax = value
        else:
            raise ValueError("Invalid register")

    def _get_reg(self, reg: Operand) -> int:
        if reg == Operand.RDI:
            return self.rdi
        elif reg == Operand.RSI:
            return self.rsi
        elif reg == Operand.RAX:
            return self.rax
        else:
            raise ValueError("Invalid register")

    @staticmethod
    def get_num_actions() -> int:
        return len(PROCESSOR_ACTIONS)

    @staticmethod
    def get_state_size() -> int:
        return NUM_REGISTERS

    def evaluate_action(self, action_id: int) -> bool:
        action = PROCESSOR_ACTIONS[action_id]
        if len(action) == 1:
            if action[0] != Instruction.RET:
                raise ValueError("Unsupported instruction")
            # doesn't matter what is in op1 and op2, RET just exits
            return True
        elif len(action) == 3:
            inst, op1, op2 = action
            if inst == Instruction.MOV:
                self._set_reg(op2, self._get_reg(op1))
            elif inst == Instruction.CMP:
                self.cmp_res = self._get_reg(op2) - self._get_reg(op1)
            elif inst == Instruction.CMOVG:
                if self.cmp_res > 0:
                    self._set_reg(op2, self._get_reg(op1))
            else:
                raise ValueError("Invalid instruction")
        else:
            raise ValueError("Invalid number of operands")
        return False

    def get_state(self) -> tuple[int, int, int, int]:
        state = (self.rdi, self.rsi, self.rax, self.cmp_res)
        assert len(state) == NUM_REGISTERS
        return state
