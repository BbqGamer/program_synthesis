from enum import Enum


class Instruction(Enum):
    MOV = 0
    CMP = 1
    CMOVG = 2


class Operand(Enum):
    RDI = 0
    RSI = 1
    RDX = 2
    RCX = 3
    RAX = 4


STATE_LEN = 6
PROCESSOR_ACTIONS = [
    (inst, op1, op2) for inst in Instruction for op1 in Operand for op2 in Operand
]


def actions_to_asm(actions: list[int]) -> str:
    result = ""
    for i in range(len(actions)):
        action = actions[i]
        inst, op1, op2 = PROCESSOR_ACTIONS[action]
        result += f"{inst.name.lower()}\t%{op1.name.lower()}, %{op2.name.lower()}\n"
    result += "ret\n"
    return result


class Processor:
    # http://6.s081.scripts.mit.edu/sp18/x86-64-architecture-guide.html
    rdi: int  # first argument
    rsi: int  # second argument
    rdx: int  # third argument
    rcx: int  # fourth argument
    rax: int
    cmp_res: int

    def __init__(self, rdi: int = 0, rsi: int = 0, rdx: int = 0, rcx: int = 0) -> None:
        self.rdi = rdi
        self.rsi = rsi
        self.rdx = rdx
        self.rcx = rcx
        self.rax = 0
        self.cmp_res = 0

    def _set_reg(self, reg: Operand, value: int) -> None:
        if reg == Operand.RDI:
            self.rdi = value
        elif reg == Operand.RSI:
            self.rsi = value
        elif reg == Operand.RDX:
            self.rdx = value
        elif reg == Operand.RCX:
            self.rcx = value
        elif reg == Operand.RAX:
            self.rax = value
        else:
            raise ValueError("Invalid register")

    def _get_reg(self, reg: Operand) -> int:
        if reg == Operand.RDI:
            return self.rdi
        elif reg == Operand.RSI:
            return self.rsi
        elif reg == Operand.RDX:
            return self.rdx
        elif reg == Operand.RCX:
            return self.rcx
        elif reg == Operand.RAX:
            return self.rax
        else:
            raise ValueError("Invalid register")

    @staticmethod
    def get_num_actions() -> int:
        return len(PROCESSOR_ACTIONS)

    @staticmethod
    def get_state_size() -> int:
        return STATE_LEN

    def evaluate_action(self, action_id: int) -> bool:
        action = PROCESSOR_ACTIONS[action_id]
        if len(action) != 3:
            raise ValueError("Invalid action")
        inst, op1, op2 = action
        if inst == Instruction.MOV:
            self._set_reg(op2, self._get_reg(op1))
        elif inst == Instruction.CMP:
            self.cmp_res = bool(self._get_reg(op2) > self._get_reg(op1))
        elif inst == Instruction.CMOVG:
            if self.cmp_res:
                self._set_reg(op2, self._get_reg(op1))
        else:
            raise ValueError("Invalid instruction")
        return False

    def get_state(self) -> tuple[int, int, int, int, int, int]:
        state = (self.rdi, self.rsi, self.rax, self.rdx, self.rcx, self.cmp_res)
        assert len(state) == STATE_LEN
        return state

    def __str__(self):
        return f"rdi={self.rdi} rsi={self.rsi} rax={self.rax} rdx={self.rdx} rcx={self.rcx} cmp_res={self.cmp_res}"
