# Example to illustrate how to link assembly with C
min.s contains minimal implementation of min function in
GNU assembly (GAS). It assumes that certain registers will
be populated with argument values.

It is assumed that the return value will be put into $rax register.

Compile and run the program with the following commands
```bash
gcc main.c min.s -o min
./min
```

## Disassembly
The easiest way to inspect the machine code of a binary is to use
the program `objdump`, here are some examples:
```bash
objdump --disassemble=min min
objdump --disassemble=main min
```
