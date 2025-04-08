.global min
.type   min, @function

min:
    # On Linux x86-64 (System V ABI):
    # - %rdi - The first argument
    # - %rsi - The second argument
    # - %rax - The return value
    mov %rdi, %rax
    cmp %rsi, %rax
    cmovg %rsi, %rax
    ret

    # Mark that no executable stack is needed
    .section .note.GNU-stack,"",@progbits
