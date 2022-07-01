Generally speaking, lox will be a dialect of C language when using JIT, that is , any
feature that is not supported by C natively will not be supported by lox neither.

Someday in the future, lox might be a dialect of C++ language, but for now, I just want to eat some low hanging fruit.

We will have two JIT backend, one is LLVM based, another one is MLIR based, I decided to do this just want to
figure out how much power the Multi-Level IR can give me, comparing to the classical single level LLVM IR.

Here is a list of features notes

* LLVM based JIT:
1. Only C style struct supported, the init() method will be used to define members, and only exprs like `self.var:int = 1;` are allowed.
2. Nested function (closure) not supported.


MLIR based JIT: