# cppLox

This repo is a cpp Lox interpreter of Lox Language from [Crafting Interpreters](https://craftinginterpreters.com/)

Hope this will be a useful one I would use in future.

There are (will be) three versions of Lox interpreter, they all share a same frontend, and has their own backend.
1. Frontend contains:
    - [x] A scanner to build tokens from raw code.
    - [x] A Parser to build AST from tokens.
    - [x] An AST definition with a simple pass-manager.

2. Backend contains:
    - [x] A naive tree-walker interpreter, it runs directly on the AST from frontend.
    - [ ] WIP: A virtual machine interpreter, this one uses only scanner from frontend, and do a one pass  compilation, which convert tokens to bytecode directly.
    - [ ] A MLIR based JIT interpreter, it will first run a pass on AST to generate an MLIR IR of Lox Dialect, and then lower to LLVM dialect, so we can leverage the LLVM JIT utilities to do a JIT run.

# Build

## Requirements
1. A C++20 compiler, gcc >= 10.0 or clang >= 10.0 would be fine, check [cppreference](https://en.cppreference.com/w/cpp/compiler_support/20) to see more
