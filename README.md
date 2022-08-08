# cppLox

This repo is a cpp Lox interpreter of Lox Language from [Crafting Interpreters](https://craftinginterpreters.com/)

Hope this will be a useful one I would use in future.

There are (will be) four versions of Lox interpreter, they all share a same frontend, and has their own backend.

1. Frontend contains:
   - [x] A scanner to build tokens from raw code.
   - [x] A RD Parser.
   - [x] A Pratt Parser.
   - [x] An AST definition with a simple AST generation tool.

2. Backend contains:
   - [x] A naive tree-walker interpreter, like the jlox, it runs directly on the AST from frontend.
   - [x] A virtual machine interpreter, like the clox, this one uses only scanner from frontend, and do a
     "Pratt Parsing" style one pass compilation, which convert tokens to bytecode directly.
   - [ ] WIP: A LLVM based JIT interpreter, Lox will be a static language with this backend.
     The LLVM backend will translate lox AST to LLVM IR , and then leverage the LLVM Optimization/JIT utilities
     to do a JIT run.
   - [ ] WIP: A MLIR based JIT interpreter, like the LLVM one, but this backend will first translate lox AST to lox
     dialect, then lowering to LLVM dialect using MLIR's multi-level lowering strategy.

# What's the difference ?

1. Pure C++ implementation.
   * No need to learn java things.
   * Because the jlox and clox are both wrote in same language, their implementation shares a lot of code,like GC system,
     lexer, parser, and so on ,which make transition from jlox to clox much easier.
2. A more clear implementation, code is (hopefully) more readable.
   * Especially true when comparing with original clox's c style code.
   * To write some clean code, it is intended to write some code in a not so efficient/well-designed way. e.g. The original
     clox uses a function map to dispatch the codegen-call, which make the implementation more structure, but at here,
     the dispatch is done by a plain switch-case, which is more easy to understand.
3. Files/Modules are well organized, which may help you understand the relationship between each module easier.
4. More language features (Virtual-Machine backend only support `break/continue`):
   * `break/continue` in loops.
   * Comma expression`a,b,c,d`, `[a,b,c,d]` style list expression and `a[i]` style element indexing.
   * Builtin `Tensor`(n-dimension dense matrix) support.
   * Optional python style type hint (used by the jit backend), e.g., `var a:float = 3;`, `fun bool add(x:float , y:float)`
5. Help you to learn LLVM/MLIR in a better toy.

# More about JIT

* The MLIR backends is intended to be a tutorial too, the initial implementation is basically a copy
  of [MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy), check the commit `79e477` to see a how the Toy6 is
  implemented in Lox

* [jit_notes.md](jit_notes.md) may give you more info.

# Requirements

## For Buiding

* A C++20 compiler is all you need. GCC >= 10.0 or clang >= 10.0 sould be fine,
  check [cppreference](https://en.cppreference.com/w/cpp/compiler_support/20) to see more.
* [Optional] Prebuild LLVM, only used for LLVM/MLIR backend.
    * For LLVM is a huge project to download/build, when JIT backend is not enabled, llvm project will not be
      needed.
    * If you want to build with the JIT backend, run the `third_party/prepare_mlir.sh` to download/prebuild llvm.

## For Testing

If you want to run the test cases in native environment, [dart-sdk](https://dart.dev/tools/sdk) is required.
If you had docker installed, there is a all-in-one script to launch test with docker.

# Build

```bash
git clone https://github.com/edimetia3d/cppLox.git --recursive
cd cppLox
mkdir build
cmake ..
make
```

## CMAKE Options

1. `-DUPSTREAM_STYLE_ERROR_MSG=ON`. This impl's parsing(compiling)/err-handling logic is a little different from
   upstream. Enable this flag will make error messages behaves like upstream. It is OFF by default, but for the unit
   tests, it is set to ON.
2. `-DENABLE_JIT_BACKEND=ON`. Enable the JIT backend, OFF by default. Note that you may need to prebuild LLVM
   to use this option (See Requirements part).

# Test

This project leverages test from [Crafting Interpreters GitHub](https://github.com/munificent/craftinginterpreters),
To launch the test, you can use either docker or native machine.

1. If you have docker, just launch the script `./test/launch_unittest_with_docker.sh`
2. If you want to run the test in native machine, you can launch the `./test/unittest.sh`

Note that: a folder test/test_cache will be created to store all the temporary files, which contains:

1. The c++ build files.
2. The dart dependency files.
3. A shallow clone of Crafting Interpreters that stores all the test cases.
