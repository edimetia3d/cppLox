# cppLox

This repo is a cpp Lox interpreter of Lox Language from [Crafting Interpreters](https://craftinginterpreters.com/)

Hope this will be a useful one I would use in future.

There are (will be) three versions of Lox interpreter, they all share a same frontend, and has their own backend.

1. Frontend contains:
    - [x] A scanner to build tokens from raw code.
    - [x] A RD Parser.
    - [x] A Pratt Parser.
    - [x] An AST definition with a simple pass-manager.

2. Backend contains:
   - [x] A naive tree-walker interpreter, it runs directly on the AST from frontend.
   - [x] A virtual machine interpreter, this one uses only scanner from frontend, and do a "Pratt Parsing" style one
     pass compilation, which convert tokens to bytecode directly.
   - [ ] WIP: A MLIR based JIT interpreter, it will first run a pass on AST to generate an MLIR IR of Lox Dialect, and
     then lower to LLVM dialect, so we can leverage the LLVM JIT utilities to do a JIT run.

# What's the difference ?

1. Pure C++ implementation, no need to know Java things.
   * Because of this, the jlox and clox implementation shares a lot of code, which make transition from jlox to clox
     much easier.
   * An object system with GC that used for both jlox and clox.
2. A more clear implementation, code is (hopefully) more readable more.
   * Especially true when comparing with original clox's c style code.
3. Files are C++ CMake style organized.
   * `bin/lox`: the main executable, which is a wrapper of the `liblox` library.
   * `bin/lox-format`: a tool to format Lox code, which is based on the AST Printer.
   * `src/**`: just look at the file-tree, it tells itself.
4. A standalone AST implementation, which is a good way to understand the AST.

# Build

## Requirements

### For Buiding

A C++20 compiler is all you need. GCC >= 10.0 or clang >= 10.0 sould be fine,
check [cppreference](https://en.cppreference.com/w/cpp/compiler_support/20) to see more.

### For Testing

**Locally**:

If you want to run the test cases in native environment, [dart-sdk](https://dart.dev/tools/sdk) is required.

**Dockerized**:

If you had docker installed, you can run the test cases in docker environment.

## Build Instructions

```bash
git clone https://github.com/edimetia3d/cppLox.git --recursive
cd cppLox
mkdir build
cmake ..
make
```

### CMAKE Options

1. `-DUPSTREAM_STYLE_ERROR_MSG=ON`. This impl's parsing(compiling)/err-handling logic is a little different from
   upstream. Enable this flag will make error messages behaves like upstream. It is OFF by default, but for the unit
   tests, it is set to ON.

# Test

This project leverages test from [Crafting Interpreters GitHub](https://github.com/munificent/craftinginterpreters), so
you can check the test cases from there.

To launch the test, you can use either docker or native machine.

1. If you have docker, just launch the script `./test/launch_unittest_with_docker.sh`
2. If you want to run the test in native machine, you can launch the `./test/unittest.sh`

A folder test/test_cache will be created to store all the temporary files, which are:

1. The c++ build files.
2. The dart dependency files.
3. A shallow clone of Crafting Interpreters that stores all the test cases.
