add_library(jit_builtins builtin.cc)
add_library(lox::backend::jit::builtins ALIAS jit_builtins)
