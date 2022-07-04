add_library(jit_builtins SHARED builtin.cc) # build shared to keep symbols in final executable
add_library(lox::backend::jit::builtins ALIAS jit_builtins)
