add_library(builtins builtin_fn.cc)
add_library(lox::backend::virtual_machine::builtins ALIAS builtins)
