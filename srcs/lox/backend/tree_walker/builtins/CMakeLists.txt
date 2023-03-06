cc_library(builtins builtin_fn.cc)
set_target_properties(lox.backend.tree_walker.builtins PROPERTIES OUTPUT_NAME "tw_builtins")