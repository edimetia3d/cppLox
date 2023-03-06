cc_library(builtins builtin_fn.cc)
set_target_properties(lox.backend.virtual_machine.builtins PROPERTIES OUTPUT_NAME "vm_builtins")
