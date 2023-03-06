cc_library(builtins SHARED builtin.cc) # build shared to keep symbols in final executable
set_target_properties(lox.backend.llvm.builtins PROPERTIES OUTPUT_NAME "llvm_builtins")