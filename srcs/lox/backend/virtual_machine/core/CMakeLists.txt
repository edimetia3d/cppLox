
add_library(core vm.cc compiler.cc chunk.cc function_unit.cc)
target_link_libraries(core PUBLIC vm_object lox::backend::virtual_machine::builtins)
