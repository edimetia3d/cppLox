
cc_library(core vm.cc compiler.cc chunk.cc function_unit.cc)
target_link_libraries(lox.backend.virtual_machine.core PRIVATE
    lox.backend.virtual_machine.object
    lox.backend.virtual_machine.builtins
    lox.frontend)

