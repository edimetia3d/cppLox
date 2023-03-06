cc_library(debug debug.cc)
target_link_libraries(lox.backend.virtual_machine.debug PRIVATE lox.backend.virtual_machine.object)