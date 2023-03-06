
add_subdirectory(core)
add_subdirectory(object)
add_subdirectory(builtins)
add_subdirectory(debug)

cc_library(virtual_machine virtual_machine.cc)

target_link_libraries(lox.backend.virtual_machine PRIVATE
    lox.backend.virtual_machine.core
    lox.backend.virtual_machine.object
    lox.backend.virtual_machine.debug
    lox.common)
