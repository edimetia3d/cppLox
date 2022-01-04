
add_subdirectory(core)
add_subdirectory(object)
add_subdirectory(builtins)
add_subdirectory(debug)

add_library(virtual_machine virtual_machine.cc)

target_link_libraries(virtual_machine PRIVATE core vm_object vm_debug)
