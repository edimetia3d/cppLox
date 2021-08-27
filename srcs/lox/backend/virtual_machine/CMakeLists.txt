
add_subdirectory(core)
add_subdirectory(common)
add_subdirectory(bytecode)

add_library(virtual_machine virtual_machine.cc)

target_link_libraries(virtual_machine PRIVATE core common bytecode)
