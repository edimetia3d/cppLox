
add_library(backend bakcend.cc)

add_subdirectory(tree_walker)
add_subdirectory(virtual_machine)

target_link_libraries(backend PRIVATE tree_walker virtual_machine)
