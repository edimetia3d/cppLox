
add_library(bytecode chunk.cc ../core/debug.cc)
target_link_libraries(bytecode PUBLIC common)
