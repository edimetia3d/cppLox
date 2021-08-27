
add_library(core vm.cc compiler.cc)
target_link_libraries(core PUBLIC bytecode common)
