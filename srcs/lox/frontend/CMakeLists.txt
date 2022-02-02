
add_library(frontend scanner.cc parser.cc)
target_link_libraries(frontend PUBLIC token ast lox_common)
