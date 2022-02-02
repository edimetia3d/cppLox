
add_subdirectory(semantic_check)
add_library(passes pass_runner.cc)
target_link_libraries(passes PUBLIC semantic_check)
