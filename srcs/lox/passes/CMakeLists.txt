
add_subdirectory(env_resolve_pass)

add_library(passes pass_runner.cc pass_manager.cc)
target_link_libraries(passes PUBLIC ast)
target_link_libraries(passes PUBLIC lox_object)
