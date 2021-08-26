

add_library(semantic_check semantic_check.cc)
target_link_libraries(env_reslove_pass PUBLIC ast)
target_link_libraries(env_reslove_pass PUBLIC lox_object)
