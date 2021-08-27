
add_library(resolve_map resolve_map.cc)

add_library(env_reslove_pass env_reslove_pass.cc)
target_link_libraries(env_reslove_pass PUBLIC ast)
target_link_libraries(env_reslove_pass PUBLIC resolve_map)
