
add_library(resolve_map resolve_map.cc)

add_library(resolver resovler.cc)
target_link_libraries(resolver PUBLIC ast)
target_link_libraries(resolver PUBLIC lox_object)
target_link_libraries(resolver PRIVATE resolve_map)
