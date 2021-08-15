
add_library(resolver resovler.cc)
target_link_libraries(resolver PUBLIC ast)
target_link_libraries(resolver PUBLIC lox_object)
