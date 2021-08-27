

add_library(semantic_check semantic_check.cc)
target_link_libraries(semantic_check PUBLIC ast)
