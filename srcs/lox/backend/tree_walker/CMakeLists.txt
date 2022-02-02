
file(GLOB LOX_SRCS ${CMAKE_CURRENT_LIST_DIR}/*.cc)
add_library(tree_walker ${LOX_SRCS})

add_subdirectory(bultins)
add_subdirectory(evaluator)

target_link_libraries(tree_walker PRIVATE evaluator)
