
file(GLOB LOX_SRCS ${CMAKE_CURRENT_LIST_DIR}/*.cc)
add_library(tree_walker ${LOX_SRCS})

add_subdirectory(lox_object)
add_subdirectory(evaluator)
add_subdirectory(passes)

target_link_libraries(tree_walker PRIVATE lox_object evaluator tree_walker_passes)
