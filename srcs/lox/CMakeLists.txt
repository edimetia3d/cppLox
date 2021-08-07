
file(GLOB LOX_SRCS ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
add_library(lox ${LOX_SRCS})

add_subdirectory(version)
add_subdirectory(lox_object)
add_subdirectory(ast)

target_link_libraries(lox PUBLIC lox_public_header)
target_link_libraries(lox PRIVATE version_info)
target_link_libraries(lox PRIVATE ast)
