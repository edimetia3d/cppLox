

add_library(lox lox.cc)

add_subdirectory(common)
add_subdirectory(object)
add_subdirectory(version)
add_subdirectory(frontend)
add_subdirectory(backend)
add_subdirectory(token)
add_subdirectory(ast)
add_subdirectory(passes)
target_link_libraries(lox PRIVATE version_info)
target_link_libraries(lox PRIVATE backend frontend)
target_link_libraries(lox PRIVATE token ast passes)