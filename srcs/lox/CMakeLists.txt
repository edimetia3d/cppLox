

add_library(lox global_setting.cc lox.cc lox_error.cc)

add_subdirectory(version)
add_subdirectory(frontend)
add_subdirectory(backend)
add_subdirectory(token)
add_subdirectory(ast)
add_subdirectory(passes)
target_link_libraries(lox PRIVATE version_info)
target_link_libraries(lox PRIVATE backend frontend)
target_link_libraries(lox PRIVATE token ast passes)
