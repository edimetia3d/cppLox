

add_subdirectory(common)
add_subdirectory(object)
add_subdirectory(version)
add_subdirectory(frontend)
add_subdirectory(backend)
add_subdirectory(token)
add_subdirectory(ast)
add_subdirectory(passes)

cc_library(lox lox.cc)
target_link_libraries(lox PRIVATE lox.version)
target_link_libraries(lox PRIVATE lox.backend lox.frontend)
target_link_libraries(lox PRIVATE lox.token lox.ast)