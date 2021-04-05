
add_subdirectory(lox)

target_include_directories(lox PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(version_info PUBLIC lox_public_header)
target_link_libraries(lox PUBLIC lox_public_header)
target_link_libraries(lox PRIVATE version_info)