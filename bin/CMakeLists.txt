
add_executable(lox-bin main.cpp)
target_link_libraries(lox-bin lox)
set_target_properties(lox-bin PROPERTIES OUTPUT_NAME "lox")
target_link_libraries(lox-bin)
