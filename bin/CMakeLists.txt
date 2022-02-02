
add_executable(lox-bin main.cpp)
target_link_libraries(lox-bin lox)
set_target_properties(lox-bin PROPERTIES OUTPUT_NAME "lox")

add_executable(lox-format lox-format.cc)
target_link_libraries(lox-format frontend)
