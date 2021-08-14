
add_library(environment environment.cpp)
target_link_libraries(environment PUBLIC lox_object)


add_library(evaluator evaluator.cpp callable_object.cc)
target_link_libraries(evaluator PUBLIC lox_object)
target_link_libraries(evaluator PUBLIC ast)
target_link_libraries(evaluator PUBLIC environment)

