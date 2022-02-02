
add_library(environment environment.cpp)
target_link_libraries(environment PUBLIC runtime_object)

add_library(evaluator
        evaluator.cpp
        )

add_library(runtime_object
        runtime_object.cc
        )
target_link_libraries(evaluator PUBLIC ast)
target_link_libraries(evaluator PUBLIC environment)
target_link_libraries(evaluator PUBLIC lox_common)
target_link_libraries(evaluator PUBLIC runtime_object)
target_link_libraries(evaluator PUBLIC talker_builtins)
