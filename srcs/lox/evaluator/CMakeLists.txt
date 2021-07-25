
add_library(evaluator
        eval_visitor.cpp
        lox_object.cpp
        )

target_link_libraries(evaluator PUBLIC ast)
