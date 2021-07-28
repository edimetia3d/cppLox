
add_library(evaluator
        eval_visitor.cpp
        )

target_link_libraries(evaluator PUBLIC ast)
