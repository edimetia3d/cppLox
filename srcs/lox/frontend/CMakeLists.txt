
cc_library(frontend scanner.cc parser.cc)
target_link_libraries(lox.frontend PRIVATE lox.token lox.ast lox.common)
