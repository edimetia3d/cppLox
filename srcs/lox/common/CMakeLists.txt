
add_library(lox_common lox_error.cc global_setting.cc)
target_compile_options(lox_common PRIVATE -fPIC)