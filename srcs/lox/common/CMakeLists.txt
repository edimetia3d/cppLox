
cc_library(common lox_error.cc global_setting.cc)
target_compile_options(lox.common PRIVATE -fPIC)