
cc_library(common lox_error.cc global_setting.cc location.cc input_file.cc)
target_compile_options(lox.common PRIVATE -fPIC)