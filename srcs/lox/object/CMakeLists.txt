add_library(object object.cc gc.cc value.cc)
add_library(lox::object:object ALIAS object)