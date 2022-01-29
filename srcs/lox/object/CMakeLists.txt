add_library(lox_object_object object.cc gc.cc value.cc)
add_library(lox::object:object ALIAS lox_object_object)