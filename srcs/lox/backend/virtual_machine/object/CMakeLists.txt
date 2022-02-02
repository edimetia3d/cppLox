
add_library(vm_object object.cc value.cc)
target_link_libraries(vm_object PUBLIC lox::object:object)