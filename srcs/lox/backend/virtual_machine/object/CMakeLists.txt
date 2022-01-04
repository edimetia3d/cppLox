
add_library(vm_object object.cc)
target_link_libraries(vm_object PUBLIC lox::object:object)