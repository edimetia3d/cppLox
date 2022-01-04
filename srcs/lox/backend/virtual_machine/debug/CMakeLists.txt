add_library(vm_debug debug.cc)
target_link_libraries(vm_debug PUBLIC vm_object)