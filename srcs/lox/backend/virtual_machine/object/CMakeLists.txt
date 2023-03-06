
cc_library(object object.cc value.cc)
target_link_libraries(lox.backend.virtual_machine.object PRIVATE lox.object)
set_target_properties(lox.backend.virtual_machine.object PROPERTIES OUTPUT_NAME "vm_object")