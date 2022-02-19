
add_subdirectory(IR)
add_subdirectory(Transforms)

target_link_libraries(MLIRLox PUBLIC lox_transforms)