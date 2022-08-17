
add_llvm_library(add_no_alias_pass add_no_alias.cc)
target_compile_options(add_no_alias_pass PRIVATE -fPIC)
add_llvm_pass_plugin(add_no_alias_plugin
        add_no_alias_plugin.cc
        LINK_LIBS
        add_no_alias_pass)

add_llvm_library(op_counter_analysis op_counter.cc)
target_compile_options(op_counter_analysis PRIVATE -fPIC)
add_llvm_pass_plugin(op_counter_analysis_plugin
        op_counter_plugin.cc
        LINK_LIBS
        op_counter_analysis)