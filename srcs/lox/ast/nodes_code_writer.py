import os.path
import sys
import datetime

file_template = """
// clang-format off
// DO NOT EDIT: this file is generated by {this_file_name} at {current_time}
// The file in source tree will only be used when python3 is not found by cmake, and might be out of sync.

namespace lox{{

namespace private_ns{{

class ExprImpl {{
 public:
  virtual ~ExprImpl() {{
    // just make ExprImpl a virtual class to support dynamic_cast
  }}
  virtual int TypeId() = 0;
}};

}} // namespace private_ns

{class_decls}

template <class RetT>
class Visitor {{

public:
RetT Visit(Expr * expr){{
switch(expr->ImplHandle()->TypeId()){{
{dispatch_call}
default: throw "Dispatch Fail";
}}
}}

protected:
{virtual_visit_decls}
}};

}} // namespace lox
// clang-format on
"""

class_template = """
class {class_name}:public private_ns::ExprImpl
{{
public:
explicit {class_name}({init_params})
:{init}{{}}
{member_def}
int TypeId() override {{
    return {type_id};
}}
}};
"""


def gen_code(input_file_path, output_file_path):
    with open(input_file_path, "r") as f:
        all_def = eval(f.read())

    with open(output_file_path, "w") as output_file:
        class_decls = ""
        dispatch_call = ""
        virtual_visit_decls = ""
        type_id = 10000
        for class_name in all_def:
            type_id += 1
            dispatch_call += f"case {type_id}:return Visit(static_cast<{class_name} *>(expr->ImplHandle()));\n"
            virtual_visit_decls += f"""
virtual RetT Visit({class_name} *) = 0;
"""
            member_list = all_def[class_name].split(",")
            member_def = ""
            member_init_params = []
            member_init = []
            for member in member_list:
                cut_by_space = list(filter(lambda x: x != "", member.split(" ")))
                member_type = cut_by_space[0]
                member_name = cut_by_space[1]
                member_def = member_def + f"{member_type} {member_name};\n"
                member_init_params.append(f"{member_type} {member_name}")
                member_init.append(f"{member_name}(std::move({member_name}))")
            member_init = ",\n".join(member_init)
            member_init_params = ",".join(member_init_params)
            class_decls = class_decls + class_template.format(class_name=class_name,
                                                              init_params=member_init_params,
                                                              init=member_init,
                                                              member_def=member_def,
                                                              type_id=type_id)
        output_file.write(file_template.format(this_file_name=os.path.basename(__file__),
                                               current_time=datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                                               class_decls=class_decls,
                                               virtual_visit_decls=virtual_visit_decls,
                                               dispatch_call=dispatch_call)
                          )


if __name__ == "__main__":
    input_file_path = sys.argv[1]
    if len(sys.argv) == 3:
        output_file_path = sys.argv[2]
    else:
        output_file_path = "tmp_gen_output.h.inc"
    gen_code(input_file_path, output_file_path)
