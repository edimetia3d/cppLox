import os.path
import sys
import datetime

file_template = """
// clang-format off
// DO NOT EDIT: this file is generated by {this_file_name} at {current_time}
// The file in source tree will only be used when python3 is not found by cmake, and might be out of sync.

#include <memory>
#include <stack>
#include "lox/frontend/token.h"
namespace lox{{
{class_forward_decl}
class IAstNodeVisitor {{

protected:
{virtual_visit_decls}
}};

#define VisitorReturn(arg) _Return(arg);return;
template<class T>
class AstNodeVisitor:public IAstNodeVisitor{{
public:
  void _Return(T&& new_ret){{
    ret_stk_.push(std::move(new_ret));
  }}
  void _Return(const T& new_ret){{
    ret_stk_.push(std::move(new_ret));
  }}
  T PopRet(){{
    auto ret = ret_stk_.top();
    ret_stk_.pop();
    return ret;
  }}
protected:
  std::stack<T> ret_stk_;
}};




{class_decls}

}} // namespace lox
// clang-format on
"""

class_template = """
class {class_name}{target_key}:public {target_key}Base
{{
private:
explicit {class_name}{target_key}({init_params})
:{init}{{
{set_parent}
}}
friend AstNode;
public:
{member_def}

bool IsModified() override{{
    return {member_call_is_modify};
}}
void ResetModify() override{{
    is_modified_=false;
    {member_reset_modify};
}}
void Accept(IAstNodeVisitor * visitor) override {{
  return visitor->Visit(this);
}}
}};
"""


def get_targetkey(classname, all_dict):
    if classname in all_dict["Stmt"]:
        return "Stmt"
    if classname in all_dict["Expr"]:
        return "Expr"

    raise RuntimeError("No target found")


def gen_code(input_file_path, output_file):
    import re
    with open(input_file_path, "r") as f:
        split_str = re.split("({|})", f.read())
        all_dict = {"Stmt": {}, "Expr": {}}
        i = 0
        while i < len(split_str):
            if split_str[i] == "{":
                all_dict[split_str[i - 1].strip()].update(eval("{" + split_str[i + 1] + "}"))
                i = i + 3
            i = i + 1
    all_def = all_dict["Stmt"].copy()
    all_def.update(all_dict["Expr"])

    class_decls = ""
    virtual_visit_decls = ""
    class_forward_decl = ""
    type_id = 10000
    for class_name in all_def:
        target_key = get_targetkey(class_name, all_dict)
        type_id += 1
        virtual_visit_decls += f"""
friend class {class_name}{target_key};
virtual void Visit({class_name}{target_key} *) = 0;
"""
        class_forward_decl += f"""class {class_name}{target_key};\n"""
        member_list = all_def[class_name].split(",")
        member_def = []
        member_call_is_modify = ["is_modified_"]
        member_reset_modify = []
        member_init_params = [f"{target_key}Base *parent"]
        member_init = [f"{target_key}Base(parent)"]
        set_parent = []
        for member in member_list:
            cut_by_space = list(filter(lambda x: x != "", member.split(" ")))
            member_type = cut_by_space[0]
            member_name = cut_by_space[1]
            member_def.append(f"""
private:
{member_type} {member_name}_;
public:
const {member_type} & {member_name}(){{
    return {member_name}_;
}}
void {member_name}({member_type} new_value){{
     if(new_value != {member_name}_){{
        is_modified_ = true;
     }}
     {member_name}_=new_value;
     BindParent({member_name}_,this);
}}
""")
            member_call_is_modify.append(f"::lox::IsModified({member_name}())")
            member_reset_modify.append(f"::lox::ResetModify({member_name}())")
            member_init_params.append(f"{member_type} {member_name}_in")
            member_init.append(f"{member_name}_(std::move({member_name}_in))")
            set_parent.append(f"BindParent({member_name}_,this);")
        member_init = ",\n".join(member_init)
        member_call_is_modify = " || ".join(member_call_is_modify)
        member_reset_modify = ";\n".join(member_reset_modify)
        member_def = "\n".join(member_def)
        member_init_params = ",".join(member_init_params)
        set_parent = "\n".join(set_parent)
        class_decls = class_decls + class_template.format(target_key = target_key,
                                                          member_call_is_modify = member_call_is_modify,
                                                          member_reset_modify = member_reset_modify,
                                                          class_name = class_name,
                                                          init_params = member_init_params,
                                                          init = member_init,
                                                          member_def = member_def,
                                                          set_parent = set_parent,
                                                          type_id = type_id)
    output_file.write(file_template.format(
        this_file_name = os.path.basename(__file__),
        current_time = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        class_forward_decl = class_forward_decl,
        class_decls = class_decls,
        virtual_visit_decls = virtual_visit_decls)
    )


if __name__ == "__main__":
    input_file_path = "ast_node_def.tpl"
    output_file_path = "ast_decl.tmp.h"
    if len(sys.argv) >= 2:
        input_file_path = sys.argv[1]

    if len(sys.argv) >= 3:
        output_file_path = sys.argv[2]

    with open(output_file_path, "w") as output_file:
        gen_code(input_file_path, output_file)
