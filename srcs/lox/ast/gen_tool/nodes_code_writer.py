import datetime
import os.path
import sys

import yaml

file_template = """
// clang-format off
// DO NOT EDIT: this file is generated by {this_file_name} at {current_time}
// The file in source tree will only be used when python3 is not found by cmake, and might be out of sync.
#ifndef LOX_AST_AST_NODE_DECL_H_INC
#define LOX_AST_AST_NODE_DECL_H_INC
#include <memory>
#include "lox/token/token.h"
#include "lox/ast/ast_node.h"

namespace lox{{
{class_forward_decl}
class IASTNodeVisitor {{
public:
virtual ~IASTNodeVisitor() = default;
protected:
{virtual_visit_decls}
}};

{class_decls}

}} // namespace lox

#endif // LOX_AST_AST_NODE_DECL_H_INC
// clang-format on
"""


def get_class_list(tpl):
    ret = []
    for category in tpl:
        for cls in tpl[category]:
            ret.append(f"{cls}{category}")
    return ret


def gen_class_forward_decl(tpl):
    ret = ""
    for cls in get_class_list(tpl):
        ret += f"class {cls};\n"
    return ret


def gen_visitor_visit(tpl):
    ret = ""
    for cls in get_class_list(tpl):
        ret += f"""
friend class {cls};
virtual void Visit({cls} *) = 0;
"""
    return ret


def gen_single_class(base_class, class_name, attrs, children):
    template = """
struct {class_name}Attr: public ASTNodeAttr{{
{attr_decl}
}};

class {class_name}:public {base_class} {{
public:
{class_name}(const {class_name}Attr & attr_{split_comma}{child_param_list}):attr(std::make_unique<{class_name}Attr>(attr_)){split_comma}{child_init_list}{{
{child_bind}
}}

void Accept(IASTNodeVisitor *visitor) override {{
    visitor->Visit(this);
}}

std::unique_ptr<{class_name}Attr> attr;
{child_decl}
}};
"""
    attr_decl = "\n".join([f"{attr[0]} {attr[1]};" for attr in attrs])
    child_param_list = ",".join([f"{child[0]}&& {child[1]}_" for child in children])
    split_comma = "," if child_param_list else ""
    child_init_list = ",".join([f"{child[1]}(std::move({child[1]}_))" for child in children])
    child_bind = "\n".join(f"AddChild(&{child[1]});" for child in children)
    child_decl = "\n".join(f"{child[0]} {child[1]};" for child in children)
    return template.format(
        class_name=class_name,
        attr_decl=attr_decl,
        base_class=base_class,
        split_comma=split_comma,
        child_param_list=child_param_list,
        child_init_list=child_init_list,
        child_bind=child_bind,
        child_decl=child_decl
    )


def split_into_list(values):
    ret = []
    if isinstance(values, list):
        for v in values:
            ret += split_into_list(v)
    else:
        ret.append(values.split(" "))
    return ret


def gen_class_decls(tpl):
    ret = ""
    for category in tpl:
        for cls in tpl[category]:
            base_class = category
            class_name = f"{cls}{category}"
            attrs = []
            children = []
            if "attr" in tpl[category][cls]:
                attrs = split_into_list(tpl[category][cls]["attr"])
            if "child" in tpl[category][cls]:
                children = split_into_list(tpl[category][cls]["child"])
            ret += gen_single_class(base_class, class_name, attrs, children) + "\n"

    return ret


def gen_code(tpl, output_file):
    output_file.write(file_template.format(
        this_file_name=os.path.basename(__file__),
        current_time=datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        class_forward_decl=gen_class_forward_decl(tpl),
        virtual_visit_decls=gen_visitor_visit(tpl),
        class_decls=gen_class_decls(tpl))

    )


if __name__ == "__main__":
    input_file_path = "../ast_node_def.yaml"
    output_file_path = "../ast_nodes_decl.h.inc"
    if len(sys.argv) >= 2:
        input_file_path = sys.argv[1]

    if len(sys.argv) >= 3:
        output_file_path = sys.argv[2]

    with open(input_file_path, "r") as stream:
        tpl = yaml.safe_load(stream)

    with open(output_file_path, "w") as output_file:
        gen_code(tpl, output_file)
