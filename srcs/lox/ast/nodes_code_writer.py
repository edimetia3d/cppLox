import sys
from os import path

all_def = {
    "Binary": "ExprPointer left, Token op, ExprPointer right",
    "Grouping": "ExprPointer expression",
    "Literal": "Token value",
    "Unary": "Token op, ExprPointer right"
}

file_template = """
namespace lox{{
class Expr;
class Token;
template<class RetT>
class Visitor;

{class_decls}

template <class RetT>
class Visitor {{
protected:
{virtual_visit_decls}
}};

template <class RetT>
RetT Expr::Accept(const Visitor<RetT>& v) {{
{dispatch_call}
throw "Dispatch Fail";
}}

template <class RetT>
RetT Expr::Accept(const Visitor<RetT>& v) const {{
{const_dispatch_call}
throw "Dispatch Fail";
}}

}} // namespace lox
"""

class_template = """
class {class_name}:public Expr
{{
public:
{class_name}({init_params})
:{init}{{}}
{member_def}

template <class RetT>
RetT _Accept(const Visitor<RetT>& visitor){{
  return visitor.Visit{class_name}(this);
}}

template <class RetT>
RetT _Accept(const Visitor<RetT>& visitor) const {{
  return visitor.Visit{class_name}(*this);
}}
  
}};
"""


def gen_code(output_file_path):
    with open(output_file_path, "w") as output_file:
        class_decls = ""
        dispatch_call = ""
        const_dispatch_call = ""
        virtual_visit_decls = ""
        for class_name in all_def:
            dispatch_call += f"if(auto p = dynamic_cast<{class_name} *>(this)){{return p->_Accept(v);}}\n"
            const_dispatch_call += f"if(auto p = dynamic_cast<const {class_name} *>(this)){{return p->_Accept(v);}}\n"
            virtual_visit_decls += f"""
friend class {class_name};
virtual RetT Visit{class_name}({class_name} &) const{{
throw "No Impl";
}};
virtual RetT Visit{class_name}(const {class_name} &) const{{
throw "No Impl";
}}"""
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
                member_init.append(f"{member_name}({member_name})")
            member_init = ",\n".join(member_init)
            member_init_params = ",".join(member_init_params)
            class_decls = class_decls + class_template.format(class_name=class_name,
                                                              init_params=member_init_params,
                                                              init=member_init,
                                                              member_def=member_def)
        output_file.write(file_template.format(class_decls=class_decls,
                                               virtual_visit_decls=virtual_visit_decls,
                                               dispatch_call=dispatch_call,
                                               const_dispatch_call=const_dispatch_call)
                          )


if __name__ == "__main__":
    if len(sys.argv) == 2:
        output_file_path = sys.argv[1]
    else:
        output_file_path = "tmp_gen_output.h.inc"
    gen_code(output_file_path)
