//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_AST_AST_H_
#define CPPLOX_SRCS_LOX_AST_AST_H_
#include "lox/frontend/ast/ast_node.h"
#include "lox/frontend/ast/expr.h"
#include "lox/frontend/ast/stmt.h"
#include "lox/frontend/token.h"

#ifdef DYNAMIC_GEN_DECL
#include "lox/frontend/ast/ast_decl_dynamic.h.inc"
#else
#include "lox/frontend/ast/ast_decl.h.inc"
#endif

#endif  // CPPLOX_SRCS_LOX_AST_AST_H_
