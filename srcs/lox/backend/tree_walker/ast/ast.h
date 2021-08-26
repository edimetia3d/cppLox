//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_AST_AST_H_
#define CPPLOX_SRCS_LOX_AST_AST_H_
#include "lox/backend/tree_walker/ast/ast_node.h"
#include "lox/backend/tree_walker/ast/expr.h"
#include "lox/backend/tree_walker/ast/stmt.h"

#ifdef DYNAMIC_GEN_DECL
#include "lox/backend/tree_walker/ast/ast_decl_dynamic.h.inc"
#else
#include "lox/backend/tree_walker/ast/ast_decl.h.inc"
#endif

#endif  // CPPLOX_SRCS_LOX_AST_AST_H_
