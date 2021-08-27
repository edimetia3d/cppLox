//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_AST_AST_H_
#define CPPLOX_SRCS_LOX_AST_AST_H_
#include "lox/backend/tree_walker/ast/ast_node.h"
#include "lox/backend/tree_walker/ast/expr.h"
#include "lox/backend/tree_walker/ast/stmt.h"

namespace lox {
static inline void BindParent(Token token, AstNode* parent) {
  // token do not need parent;
}
static inline void BindParent(const std::vector<Token>& tokens, AstNode* parent) {
  // token do not need parent;
}
static inline bool IsModified(Token token) { return false; }

static inline bool IsModified(const std::vector<Token>& tokens) { return false; }

static inline void ResetModify(Token token) {
  // token do not need parent;
}
static inline void ResetModify(const std::vector<Token>& tokens) {
  // token do not need parent;
}
}  // namespace lox
#ifdef DYNAMIC_GEN_DECL
#include "lox/backend/tree_walker/ast/ast_decl_dynamic.h.inc"
#else
#include "lox/backend/tree_walker/ast/ast_decl.h.inc"
#endif

#endif  // CPPLOX_SRCS_LOX_AST_AST_H_
