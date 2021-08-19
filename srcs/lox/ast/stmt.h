//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_AST_STMT_H_
#define CPPLOX_SRCS_LOX_AST_STMT_H_

#include <type_traits>
#include <vector>

#include "lox/ast/expr.h"

namespace lox {
class StmtBase;
template <class T>
concept SubclassOfStmt = std::is_base_of<StmtBase, T>::value;

class StmtVisitor;
class StmtBase {
 public:
  virtual ~StmtBase() {}

  virtual object::LoxObject Accept(StmtVisitor* visitor) = 0;

  template <SubclassOfStmt T>
  T* DownCastState() {
    return dynamic_cast<T*>(this);
  }
};

using Stmt = std::shared_ptr<StmtBase>;

static inline bool IsValid(const Stmt& stmt) { return stmt.get(); }
}  // namespace lox

#ifdef DYNAMIC_GEN_DECL
#include "lox/ast/stmt_decl_dynamic.h.inc"
#else
#include "lox/ast/stmt_decl.h.inc"
#endif

#endif  // CPPLOX_SRCS_LOX_AST_STMT_H_
