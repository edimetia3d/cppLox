//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_AST_STMT_H_
#define CPPLOX_SRCS_LOX_AST_STMT_H_

#include <type_traits>
#include <vector>

#include "lox/ast/expr.h"

namespace lox {

class StmtState;
class StmtVisitor;
template <class T>
concept SubclassOfStmtState = std::is_base_of<StmtState, T>::value;

class Stmt {
 public:
  /**
   * This fn will take ownership of object state_ pointed to
   */
  explicit Stmt(StmtState* state) : state_(state) {}
  explicit Stmt(std::shared_ptr<StmtState> state) : state_(std::move(state)) {}
  bool IsValid() { return static_cast<bool>(state_); }
  object::LoxObject Accept(StmtVisitor* visitor);

  template <SubclassOfStmtState T>
  T* DownCastState() {
    return dynamic_cast<T*>(state_.get());
  }

 private:
  std::shared_ptr<StmtState> state_;
};
}  // namespace lox

#ifdef DYNAMIC_GEN_DECL
#include "lox/ast/stmt_decl_dynamic.h.inc"
#else
#include "lox/ast/stmt_decl.h.inc"
#endif

#endif  // CPPLOX_SRCS_LOX_AST_STMT_H_
