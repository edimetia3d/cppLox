//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_EVALUATOR_EVAL_VISITOR_H_
#define CPPLOX_SRCS_LOX_EVALUATOR_EVAL_VISITOR_H_

#include <cassert>

#include "environment.h"
#include "lox/ast/expr.h"
#include "lox/ast/stmt.h"
#include "lox/lox_object/lox_object.h"
#include "lox/visitors/resolver_pass/resolve_map.h"
namespace lox {
class Evaluator : public ExprVisitor, public StmtVisitor {
 public:
  explicit Evaluator(std::shared_ptr<Environment> env) : work_env_(std::move(env)) {}

  object::LoxObject Eval(Expr expr) {
    assert(expr.IsValid());
    assert(active_map_);
    return expr.Accept(this);
  }
  object::LoxObject Eval(Stmt stmt) {
    assert(stmt.IsValid());
    assert(active_map_);
    return stmt.Accept(this);
  }

  void SetActiveResolveMap(std::shared_ptr<EnvResolveMap> map) { active_map_ = map; }

  std::shared_ptr<Environment>& WorkEnv() { return work_env_; }
  std::shared_ptr<Environment> WorkEnv(std::shared_ptr<Environment> new_env) {
    auto old_env = work_env_;
    work_env_ = new_env;
    return old_env;
  }

  std::shared_ptr<Environment> FreezeEnv() {
    auto old_env = work_env_;
    work_env_ = Environment::Make(old_env);
    return old_env;
  }

  struct EnterNewScopeGuard {
    EnterNewScopeGuard(Evaluator* ev, std::shared_ptr<Environment> base_env = nullptr) : evaluator(ev) {
      std::shared_ptr<Environment> new_env;
      if (!base_env) {
        new_env = Environment::Make(evaluator->WorkEnv());
      } else {
        new_env = Environment::Make(base_env);
      }
      backup = evaluator->WorkEnv(new_env);
    }
    ~EnterNewScopeGuard() { evaluator->WorkEnv(backup); }
    std::shared_ptr<Environment> backup;
    Evaluator* evaluator;
  };


 protected:
  std::shared_ptr<Environment> work_env_;
  std::shared_ptr<EnvResolveMap> active_map_;
  object::LoxObject Visit(LogicalState* state) override;
  object::LoxObject Visit(BinaryState* state) override;
  object::LoxObject Visit(GroupingState* state) override;
  object::LoxObject Visit(LiteralState* state) override;
  object::LoxObject Visit(UnaryState* state) override;
  object::LoxObject Visit(VariableState* state) override;
  object::LoxObject Visit(AssignState* state) override;
  object::LoxObject Visit(CallState* state) override;
  object::LoxObject Visit(GetAttrState* state) override;
  object::LoxObject Visit(PrintStmtState* state) override;
  object::LoxObject Visit(ReturnStmtState* state) override;
  object::LoxObject Visit(WhileStmtState* state) override;
  object::LoxObject Visit(ExprStmtState* state) override;
  object::LoxObject Visit(BreakStmtState* state) override;
  object::LoxObject Visit(VarDeclStmtState* state) override;
  object::LoxObject Visit(FunctionStmtState* state) override;
  object::LoxObject Visit(ClassStmtState* state) override;
  object::LoxObject Visit(BlockStmtState* state) override;
  object::LoxObject Visit(IfStmtState* state) override;
};

}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_EVALUATOR_EVAL_VISITOR_H_
