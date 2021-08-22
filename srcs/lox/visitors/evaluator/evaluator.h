//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_EVALUATOR_EVAL_VISITOR_H_
#define CPPLOX_SRCS_LOX_EVALUATOR_EVAL_VISITOR_H_

#include <cassert>

#include "environment.h"
#include "lox/ast/ast.h"
#include "lox/lox_object/lox_object.h"
#include "lox/passes/env_resolve_pass/resolve_map.h"
namespace lox {

struct ReturnValue : public std::exception {
  explicit ReturnValue(object::LoxObject obj) : ret(std::move(obj)) {}

  object::LoxObject ret;
};

class Evaluator : public AstNodeVisitor {
 public:
  explicit Evaluator(std::shared_ptr<Environment> env) : work_env_(std::move(env)) {}

  object::LoxObject Eval(std::shared_ptr<AstNode> node) {
    assert(IsValid(node));
    assert(active_map_);
    return node->Accept(this);
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
  object::LoxObject Visit(LogicalExpr* state) override;
  object::LoxObject Visit(BinaryExpr* state) override;
  object::LoxObject Visit(GroupingExpr* state) override;
  object::LoxObject Visit(LiteralExpr* state) override;
  object::LoxObject Visit(UnaryExpr* state) override;
  object::LoxObject Visit(VariableExpr* state) override;
  object::LoxObject Visit(AssignExpr* state) override;
  object::LoxObject Visit(CallExpr* state) override;
  object::LoxObject Visit(GetAttrExpr* state) override;
  object::LoxObject Visit(SetAttrExpr* state) override;
  object::LoxObject Visit(PrintStmt* state) override;
  object::LoxObject Visit(ReturnStmt* state) override;
  object::LoxObject Visit(WhileStmt* state) override;
  object::LoxObject Visit(ExprStmt* state) override;
  object::LoxObject Visit(BreakStmt* state) override;
  object::LoxObject Visit(VarDeclStmt* state) override;
  object::LoxObject Visit(FunctionStmt* state) override;
  object::LoxObject Visit(ClassStmt* state) override;
  object::LoxObject Visit(BlockStmt* state) override;
  object::LoxObject Visit(IfStmt* state) override;
};

}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_EVALUATOR_EVAL_VISITOR_H_
