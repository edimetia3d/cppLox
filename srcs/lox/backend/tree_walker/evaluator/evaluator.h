//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_EVALUATOR_EVAL_VISITOR_H_
#define CPPLOX_SRCS_LOX_EVALUATOR_EVAL_VISITOR_H_

#include <cassert>

#include "lox/ast/ast_node_visitor.h"
#include "lox/backend/tree_walker/evaluator/environment.h"
#include "lox/backend/tree_walker/lox_object/lox_object.h"
#include "lox/backend/tree_walker/passes/env_resolve_pass/resolve_map.h"
namespace lox {

struct ReturnValue : public std::exception {
  explicit ReturnValue(object::LoxObject obj) : ret(std::move(obj)) {}

  object::LoxObject ret;
};

class Evaluator : public AstNodeVisitor<object::LoxObject> {
 public:
  explicit Evaluator(std::shared_ptr<Environment> env) : work_env_(std::move(env)) {}

  object::LoxObject Eval(std::shared_ptr<AstNode> node) {
    assert(IsValid(node));
    assert(active_map_);
    node->Accept(this);
    return PopRet();
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
  void Visit(LogicalExpr* state) override;
  void Visit(BinaryExpr* state) override;
  void Visit(GroupingExpr* state) override;
  void Visit(LiteralExpr* state) override;
  void Visit(UnaryExpr* state) override;
  void Visit(VariableExpr* state) override;
  void Visit(AssignExpr* state) override;
  void Visit(CallExpr* state) override;
  void Visit(GetAttrExpr* state) override;
  void Visit(SetAttrExpr* state) override;
  void Visit(PrintStmt* state) override;
  void Visit(ReturnStmt* state) override;
  void Visit(WhileStmt* state) override;
  void Visit(ExprStmt* state) override;
  void Visit(BreakStmt* state) override;
  void Visit(VarDeclStmt* state) override;
  void Visit(FunctionStmt* state) override;
  void Visit(ClassStmt* state) override;
  void Visit(BlockStmt* state) override;
  void Visit(IfStmt* state) override;
};

}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_EVALUATOR_EVAL_VISITOR_H_
