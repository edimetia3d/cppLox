//
// License: MIT
//

#include "lox/backend/tree_walker/passes/pass_manager.h"
std::vector<lox::Stmt> lox::PassManager::Run(std::vector<Stmt> stmts) {
  for (auto &pass : passes_) {
    runner_->SetPass(pass);
    for (auto &stmt : stmts) {
      auto new_stmt = runner_->RunPass(stmt);
      stmt = new_stmt;
    }
  }
  return stmts;
}
