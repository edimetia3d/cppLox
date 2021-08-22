//
// License: MIT
//

#include "lox/passes/pass_manager.h"
std::vector<lox::Stmt> lox::PassManager::Run(std::vector<Stmt> stmts) {
  for (auto &pass : passes_) {
    runner_->SetPass(pass);
    for (auto &stmt : stmts) {
      auto old_stmt = stmt;
      auto new_stmt = stmt;
      do {
        old_stmt = new_stmt;
        ResetModify(old_stmt);
        new_stmt = runner_->RunPass(old_stmt);
      } while ((new_stmt != old_stmt));
      new_stmt->SetParent(stmt->Parent());
      stmt = new_stmt;
    }
  }
  return stmts;
}
