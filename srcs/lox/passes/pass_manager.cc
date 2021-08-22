//
// License: MIT
//

#include "lox/passes/pass_manager.h"
void lox::PassManager::Run(std::vector<Stmt> stmts) {
  for (auto &pass : passes_) {
    runner_->SetPass(pass);
    for (auto &stmt : stmts) {
      auto new_stmt = stmt;
      do {
        new_stmt = runner_->RunPass(new_stmt);
      } while ((new_stmt != stmt));
      new_stmt->SetParent(stmt->Parent());
      stmt = new_stmt;
    }
  }
}
