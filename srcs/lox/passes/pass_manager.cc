//
// Created by edi on 8/21/21.
//

#include "lox/passes/pass_manager.h"
void lox::PassManager::Run(std::vector<Stmt> stmts) {
  for (auto &pass : passes_) {
    runner_->SetPass(pass);
    for (auto &stmt : stmts) {
      runner_->RunPass(stmt);
    }
  }
}
