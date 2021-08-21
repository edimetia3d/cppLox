//
// LICENSE: MIT
//

#include "lox/lox.h"

#include <fstream>
#include <iostream>

#include "lox/global_setting/global_setting.h"
#include "lox/parser.h"
#include "lox/passes/env_resolve_pass/env_reslove_pass.h"
#include "lox/passes/env_resolve_pass/resolve_map.h"
#include "lox/passes/pass_manager.h"
#include "lox/passes/semantic_check/semantic_check.h"
#include "lox/scanner.h"
#include "lox/visitors/ast_printer/ast_printer.h"
#include "lox/visitors/evaluator/callable_object.h"
#include "lox/visitors/evaluator/environment.h"
#include "lox/visitors/evaluator/evaluator.h"
namespace lox {
static bool g_debug = true;
Lox::Lox() {
  global_env_ = Environment::Make();
  for (auto it : BuiltinCallables()) {
    global_env_->Define(it.first, it.second);
  }
  resolve_map_ = std::make_shared<EnvResolveMap>();
  evaluator_ = std::make_shared<Evaluator>(global_env_);
  evaluator_->SetActiveResolveMap(resolve_map_);
}

std::string Lox::CLIHelpString() { return std::string(); }

Error Lox::RunFile(const std::string &file_path) {
  std::ifstream infile(file_path);
  GlobalSetting().interactive_mode = false;
  return RunStream(&infile);
}

Error Lox::RunPrompt() {
  GlobalSetting().interactive_mode = true;
  return RunStream(&std::cin);
}

Error Lox::RunStream(std::istream *istream) {
  Error err;
  if (GlobalSetting().interactive_mode) {
    std::string one_line;
    std::cout << ">> ";
    while (std::getline(*istream, one_line)) {
      if (one_line == "exit()") {
        break;
      }
      Eval(one_line);
      std::cout << ">> ";
    }
  } else {
    std::string all_line((std::istreambuf_iterator<char>(*istream)), std::istreambuf_iterator<char>());
    Eval(all_line);
  }

  return err;
}

void Lox::Eval(const std::string &code) {
  Scanner scanner(code);
  auto err = scanner.Scan();

  Parser parser(scanner.Tokens());
  auto statements = parser.Parse();

  PassManager pass_mgr;
  pass_mgr.Append(std::make_shared<SemanticCheck>());
  pass_mgr.Append(std::make_shared<EnvResovlePass>(resolve_map_));
  try {
    pass_mgr.Run(statements);
  } catch (std::exception &err) {
    std::cout << err.what() << std::endl;
    return;
  }

  try {
    for (auto &stmt : statements) {
      if (g_debug) {
        static AstPrinter printer;
        std::cout << "Debug Stmt: `" << printer.Print(stmt) << "`" << std::endl;
      }
      evaluator_->Eval(stmt);
    }
  } catch (RuntimeError &rt_err) {
    std::cout << rt_err.what() << std::endl;
    return;
  }
}
}  // namespace lox
