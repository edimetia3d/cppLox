//
// LICENSE: MIT
//

#include "lox/lox.h"

#include <fstream>
#include <iostream>

#include "lox/ast/ast_printer.h"
#include "lox/ast/eval_visitor.h"
#include "lox/parser.h"
#include "lox/scanner.h"

namespace lox {
Lox::Lox() { evaluator_ = std::make_shared<AstEvaluator>(); }

std::string Lox::CLIHelpString() { return std::string(); }

Error Lox::RunFile(const std::string &file_path) {
  std::ifstream infile(file_path);
  return RunStream(&infile, false);
}

Error Lox::RunPrompt() { return RunStream(&std::cin, true); }

Error Lox::RunStream(std::istream *istream, bool interactive_mode) {
  Error err;
  if (interactive_mode) {
    std::string one_line;
    std::cout << ">> ";
    while (std::getline(*istream, one_line)) {
      if (one_line == "exit()") {
        break;
      }
      auto line_err = Eval(one_line);
      if (line_err.ToErrCode() > 0) {
        std::cout << line_err.Message() << std::endl;
      }
      std::cout << ">> ";
    }
  } else {
    std::string all_line((std::istreambuf_iterator<char>(*istream)), std::istreambuf_iterator<char>());
    err = Eval(all_line);
    if (err.ToErrCode() > 0) {
      std::cout << err.Message() << std::endl;
    }
  }

  return err;
}

Error Lox::Eval(const std::string &code) {
  Scanner scanner(code);
  auto err = scanner.Scan();

  Parser parser(scanner.Tokens());
  auto statements = parser.Parse();

  std::string output;
  try {
    for (auto &stmt : statements) {
      evaluator_->Eval(stmt);
    }
  } catch (RuntimeError &rt_err) {
    err.Append(rt_err.err);
  }
  return err;
}
}  // namespace lox
