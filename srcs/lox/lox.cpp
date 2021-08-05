//
// LICENSE: MIT
//

#include "lox/lox.h"

#include <fstream>
#include <iostream>

#include "lox/ast/ast_printer.h"
#include "lox/parser.h"
#include "lox/scanner.h"

namespace lox {
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
      std::string oneline_output;
      auto line_err = Eval(one_line, &oneline_output);
      if (line_err.ToErrCode() > 0) {
        std::cout << line_err.Message() << std::endl;
      } else {
        std::cout << oneline_output << std::endl;
      }
      std::cout << ">> ";
    }
  } else {
    std::string all_line((std::istreambuf_iterator<char>(*istream)), std::istreambuf_iterator<char>());
    err = Eval(all_line, nullptr);
    if (err.ToErrCode() > 0) {
      std::cout << err.Message() << std::endl;
    }
  }

  return err;
}

Error Lox::Eval(const std::string &code, std::string *eval_output) {
  Scanner scanner(code);
  auto err = scanner.Scan();

  Parser parser(scanner.Tokens());
  auto expr = parser.Parse();

  std::string output;
  if (expr) {
    AstPrinter printer;
    output = printer.Print(expr) + "\n";
    try {
      auto val = evaluator_.Eval(expr);
      output += val.ToString();
    } catch (RuntimeError &rt_err) {
      err.Append(rt_err.err);
    }
  }

  if (eval_output) {
    *eval_output = std::move(output);
  }
  return err;
}
}  // namespace lox
