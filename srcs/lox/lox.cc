//
// LICENSE: MIT
//

#include "lox/lox.h"

#include <fstream>
#include <iostream>

#include "lox/backend/backend.h"
#include "lox/frontend/scanner.h"
#include "lox/global_setting.h"

namespace lox {

std::string LoxInterpreter::CLIHelpString() { return std::string(); }

LoxError LoxInterpreter::RunFile(const std::string &file_path) {
  std::ifstream infile(file_path);
  GlobalSetting().interactive_mode = false;
  return RunStream(&infile);
}

LoxError LoxInterpreter::RunPrompt() {
  GlobalSetting().interactive_mode = true;
  return RunStream(&std::cin);
}

LoxError LoxInterpreter::RunStream(std::istream *istream) {
  LoxError err;
  if (GlobalSetting().interactive_mode) {
    std::string one_line;
    std::cout << ">> ";
    while (std::getline(*istream, one_line)) {
      if (one_line == "exit()") {
        break;
      }
      err.Merge(Eval(one_line));
      std::cout << ">> ";
    }
  } else {
    std::string all_line((std::istreambuf_iterator<char>(*istream)), std::istreambuf_iterator<char>());
    err = Eval(all_line);
  }

  return err;
}

LoxError LoxInterpreter::Eval(const std::string &code) {
  Scanner scanner(code);
  return back_end_->Run(scanner);
}
LoxInterpreter::LoxInterpreter(const std::string &backend_name) {
  back_end_ = BackEnd::CreateBackEnd(GlobalSetting().default_backend);
}
}  // namespace lox
