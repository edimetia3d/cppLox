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

void LoxInterpreter::RunFile(const std::string &file_path) {
  std::ifstream infile(file_path);
  GlobalSetting().interactive_mode = false;
  std::string all_line((std::istreambuf_iterator<char>(infile)), std::istreambuf_iterator<char>());
  Eval(all_line, file_path);
}

void LoxInterpreter::RunPrompt() {
  GlobalSetting().interactive_mode = true;
  std::string one_line;
  std::cout << ">> ";
  while (std::getline(std::cin, one_line)) {
    if (one_line == "exit()") {
      break;
    }
    Eval(one_line, "stdin");
    fflush(stdout);
    std::cout << ">> ";
  }
}

void LoxInterpreter::Eval(const std::string &code, const std::string &file_name) {
  Scanner scanner(code, file_name);
  return back_end_->Run(scanner);
}
LoxInterpreter::LoxInterpreter() { back_end_ = BackEndRegistry::Instance().Get(GlobalSetting().backend)(); }
}  // namespace lox
