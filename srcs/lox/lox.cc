//
// LICENSE: MIT
//

#include "lox/lox.h"

#include <fstream>
#include <iostream>

#include "lox/backend/backend.h"
#include "lox/common/global_setting.h"
#include "lox/frontend/scanner.h"

namespace lox {

std::string LoxInterpreter::CLIHelpString() { return std::string(); }

void LoxInterpreter::RunFile(const std::string &file_path) {
  std::ifstream infile(file_path);
  GlobalSetting().interactive_mode = false;
  auto file = std::make_shared<InputFile>(file_path);
  Eval(file);
}

void LoxInterpreter::RunPrompt() {
  GlobalSetting().interactive_mode = true;
  std::string one_line;
  std::cout << ">> ";
  while (std::getline(std::cin, one_line)) {
    if (one_line == "exit()") {
      break;
    }
    auto line_code = std::make_shared<InputString>(one_line.c_str(), one_line.size());
    try {
      Eval(line_code);
    } catch (const LoxError &e) {
      std::cout << e.what() << std::endl;
    }
    fflush(stdout);
    std::cout << ">> ";
  }
}

void LoxInterpreter::Eval(std::shared_ptr<CharStream> input) {
  Scanner scanner(input);
  return back_end_->Run(scanner);
}
LoxInterpreter::LoxInterpreter() { back_end_ = BackEndRegistry::Instance().Get(GlobalSetting().backend)(); }
} // namespace lox
