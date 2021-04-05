
#include <iostream>

#include "lox/lox.h"
#include "lox/version.h"

using namespace lox;

struct BuildInfoPrinter {
  BuildInfoPrinter() {
    auto str = std::string("Build info:\nSHA1: ") +
               std::string(version::GitSha1()) +
               "\nDate: " + std::string(version::GitDate());
    std::cout << str << std::endl;
  }
};

static BuildInfoPrinter g_lox_build_info_inited;

int main(int argn, char* argv[]) {
  Error ret;
  Lox interpreter;
  if (argn > 2) {
    std::cout << Lox::CLIHelpString() << std::endl;
  } else if (argn == 2) {
    ret = interpreter.RunFile(argv[1]);
  } else {
    ret = interpreter.RunPrompt();
  }
  return ret.ToErrCode();
}