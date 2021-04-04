#include <iostream>

#include "lox/lox.h"

using namespace lox;

int main(int argn, char* argv[]) {
  if (argn > 2) {
    std::cout << Lox::CLIHelpString() << std::endl;
  } else if (argn == 2) {
    Lox::RunFile(argv[1]);
  } else {
    Lox::RunPrompt();
  }
}