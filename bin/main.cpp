

#include <iostream>

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>

#include "lox/global_setting.h"
#include "lox/lox.h"
#include "lox/version.h"
using namespace lox;

void PrintBuildInfo() {
  auto str = std::string("Build info:\nSHA1: ") + std::string(version::GitSha1()) +
             "\nDate: " + std::string(version::GitDate());
  std::cout << str << std::endl;
}

struct CLIArgs {
  std::string input_file;
  bool just_man = false;
};
void ArgsDef(CLI::App& app, CLIArgs& args) {
  app.add_option("script_file", args.input_file, "Input file path, if not provided, interactive mode will be used.");
  app.add_option("--backend", GlobalSetting().backend,
                 "Specify the backend, "
                 "could be one of{\"TreeWalker\",\"VirtualMachine\"}, default is \"VirtualMachine\"");

  app.add_option("--man", args.just_man, "Print user manual and return");
}

int main(int argn, char* argv[]) {
#ifndef NDEBUG
  spdlog::set_level(spdlog::level::debug);
#endif
  CLI::App app{"Lox Interpreter"};
  CLIArgs args;
  ArgsDef(app, args);
  CLI11_PARSE(app, argn, argv);

  LoxError ret;
  LoxInterpreter interpreter;

  if (args.just_man) {
    PrintBuildInfo();
    std::cout << LoxInterpreter::CLIHelpString() << std::endl;
    return 0;
  }
  if (!args.input_file.empty()) {
    ret = interpreter.RunFile(args.input_file);
  } else {
    ret = interpreter.RunPrompt();
  }
  return ret.ToErrCode();
}
