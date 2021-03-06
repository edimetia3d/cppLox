

#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <iostream>

#include "lox/common/global_setting.h"
#include "lox/common/lox_error.h"
#include "lox/lox.h"
#include "lox/object/gc.h"
#include "lox/version/version.h"

using namespace lox;

void PrintBuildInfo() {
  auto str = std::string("Build info:\nSHA1: ") + std::string(version::GitSha1()) +
             "\nDate: " + std::string(version::GitDate());
  std::cout << str << std::endl;
}

struct CLIArgs {
  std::string input_file;
  bool just_man = false;
  int log_level = spdlog::level::n_levels;  // means not set
};
void ArgsDef(CLI::App& app, CLIArgs& args) {
  app.add_option("script_file", args.input_file, "Input file path, if not provided, interactive mode will be used.");
  app.add_option(
      "--backend", GlobalSetting().backend,
      "Specify the backend, "
      "could be one of{\"TreeWalker\",\"VirtualMachine\", \"MLIRJIT\",\"LLVMJIT\"}, default is \"VirtualMachine\"");
  app.add_option("--parser", GlobalSetting().parser,
                 "Specify the frontend parser, VirtualMachine backend will ignore this option."
                 "could be one of{\"RecursiveDescent\",\"PrattParser\"}, default is \"PrattParser\"");
  app.add_option("--man", args.just_man, "Print user manual and return");
  app.add_option("--log_level", args.log_level, "Set the log level");
  app.add_option("--mlir_cli_options", GlobalSetting().mlir_cli_options,
                 "When using MLIRJIT backend, this option is used to pass the options to mlir-cli. Note that to use "
                 "the quote mark when passing multiple options.");
  app.add_option("--opt", GlobalSetting().opt_level, "Set the optimization level, could be one of {0,1}, default is 1");
  app.add_flag("--debug", GlobalSetting().debug, "Enable debug mode");
}

int main(int argn, const char* argv[]) {
  CLI::App app{"Lox Interpreter"};
  CLIArgs args;
  ArgsDef(app, args);
  CLI11_PARSE(app, argn, argv);
  if (args.just_man) {
    PrintBuildInfo();
    std::cout << LoxInterpreter::CLIHelpString() << std::endl;
    return 0;
  }
  if (args.log_level != spdlog::level::n_levels) {
    spdlog::set_level(static_cast<spdlog::level::level_enum>(args.log_level));
  } else {
#ifndef NDEBUG
    spdlog::set_level(spdlog::level::debug);
#else
    spdlog::set_level(spdlog::level::err);
#endif
  }
  {
    // use a scope to kill interpreter
    LoxInterpreter interpreter;
    try {
      if (!args.input_file.empty()) {
        interpreter.RunFile(args.input_file);
      } else {
        interpreter.RunPrompt();
      }
    } catch (const lox::LoxError& e) {
      std::cerr << e.what() << std::endl;
      return e.exit_code;
    }
  }
  GC::Instance().ForceClearAll();
  return 0;
}
