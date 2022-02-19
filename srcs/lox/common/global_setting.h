//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_GLOBAL_SETTING_H_
#define CPPLOX_SRCS_LOX_GLOBAL_SETTING_H_
#include <string>
enum class RuntimeDumpFrequency { NONE, EVERY_INSTRUCTION, EVERY_LINE, EVERY_FUNCTION };
namespace lox {
struct LoxGlobalSetting {
  bool interactive_mode = true;
  bool debug = false;
  bool single_step_mode = false;
  RuntimeDumpFrequency runtime_dump_frequency = RuntimeDumpFrequency::EVERY_LINE;
  std::string backend = "VirtualMachine";
  std::string parser = "PrattParser";
  std::string mlir_cli_options = "";
};

LoxGlobalSetting& GlobalSetting();
}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_GLOBAL_SETTING_H_
