//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_GLOBAL_SETTING_H_
#define CPPLOX_SRCS_LOX_GLOBAL_SETTING_H_
#include <string>

namespace lox {
struct _GlobalSetting {
  bool interactive_mode = true;
  bool debug = false;
  bool single_step_mode = false;
  std::string backend = "VirtualMachine";
};

_GlobalSetting& GlobalSetting();
}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_GLOBAL_SETTING_H_
