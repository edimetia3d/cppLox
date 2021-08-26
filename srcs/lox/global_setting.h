//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_GLOBAL_SETTING_H_
#define CPPLOX_SRCS_LOX_GLOBAL_SETTING_H_

namespace lox {
struct _GlobalSetting {
  bool interactive_mode = true;
  bool debug = true;
};

_GlobalSetting& GlobalSetting();
}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_GLOBAL_SETTING_H_
