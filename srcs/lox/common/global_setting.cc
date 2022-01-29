//
// LICENSE: MIT
//

#include "lox/common/global_setting.h"
lox::LoxGlobalSetting& lox::GlobalSetting() {
  static LoxGlobalSetting v;
  return v;
}
