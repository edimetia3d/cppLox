//
// LICENSE: MIT
//

#include "lox/global_setting.h"
lox::LoxGlobalSetting& lox::GlobalSetting() {
  static LoxGlobalSetting v;
  return v;
}
