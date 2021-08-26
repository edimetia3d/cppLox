//
// LICENSE: MIT
//

#include "global_setting.h"
lox::_GlobalSetting& lox::GlobalSetting() {
  static _GlobalSetting v;
  return v;
}
