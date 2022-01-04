//
// LICENSE: MIT
//

#include "lox/object/object.h"

namespace lox {
std::set<Object *> &Object::AllCreatedObj() {
  static std::set<Object *> ret;
  return ret;
}

Object::Object() { AllCreatedObj().insert(this); }

Object::~Object() { AllCreatedObj().erase(this); }

}  // namespace lox