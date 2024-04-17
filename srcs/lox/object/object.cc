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

Object::~Object() {
  if (AllSharedPtrObj().contains(this)) {
    AllSharedPtrObj().erase(this);
  }
  AllCreatedObj().erase(this);
}
void *Object::operator new(size_t size) { return ::operator new(size); }
std::map<Object *, ObjectWeakPtr> &Object::AllSharedPtrObj() {
  static std::map<Object *, ObjectWeakPtr> ret;
  return ret;
}

GCSpObject::~GCSpObject() {
  if (obj_) {
    delete obj_;
  }
}
void GCSpObject::ForceDelete() {
  delete obj_;
  obj_ = nullptr;
}
ObjectPtr NullObject() { return ObjectPtr(); }
} // namespace lox