//
// LICENSE: MIT
//

#include <spdlog/spdlog.h>

#include "lox/backend/virtual_machine/object/object.h"

#include "lox/backend/virtual_machine/core/chunk.h"

template <class... T>
static inline std::string string_sprint(const char *format, T... args) {
  char buf[100];
  std::snprintf(buf, 100, format, args...);
  return buf;
}

namespace lox::vm {
static inline void InjectValueToObjVec(std::vector<Object *> &vec, Value value) {
  if (value.IsObject()) {
  }
  vec.push_back(value.AsObject());
}

ObjFunction::ObjFunction(std::string name) : name(name) { chunk.reset(new Chunk()); }
std::vector<Object *> ObjFunction::References() {
  int n = chunk->constants.size();
  auto p = chunk->constants.data();
  std::vector<Object *> ret;
  for (int i = 0; i < n; ++i) {
    InjectValueToObjVec(ret, p[i]);
  }
  return ret;
}
std::string ObjFunction::Str() const { return string_sprint("<fn %s>", name.c_str()); }

std::unordered_map<std::string, Symbol *> &Symbol::GetInternMap() {
  static std::unordered_map<std::string, Symbol *> map;
  return map;
}
std::vector<Object *> Symbol::References() { return {}; }
Symbol *Symbol::Intern(const std::string &str) {
  if (!GetInternMap().contains(str)) {
    GetInternMap()[str] = Object::Make<Symbol>(str);
  }
  return GetInternMap()[str];
}
Symbol::Symbol(std::string d) : data(std::move(d)) {}
std::vector<Object *> ObjUpvalue::References() { return {location->AsObject()}; }
std::string ObjUpvalue::Str() const { return "upvalue"; }
std::vector<Object *> ObjClosure::References() {
  std::vector<Object *> ret = {function};
  for (auto &upvalue : upvalues) {
    ret.push_back(static_cast<Object *>(upvalue));
  }
  return ret;
}
ObjClosure::ObjClosure(ObjFunction *func) : function(func) {}
std::string ObjClosure::Str() const {
  if (isClosure()) {
    return string_sprint("<closure %s>", function->name.c_str());
  } else {
    return string_sprint("<fn %s>", function->name.c_str());
  }
}
std::vector<Object *> ObjBoundMethod::References() { return {bounded_this, method}; }
ObjBoundMethod::ObjBoundMethod(ObjInstance *this_instance, ObjClosure *method)
    : bounded_this(this_instance), method(method) {}
std::string ObjBoundMethod::Str() const { return string_sprint("<fn %s>", method->function->name.c_str()); }
std::vector<Object *> ObjNativeFunction::References() { return {}; }
ObjNativeFunction::ObjNativeFunction(ObjNativeFunction::NativeFn fn) : function(fn) {}
std::string ObjNativeFunction::Str() const { return "<native fn>"; }
std::vector<Object *> ObjClass::References() {
  std::vector<Object *> ret;
  for (auto pair : methods) {
    ret.push_back(pair.second);
    ret.push_back(pair.first);
  }
  if (superclass) {
    ret.push_back(superclass);
  }
  return ret;
}
std::string ObjClass::Str() const { return name; }
ObjClass::ObjClass(std::string name) : name(std::move(name)) {}
std::vector<Object *> ObjInstance::References() {
  if (is_cast) {
    return {};
  }
  std::vector<Object *> ret;
  for (auto pair : dict()) {
    InjectValueToObjVec(ret, pair.second);
    ret.push_back(pair.first);
  }
  ret.push_back(klass);
  return ret;
}
std::string ObjInstance::Str() const { return string_sprint("%s instance", klass->name.c_str()); }
bool ObjInstance::IsInstance(ObjClass *target) const {
  auto check = klass;
  while (check && check != target) {
    check = check->superclass;
  }
  return check == target;
}
ObjInstance *ObjInstance::Cast(ObjClass *target) {
  if (klass == target) return this;
  if (IsInstance(target)) {
    auto ret = Object::Make<ObjInstance>(target);
    ret->dict_data = dict_data;
    ret->is_cast = true;
    return ret;
  }
  return nullptr;
}

}  // namespace lox::vm