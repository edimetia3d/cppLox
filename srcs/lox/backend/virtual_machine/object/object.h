//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_OBJECT_H
#define LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_OBJECT_H

#include <set>
#include <string>
#include <unordered_map>

#include "lox/object/object.h"
#include "lox/object/value.h"

namespace lox::vm {

class Chunk;
struct ObjFunction : public lox::Object {
  ObjFunction(std::string name);
  int arity = 0;
  std::string name;
  std::unique_ptr<Chunk> chunk;
  [[nodiscard]] std::string Str() const override;

  std::vector<Object*> References() override;
};

struct ObjUpvalue : public lox::Object {
  explicit ObjUpvalue(Value* location) : location(location) {}
  Value* location;
  Value closed;
  ObjUpvalue* next = nullptr;
  [[nodiscard]] std::string Str() const override;
  std::vector<Object*> References() override;
};

/**
 * Every function created by compiler will become a closure at runtime.
 * When no closed variable is used, runtime function is basically same as compile time function
 */
struct ObjClosure : public lox::Object {
  ObjClosure(ObjFunction* func);
  bool isClosure() const { return upvalues.size() > 0; }
  ObjFunction* function;
  std::vector<ObjUpvalue*> upvalues;
  [[nodiscard]] std::string Str() const override;
  std::vector<Object*> References() override;
};

struct ObjNativeFunction : public lox::Object {
  using NativeFn = Value (*)(int argCount, Value* args);
  explicit ObjNativeFunction(NativeFn fn);
  NativeFn function = nullptr;
  [[nodiscard]] std::string Str() const override;
  std::vector<Object*> References() override;
};

struct Symbol : public lox::Object {
  explicit Symbol(std::string d);

  static Symbol* Intern(const std::string& str);

  operator const std::string&() const { return data; }

  const char* c_str() const { return data.c_str(); }

  [[nodiscard]] std::string Str() const override { return data; }
  std::vector<Object*> References() override;

 protected:
  static std::unordered_map<std::string, Symbol*>& GetInternMap();
  std::string data;
};

struct ObjClass : public lox::Object {
  explicit ObjClass(std::string name);
  std::string name;
  ObjClass* superclass = nullptr;
  std::unordered_map<Symbol*, ObjClosure*> methods;

  [[nodiscard]] std::string Str() const override;
  std::vector<Object*> References() override;
};

struct ObjInstance : public lox::Object {
  explicit ObjInstance(ObjClass* klass)
      : klass(klass), dict_data(std::make_shared<std::unordered_map<Symbol*, Value>>()){};
  ObjClass* klass;
  bool IsInstance(ObjClass* target) const;
  [[nodiscard]] std::string Str() const override;
  std::unordered_map<Symbol*, Value>& dict() { return *dict_data; }
  std::vector<Object*> References() override;
  ObjInstance* Cast(ObjClass* target);

 private:
  std::shared_ptr<std::unordered_map<Symbol*, Value>> dict_data;
  bool is_cast = false;
};

struct ObjBoundMethod : public lox::Object {
  ObjBoundMethod(ObjInstance* this_instance, ObjClosure* method);
  ObjInstance* bounded_this;  // the bounded `this`
  ObjClosure* method;
  [[nodiscard]] std::string Str() const override;
  std::vector<Object*> References() override;
};

}  // namespace lox::vm

#endif  // LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_OBJECT_H
