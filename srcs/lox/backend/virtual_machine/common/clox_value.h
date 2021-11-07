//
// LICENSE: MIT
//

#ifndef CLOX_SRCS_CLOX_CLOX_VALUE_H_
#define CLOX_SRCS_CLOX_CLOX_VALUE_H_

#include <cassert>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace lox {
namespace vm {
enum class ObjectType { NIL, NUMBER, BOOL, OBJ_HANDLE };
enum class ObjType {
  UNKNOWN,
  OBJ_SYMBOL,
  OBJ_FUNCTION,
  OBJ_UPVALUE,
  OBJ_RUNTIME_FUNCTION,
  OBJ_BOUND_METHOD,
  OBJ_NATIVE_FUNCTION,
  OBJ_CLASS,
  OBJ_INSTANCE,
};

struct ObjHandle {
  ObjType type;
  bool isMarked = false;  // gc mark

  template <class T>
  const T* const As() const {
    assert(type == T::TYPE_ID);
    return reinterpret_cast<const T*>(this);
  }
  template <class T>
  T* As() {
    assert(type == T::TYPE_ID);
    return reinterpret_cast<T*>(this);
  }
  template <class T>
  bool IsType() const {
    return type == T::TYPE_ID;
  }
  bool Equal(const ObjHandle* rhs) const;
  void Print(bool print_to_debug = false) const;
  static void Destroy(ObjHandle* obj);

  static std::set<ObjHandle*>& AllCreatedObj();
  static int& ObjCount();

  static void MarkReference(ObjHandle*);

 protected:
  explicit ObjHandle(ObjType type);
  ~ObjHandle();
};

struct Object {
  Object() : type(ObjectType::NIL), as{.number = 0} {};
  explicit Object(double number) : type(ObjectType::NUMBER), as{.number = number} {}
  explicit Object(bool boolean) : type(ObjectType::BOOL), as{.boolean = boolean} {}
  explicit Object(ObjHandle* handle) : type(ObjectType::OBJ_HANDLE), as{.handle = handle} {}
  bool AsBool() const {
    assert(IsBool());
    return as.boolean;
  };
  double AsNumber() const {
    assert(IsNumber());
    return as.number;
  };
  const ObjHandle* AsHandle() const {
    assert(IsHandle());
    return as.handle;
  }
  ObjHandle* AsHandle() {
    assert(IsHandle());
    return as.handle;
  }
  bool IsNil() const { return type == ObjectType::NIL; }
  bool IsBool() const { return type == ObjectType::BOOL; }
  bool IsNumber() const { return type == ObjectType::NUMBER; }
  bool IsHandle() const { return type == ObjectType::OBJ_HANDLE; }
  ObjectType Type() const { return type; }
  bool Equal(Object rhs);

  bool IsTrue() { return !IsNil() && IsBool() && AsBool(); }

 private:
  ObjectType type;
  union {
    bool boolean;
    double number;
    ObjHandle* handle;
  } as;
};

template <ObjType TYPE>
struct ObjWithID : public ObjHandle {
  constexpr static ObjType TYPE_ID = TYPE;

 protected:
  ObjWithID() : ObjHandle(TYPE) {}
};
class Chunk;
struct ObjFunction : public ObjWithID<ObjType::OBJ_FUNCTION> {
  ObjFunction();
  int arity = 0;
  std::string name;
  Chunk* chunk;
  int upvalueCount = 0;
  ~ObjFunction();
};

struct ObjUpvalue : public ObjWithID<ObjType::OBJ_UPVALUE> {
  explicit ObjUpvalue(Object* location) : location(location) {}
  Object* location;
  Object closed;
  struct ObjUpvalue* next = nullptr;
};

/**
 * Runtime function is used for supporting of closure, when no closure is used
 * runtime function is same as compile time function
 */
struct ObjRuntimeFunction : public ObjWithID<ObjType::OBJ_RUNTIME_FUNCTION> {
  using ObjUpvaluePtr = ObjUpvalue*;
  ObjRuntimeFunction(ObjFunction* func) : function(func) {
    upvalueCount = func->upvalueCount;
    upvalues = new ObjUpvaluePtr[upvalueCount]{};
  }
  bool isClosure() const { return function->upvalueCount > 0; }
  ObjFunction* function;
  ObjUpvaluePtr* upvalues;
  int upvalueCount;
  ~ObjRuntimeFunction() { delete[] upvalues; }
};

struct ObjBoundMethod : public ObjWithID<ObjType::OBJ_BOUND_METHOD> {
  ObjBoundMethod(Object val, ObjRuntimeFunction* method) : receiver(val), method(method) {}
  Object receiver;
  ObjRuntimeFunction* method;
};
struct ObjNativeFunction : public ObjWithID<ObjType::OBJ_NATIVE_FUNCTION> {
  using NativeFn = Object (*)(int argCount, Object* args);
  explicit ObjNativeFunction(NativeFn fn) : function(fn) {}
  NativeFn function = nullptr;
};

struct Symbol : public ObjWithID<ObjType::OBJ_SYMBOL> {
  static Symbol* Intern(const std::string& str) {
    if (!GetInternMap().contains(str)) {
      GetInternMap()[str] = new Symbol(str);
    }
    return GetInternMap()[str];
  }

  operator const std::string&() const { return data; }

  const char* c_str() const { return data.c_str(); }

  const std::string& str() const { return data; }

 protected:
  explicit Symbol(std::string d) : data(std::move(d)) {}
  static std::unordered_map<std::string, Symbol*>& GetInternMap();
  std::string data;
};

struct ObjClass : public ObjWithID<ObjType::OBJ_CLASS> {
  explicit ObjClass(std::string name) : name(std::move(name)) {}
  std::string name;
  ObjClass* superclass = nullptr;
  std::unordered_map<Symbol*, ObjRuntimeFunction*> methods;
};

struct ObjInstance : public ObjWithID<ObjType::OBJ_INSTANCE> {
  explicit ObjInstance(ObjClass* klass) : klass(klass){};
  ObjClass* klass;
  bool IsInstance(ObjClass* target) {
    auto check = klass;
    while (check && check != target) {
      check = check->superclass;
    }
    return check == target;
  }
  std::unordered_map<Symbol*, Object> dict;
};

struct GC {
  // singleton
  using MarkerFn = void (*)(void*);
  struct Marker {
    MarkerFn marker_fn;
    void* marker_fn_arg;
    bool operator<(const Marker& rhs) const { return rhs.marker_fn_arg < marker_fn_arg && rhs.marker_fn < marker_fn; }
  };
  GC(const GC&) = delete;
  GC(GC&&) = delete;
  GC& operator=(const GC&) = delete;
  GC& operator=(GC&&) = delete;

  static GC& Instance();
  void collectGarbage();
  void RegisterMarker(MarkerFn fn, void* arg) { markers.insert(Marker{.marker_fn = fn, .marker_fn_arg = arg}); }
  void UnRegisterMarker(MarkerFn fn, void* arg) { markers.erase(Marker{.marker_fn = fn, .marker_fn_arg = arg}); }
  void mark(Object value);
  void mark(ObjHandle* object);
  template <class KeyT, class ValueT>
  void mark(const std::unordered_map<KeyT*, ValueT>& map) {
    for (const auto& pair : map) {
      mark(pair.first);
      mark(pair.second);
    }
  }

  struct RegisterMarkerGuard {
    RegisterMarkerGuard(MarkerFn fn, void* arg) : marker{fn, arg} {
      GC::Instance().RegisterMarker(marker.marker_fn, marker.marker_fn_arg);
    }
    ~RegisterMarkerGuard() { GC::Instance().UnRegisterMarker(marker.marker_fn, marker.marker_fn_arg); }
    Marker marker;
  };

  int gc_threashold = 1024;

 private:
  GC() = default;
  void markRoots() {
    auto node = markers.begin();
    while (node != markers.end()) {
      node->marker_fn(node->marker_fn_arg);
      ++node;
    }
  }
  void Sweep();
  std::set<Marker> markers;
};

}  // namespace vm
void printValue(const vm::Object& value, bool print_to_debug = false);
}  // namespace lox
#endif  // CLOX_SRCS_CLOX_CLOX_VALUE_H_
