//
// LICENSE: MIT
//

#ifndef CLOX_SRCS_CLOX_CLOX_VALUE_H_
#define CLOX_SRCS_CLOX_CLOX_VALUE_H_

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "lox/backend/virtual_machine/common/buffer.h"
#include "lox/backend/virtual_machine/common/hash_map.h"
#include "lox/backend/virtual_machine/common/link_list.h"

namespace lox {
namespace vm {
enum class ValueType { NIL, NUMBER, BOOL, OBJ };
class Obj;
struct Value {
  Value() : type(ValueType::NIL), as{.number = 0} {};
  explicit Value(double number) : type(ValueType::NUMBER), as{.number = number} {}
  explicit Value(bool boolean) : type(ValueType::BOOL), as{.boolean = boolean} {}
  explicit Value(Obj* obj) : type(ValueType::OBJ), as{.obj = obj} {}
  bool AsBool() const {
    assert(IsBool());
    return as.boolean;
  };
  double AsNumber() const {
    assert(IsNumber());
    return as.number;
  };
  const Obj *AsObj() const {
    assert(IsObj());
    return as.obj;
  }
  Obj *AsObj() {
    assert(IsObj());
    return as.obj;
  }
  bool IsNil() const { return type == ValueType::NIL; }
  bool IsBool() const { return type == ValueType::BOOL; }
  bool IsNumber() const { return type == ValueType::NUMBER; }
  bool IsObj() const { return type == ValueType::OBJ; }
  ValueType Type() const { return type; }
  bool Equal(Value rhs);

  bool IsTrue() { return !IsNil() && IsBool() && AsBool(); }

 private:
  ValueType type;
  union {
    bool boolean;
    double number;
    Obj* obj;
  } as;
};
void printValue(const Value& value, bool print_to_debug = false);
enum class ObjType {
  UNKNOWN,
  OBJ_STRING,
  OBJ_FUNCTION,
  OBJ_UPVALUE,
  OBJ_RUNTIME_FUNCTION,
  OBJ_BOUND_METHOD,
  OBJ_NATIVE_FUNCTION,
  OBJ_CLASS,
  OBJ_INSTANCE,
};

struct Obj {
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
  bool Equal(const Obj* rhs) const;
  void Print(bool print_to_debug = false) const;
  static void Destroy(Obj* obj);

  static LinkList<Obj*>& AllCreatedObj();
  static int& ObjCount();

  static void MarkReference(Obj*);

 protected:
  explicit Obj(ObjType type);
  ~Obj();
};

template <ObjType TYPE>
struct ObjWithID : public Obj {
  constexpr static ObjType TYPE_ID = TYPE;

 protected:
  ObjWithID() : Obj(TYPE) {}
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

class Value;
struct ObjUpvalue : public ObjWithID<ObjType::OBJ_UPVALUE> {
  explicit ObjUpvalue(Value* location) : location(location) {}
  Value* location;
  Value closed;
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
  ObjBoundMethod(Value val, ObjRuntimeFunction* method) : receiver(val), method(method) {}
  Value receiver;
  ObjRuntimeFunction* method;
};
struct ObjNativeFunction : public ObjWithID<ObjType::OBJ_NATIVE_FUNCTION> {
  using NativeFn = Value (*)(int argCount, Value* args);
  explicit ObjNativeFunction(NativeFn fn) : function(fn) {}
  NativeFn function = nullptr;
};

struct ObjInternedString : public ObjWithID<ObjType::OBJ_STRING> {
  uint32_t hash;

  uint32_t static Hash(ObjInternedString* obj_str) { return obj_str->hash; }

  static ObjInternedString* Make(const char* data, int size);

  static ObjInternedString* Concat(const ObjInternedString* lhs, const ObjInternedString* rhs);

  Buffer<char> data;
  char* c_str() { return data.data(); }
  [[nodiscard]] const char* c_str() const { return data.data(); }
  int size() const;
  ~ObjInternedString();

 protected:
  struct InternView {
    const char* data;
    uint32_t hash;
    int size;
    bool operator==(const InternView& rhs) const {
      return size == rhs.size && hash == rhs.hash && strncmp(data, rhs.data, size) == 0;
      ;
    }
    bool operator!=(const InternView& rhs) const { return !(*this == rhs); }
  };
  InternView GetInternView() { return {.data = c_str(), .hash = hash, .size = size()}; }
  static uint32_t CachedStringHash(InternView string) { return string.hash; }

  using InternMap = HashMap<InternView, ObjInternedString*, CachedStringHash>;
  friend class GC;
  static InternMap& GetInternMap();
  uint32_t UpdateHash();
  ObjInternedString(const char* buf, int size);
  explicit ObjInternedString(Buffer<char>&& buffer);
  ObjInternedString() = delete;
  void TryInternThis();
};

struct ObjClass : public ObjWithID<ObjType::OBJ_CLASS> {
  explicit ObjClass(std::string name) : name(std::move(name)) {}
  std::string name;
  ObjClass* superclass = nullptr;
  std::unordered_map<ObjInternedString*, ObjRuntimeFunction*> methods;
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
  std::unordered_map<ObjInternedString*, Value> dict;
};

struct GC {
  // singleton
  using MarkerFn = void (*)(void*);
  struct Marker {
    MarkerFn marker_fn;
    void* marker_fn_arg;
    bool operator==(const Marker& rhs) const {
      return rhs.marker_fn_arg == marker_fn_arg && rhs.marker_fn == marker_fn;
    }
    bool operator!=(const Marker& rhs) const { return !(*this == rhs); }
  };
  GC(const GC&) = delete;
  GC(GC&&) = delete;
  GC& operator=(const GC&) = delete;
  GC& operator=(GC&&) = delete;

  static GC& Instance();
  void collectGarbage();
  void RegisterMarker(MarkerFn fn, void* arg) { markers.Insert(Marker{.marker_fn = fn, .marker_fn_arg = arg}); }
  void UnRegisterMarker(MarkerFn fn, void* arg) { markers.Delete(Marker{.marker_fn = fn, .marker_fn_arg = arg}); }
  void mark(Value value);
  void mark(Obj* object);
  void mark(HashMap<ObjInternedString*, Value, ObjInternedString::Hash>* table);

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
    auto node = markers.Head();
    while (node) {
      node->val.marker_fn(node->val.marker_fn_arg);
      node = node->next;
    }
  }
  void Sweep();
  LinkList<Marker> markers;
};

}  // namespace vm
}  // namespace lox
#endif  // CLOX_SRCS_CLOX_CLOX_VALUE_H_
