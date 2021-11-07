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

#include "lox/backend/virtual_machine/common/buffer.h"
#include "lox/backend/virtual_machine/common/hash_map.h"

namespace lox {
namespace vm {
enum class ObjectType { NIL, NUMBER, BOOL, OBJ_HANDLE };
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
  explicit Object(ObjHandle* obj) : type(ObjectType::OBJ_HANDLE), as{.obj = obj} {}
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
    return as.obj;
  }
  ObjHandle* AsHandle() {
    assert(IsHandle());
    return as.obj;
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
    ObjHandle* obj;
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
  std::unordered_map<ObjInternedString*, Object> dict;
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
  void mark(HashMap<ObjInternedString*, Object, ObjInternedString::Hash>* table);

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
