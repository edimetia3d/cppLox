//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_OBJECT_OBJECT_H
#define LOX_SRCS_LOX_OBJECT_OBJECT_H

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace lox {

class Object;
template <class T>
concept ConcreteObject = std::is_base_of<Object, T>::value;

struct GCSpObject {
  explicit GCSpObject(Object* obj) : obj_(obj){};
  Object* obj() { return obj_; }
  void ForceDelete();
  ~GCSpObject();

 private:
  Object* obj_;
};

using ObjectPtr = std::shared_ptr<GCSpObject>;
using ObjectWeakPtr = std::weak_ptr<GCSpObject>;

ObjectPtr NullObject();

struct Object {
  Object();

  Object(const Object&) = delete;
  Object(Object&&) = delete;

  /**
   * Make a object managed by the sweeper
   */
  template <ConcreteObject T, class... Args>
  static T* Make(Args&&... args) {
    return new T(std::forward<Args>(args)...);
  }

  /**
   * Make a object managed by shared_ptr, most gc will be handled by reference count, partial will be handled by
   * the sweeper.
   */
  template <ConcreteObject T, class... Args>
  static ObjectPtr MakeShared(Args&&... args) {
    Object* p_obj = new T(std::forward<Args>(args)...);
    auto ret = std::make_shared<GCSpObject>(p_obj);
    AllSharedPtrObj()[p_obj] = ret;
    return ret;
  }

  virtual ~Object();

  template <ConcreteObject T>
  T* DynAs() {
    return dynamic_cast<T*>(this);
  }

  template <ConcreteObject T>
  bool Is() {
    return dynamic_cast<T*>(this);
  }

  template <ConcreteObject T>
  const T* DynAs() const {
    return dynamic_cast<const T*>(this);
  }

  template <ConcreteObject T>
  T* As() {
    return static_cast<T*>(this);
  }

  template <ConcreteObject T>
  const T* As() const {
    return static_cast<const T*>(this);
  }

  virtual bool Equal(const Object* rhs) const { return rhs == this; }
  [[nodiscard]] virtual std::string Str() const = 0;

  [[nodiscard]] virtual bool IsTrue() const { return true; }  // all valid object are true by default

  /**
   * Get the objects that this object referenced directly , so that these object will not be garbage collected.
   * @return
   */
  virtual std::vector<Object*> References() = 0;

  static std::set<Object*>& AllCreatedObj();

  static std::map<Object*, ObjectWeakPtr>& AllSharedPtrObj();

 private:
  void* operator new(size_t size);
};
}  // namespace lox
#endif  // LOX_SRCS_LOX_OBJECT_OBJECT_H
