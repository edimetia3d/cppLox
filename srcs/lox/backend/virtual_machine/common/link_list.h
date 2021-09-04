//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_LINK_LIST_H_
#define LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_LINK_LIST_H_
namespace lox {
namespace vm {
template <class T>
struct LinkList {
  LinkList() = default;
  struct Node {
    Node(T v, Node *next = nullptr) : val(v), next(next) {}
    T val;
    Node *next = nullptr;
  };
  Node *Head() { return dummy_head.next; }
  void Insert(T v) {
    Node *tail = &dummy_head;
    while (tail->next) {
      tail = tail->next;
    }
    tail->next = new Node(v);
  }
  void Delete(T v) {
    Node *prev = &dummy_head;
    Node *cur = dummy_head.next;
    while (cur && cur->val != v) {
      prev = cur;
      cur = cur->next;
    }
    assert(cur->val == v);
    prev->next = cur->next;
    delete cur;
  }
  ~LinkList() {
    Node *p = Head();
    while (p) {
      auto next = p->next;
      delete p;
      p = next;
    }
  }

 private:
  Node dummy_head{T{}, nullptr};
};
}  // namespace vm
}  // namespace lox
#endif  // LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_LINK_LIST_H_
