#include <stdio.h>
#include <iostream>
#include <functional>



template <typename T>
struct function_traits
    : public function_traits<decltype(&T::operator())>
{};
// For generic types, directly use the result of the signature of its 'operator()'

template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType(ClassType::*)(Args...) const>
// we specialize for pointers to member function
{
    enum { arity = sizeof...(Args) };
    // arity is the number of arguments.

    typedef ReturnType result_type;

    typedef std::tuple<Args...> args;


    template <size_t i>
    struct arg
    {
        typedef typename std::tuple_element<i, args>::type type;
        // the i-th argument is equivalent to the i-th tuple element of a tuple
        // composed of those arguments.
    };
};

template<typename T>
static void print_type() {
  std::cout << "print_type " << typeid(T).name() << "\n";
}

template<typename func_t>
void invoker(func_t f) {
  typedef function_traits<func_t> traits;
  using tmp = typename traits::template arg<0>::type;
  print_type<tmp>();

  // print_type<t(typename traits::arg<0>)::type>();
  // std::cout << typeid(foo).name() << "\n";
  // static_assert(std::is_same<int, traits::arg<0>::type>::value, "err");

}

int main(int argc, char* argv[]) {
  invoker([](int a, float b) {
    return a + b;
  });
}
