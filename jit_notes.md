# Intro
We will have two JIT backend, one is LLVM based, another one is MLIR based, I decided to do this just want to
figure out how much power the Multi-Level IR can give me, comparing to the classical single level LLVM IR.

Generally speaking, lox will be a dialect of C language when using JIT, that is , any
feature that is not supported by C natively will not be supported by lox neither.

Someday in the future, lox might be a dialect of C++ language, but for now, I just want to eat some low hanging fruit.

To make JIT backend work more like a interpreter, that is, support REPL mode and global level stmts evaluation,
JIT will make all stmts that do not define var nor function in the global scope
compile into an anonymous function and then execute it after compilation.
After the module has been compiled and executed, the newly defined func/var will be kept, while the anonymous function
that used to execute other global stmts will be discarded.

e.g.

The code following

```
var a:float = 8;
fun float f(){
    return a*2;
}
a = f();
a = a + 8;
```

Will be transform to a code like

```
var a:float = 8;
fun float f(){
    return a*2;
}
fun __anonymous_0__(){
    a = f();
}
fun __anonymous_1__(){
    a = a + 8;
}

```

The `__anonymous_0__` and `__anonymous_1__` will be called after JIT compilation, and then discarded.
The symbol `f` and `a` will be kept, so when in REPL mode, later expression would be able to access them.


# Here is a list of features notes
1. Native datatypes: "float", "bool", "str"
    * "str" is C style `\0` ended constant char array.
    * "float" is double precision float, that is, 64bit IEEE 754 floating point.
1. Only C style struct supported, the init() method will be used to define members, and only expression like `this.var:float = 1;` are allowed.
2. Nested function (closure) not supported.
3. Global var must be inited with constant value, if dynamic init is required, use assignment to do it. e.g. `var a:float=0; a=foo();`
4. When launching script
    * A function named of "main" will be treated as the entry point, it will be called **after** all other top level stmts are executed.
    * All global stmts that not define var or function will be guaranteed to be executed in order.
    * Entry point "main" is only allowed to return nothing or `float`