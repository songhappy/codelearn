// data types and variables and pointers, 
// operators, control structures,
// methods, functions,
// classes and objects, inheritance, polymorphism, 
// access specifiers, static members
// file handling, exception handling, dynamic memory, 
// data structures, algorithms, 
// multithreading, web programming, etc.

#include <string>
#include <iostream>
#include <memory>

// Chap 1: C++ program to demonstrate the various data types and variables
void def_vars() {
    int integerVar = 10;    // Integer variable
    float floatVar = 10.5f;    // Floating point variable
    double doubleVar = 20.99;    // Double precision floating point variable
    char charVar = 'A';    // Character variable
    bool boolVar = true;    // Boolean variable
    std::string stringVar = "Hello, World!";    // String variable (using the C++ string class)
}
 
// Pointers
void def_pointers() {
    int integerVar = 10;    // Integer variable
    int* intPtr = &integerVar;    // Pointer to an integer variable
    float floatVar = 10.5f;    // Floating point variable
    float* floatPtr = &floatVar;    // Pointer to a floating point variable
    double doubleVar = 20.99;    // Double precision floating point variable
    double* doublePtr = &doubleVar;    // Pointer to a double precision floating point variable
    char charVar = 'A';    // Character variable
    char* charPtr = &charVar;    // Pointer to a character variable
    bool boolVar = true;    // Boolean variable
    bool* boolPtr = &boolVar;    // Pointer to a boolean variable
    std::string stringVar = "Hello, World!";    // String variable (using the C++ string class)
    std::string* stringPtr = &stringVar;    // Pointer to a string variable

    std::unique_ptr<int> ptr = std::make_unique<int>(10);
    std::cout << *ptr << std::endl; // Output: 10
    // No need to manually delete, memory is automatically cleaned up.


}

// Chap 2: C++ program to demonstrate the various operators, controls structures and methods
void def_operators() {
    int a = 10, b = 20, c = 0;    // Variables
    c = a + b;    // Addition
    c = a - b;    // Subtraction
    c = a * b;    // Multiplication
    c = a / b;    // Division
    c = a % b;    // Modulus
    c = a++;    // Post-increment, c = a, a = a + 1
    c = ++a;    // Pre-increment, a = a + 1, c = a
    c = a--;    // Post-decrement
    c = --a;    // Pre-decrement
    c = a && b;    // Logical AND
    c = a || b;    // Logical OR
    c = !a;    // Logical NOT
    c = a == b;    // Equal to
    c = a != b;    // Not equal to
    c = a > b;    // Greater than
    c = a < b;    // Less than
    c = a >= b;    // Greater than or equal to
    c = a <= b;    // Less than or equal to
    c = a & b;    // Bitwise AND
    c = a | b;    // Bitwise OR
    c = a ^ b;    // Bitwise XOR
    c = ~a;    // Bitwise NOT
    c = a << b;    // Left shift,  Useful for fast multiplication by powers of 2^b.
    c = a >> b;    // Right shift, Useful for fast division by powers of 2^b
}
void def_control_structures() {
    // if statement
    int a = 10, b = 20;    // Variables
    if (a > b) {    // If statement
        std::cout << a << '\n'; // Code to execute if a is greater than b
    } else {    // Else statement
        std::cout << b << '\n'; // Code to execute if a is less than or equal to b
    }
    switch (a) {    // Switch statement
        case 1:    // Case 1
            std::cout << 1 << '\n'; // Code to execute if a is 1
            break;    // Break statement
        case 2:    // Case 2
            std::cout << 2 << '\n'; // Code to execute if a is 2
            break;    // Break statement
        default:    // Default case
            std::cout << 0 << '\n'; // Code to execute if a is neither 1 nor 2
            break;    // Break statement
    }
    // Loops
    int sum = 0;    // Variable
    for (int i = 0; i < 10; i++) {    // For loop
        sum = sum + i; // Code to execute 10 times
    }
    a = 10; // Reset a to its initial value
    while (a < b) {    // While loop
        sum = sum + a; // Code to execute while a is less than b
        a++; // Code to execute while a is less than b
    }
    a = 10; // Reset a to its initial value
    do {    // Do-while loop
        sum = sum + a; // Code to execute at least once
        a++; 
    } while (a < b);
}

// Chap 3: C++ program to demonstrate the various functions and methods
    // Function declaration
    int add(int a, int b);    // Function prototype
    // Function definition outside of any function
    int add(int a, int b) {
        return a + b; 
    }
    // Function call
 
    // Method declaration
    class MyClass {    // Class definition
    public:    // Access specifier
        int add(int a, int b);    // Method prototype
    };
    int MyClass::add(int a, int b) {    
        // Defining methods outside the class is a common practice in C++ that enhances code organization, readability, maintainability, and encapsulation. 
        // While it is perfectly valid to define methods inside the class, especially for smaller or simpler classes, using external definitions is beneficial for larger and more complex projects.
        return a + b;    // Method body
    }

    void def_functions(){
    int sum = add(10, 20);    // Function call
    std::cout << sum << '\n';    // Output: 30
        // Method call
    MyClass obj;    // Object creation
    sum = obj.add(10, 20);    // Method call
    std::cout << sum << '\n';    // Output: 30
    }



// Chap 4: C++ program to demonstrate the various classes and objects, inheritance and polymorphism
// Inheritance
class Animal {    // Base class
public:    // Access specifier
    void eat() {    // Method
        std::cout << "I can eat." << '\n';    // Method body
    }
public:    // Access specifier
    virtual void sound() {    // Virtual method
        std::cout << "Animal makes a sound." << '\n';    // Method body
    }
};

// Polymorphism
class Dog : public Animal {    // Derived class
public:    // Access specifier
    void sound() {    // Method
        std::cout << "Dog barks." << '\n';    // Method body
    }
};
class Cat : public Animal {    // Derived class
public:    // Access specifier
    void sound() {    // Method
        std::cout << "Cat meows." << '\n';    // Method body
    }
};

// Object creation
void  demonstratePolymorphism() {
    Animal* animal;    // Pointer to base class
    Dog dog;    // Object creation
    Cat cat;    // Object creation
    animal = &dog;    // Pointer to derived class
    animal->sound();    // Call to derived class method
    animal = &cat;    // Pointer to derived class
    animal->sound();    // Call to derived class method
}

// chap 5: C++ program to demonstrate the use of access specifiers and static members
// Access Specifiers
// public, private, protected, static, and other access specifiers, are used to define the visibility and accessibility of class members in C++.
// Three access specifiers in C++: public, private, and protected.
// Public members are accessible from outside the class, private members are not accessible from outside the class, and protected members are accessible from derived classes.
// The default access specifier in a class is private.
class Base {
public:
    int publicVar;    // Public variable
    Base() : publicVar(0), protectedVar(0), privateVar(0) {}    // Constructor give default values of 0
    void publicMethod() {    // Public method
        std::cout << "Public method called: " << publicVar<< std::endl;
    }

protected:
    int protectedVar;    // Protected variable
    void protectedMethod() {    // Protected method
        std::cout << "Protected method called :" << protectedVar  << std::endl;
    }

private:
    int privateVar;    // Private variable

    void privateMethod() {    // Private method
        std::cout << "Private method called: " << privateVar << std::endl;
    }
};

class Derived : public Base {
public:
    void accessBaseMembers() {
        publicVar = 10;    // Accessing public variable
        protectedVar = 20;    // Accessing protected variable
        // privateVar = 30;    // Error: Cannot access private variable

        publicMethod();    // Calling public method
        protectedMethod();    // Calling protected method
        // privateMethod();    // Error: Cannot call private method
    }
};

int demonstrateAcess() {
    Base baseObj;
    baseObj.publicVar = 10;    // Accessing public variable
    baseObj.publicMethod();    // Calling public method
    // baseObj.protectedVar = 20;    // Error: Cannot access protected variable
    // baseObj.privateVar = 30;    // Error: Cannot access private variable

    Derived derivedObj;
    derivedObj.accessBaseMembers();    // Accessing base class members through derived class

    return 0;
}


// static members
// Static members are shared among all instances of a class and can be accessed without creating an object of the class.
// Static members are commonly used for constants, counters, flags, and utility functions that do not depend on the state of a specific object.
// Static members can be public, private, or protected, and they can be variables, methods, or classes.
// Static members are initialized outside the class using the scope resolution operator (::).
// Static members are shared among all instances of a class and can be accessed without creating an object of the class.
class Counter {
public:
    static int count;    // Static variable to keep track of the number of objects
    static int constant1;    // Static variable to keep track of the number of objects

    Counter() {
        count++;    // Increment count when a new object is created
    }

    static int getCount() {    // Static method to get the current count
        return count;
    }
};

int Counter::constant1 = 10;    // Initialize static variable
int Counter::count = 0;    // Initialize static variable

void demonstrateStatics() {
    Counter obj1;    // Create first object
    Counter obj2;    // Create second object
    Counter obj3;    // Create third object

    std::cout << "Constant1: " << Counter::constant1 << std::endl;    // Output: 10
    std::cout << "Number of objects created: " << Counter::getCount() << std::endl;    // Output: 3
    std::cout << "Number of objects created: " << obj1.getCount() << std::endl;   // Output: 3
    std::cout << "Number of objects created: " << obj2.getCount() << std::endl;   // Output: 3
    std::cout << "Number of objects created: " << obj3.getCount() << std::endl;   // Output: 3
}


// Chap 6: C++ program to demonstrate the use of file handling, exception handling, and dynamic memory
// File Handling
#include <fstream>
#include <iostream>
void demonstrateFileHandling() {
    std::ofstream outFile("examplein.txt");     // Create and write to a file
    if (outFile.is_open()) {
        outFile << "Hello, World!" << std::endl;
        outFile << "This is a file handling example in C++." << std::endl;
        outFile.close();
    } else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }

    // Read from a file
    std::ifstream inFile("exampleout.txt");
    if (inFile.is_open()) {
        std::string line;
        while (std::getline(inFile, line)) {
            std::cout << line << std::endl;
        }
        inFile.close();
    } else {
        std::cerr << "Unable to open file for reading." << std::endl;
    }
}



// Exception Handling   
void demonstrateExceptionHandling() {
    try {
        int a = 10, b = 0;
        if (b == 0) {
            throw "Division by zero error.";    // Throw an exception
        }
        int result = a / b;
        std::cout << "Result: " << result << std::endl;
    } catch (const char* msg) {    // Catch the exception
        std::cerr << "Error: " << msg << std::endl;
    }
}



// Dynamic Memory
void demonstrateDynamicMemory() {
    int* ptr = new int;    // Allocate memory
    *ptr = 10;    // Assign value
    std::cout << "Value: " << *ptr << std::endl;    // Output: 10
    delete ptr;    // Deallocate memory
}


// Chap 7: C++ program to demonstrate the various data structures and algorithms
// Data Structures
// Vectors, Arrays, Linked Lists, Stacks, Queues, Trees, Graphs, Hash Tables, etc.

#include <iostream>
#include <vector>
#include <list>
#include <stack>
#include <queue>
#include <unordered_map>
#include <unordered_set>

// Vector
void demonstrateVector() {
    std::vector<std::string> stringVector = {"C++", "Java", "Python"};    // Vector of strings
    std::vector<float> floatVector = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f};    // Vector of floats
    std::vector<int> vec = {1, 2, 3, 4, 5}; //A dynamic array provided by the C++ Standard Library (std::vector
    vec.push_back(6);
        vec.push_back(7);
    vec.pop_back();
    for (int val : vec) {
        std::cout << "Vector element: " << val << std::endl; 
    }
}

// Array
void demonstrateArray() {
    int arr[5] = {1, 2, 3, 4, 5}; //A fixed-size sequence of elements of the same type
    std::string stringArray[3] = {"C++", "Java", "Python"};    // Array of strings
    float floatArray[5] = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f};    // Array of floats
    for (int i = 0; i < 5; ++i) {
        std::cout << "Array element " << i << ": " << arr[i] << std::endl;
    }
}

// Linked List
void demonstrateLinkedList() {
    std::list<int> linkedList = {1, 2, 3, 4, 5};
    for (int val : linkedList) {
        std::cout << "Linked List element: " << val << std::endl;
    }
}

// Stack
void demonstrateStack() {
    std::stack<int> stack;
    stack.push(1);
    stack.push(2);
    stack.push(3);
    while (!stack.empty()) {
        std::cout << "Stack element: " << stack.top() << std::endl;
        stack.pop();
    }
}

// Queue
void demonstrateQueue() {
    std::queue<int> queue;
    queue.push(1);
    queue.push(2);
    queue.push(3);
    while (!queue.empty()) {
        std::cout << "Queue element: " << queue.front() << std::endl;
        queue.pop();
    }
}

// Graph (using adjacency list)
void demonstrateGraph() {
    std::unordered_map<int, std::vector<int>> graph;
    graph[0] = {1, 2};
    graph[1] = {2};
    graph[2] = {0, 3};
    graph[3] = {3};

    for (const auto& node : graph) {
        std::cout << "Node " << node.first << ": ";
        for (int neighbor : node.second) {
            std::cout << neighbor << " ";
        }
        std::cout << std::endl;
    }
}

// Hash Table
void demonstrateHashTable() {
    std::unordered_map<std::string, int> hashTable;
    hashTable["one"] = 1;
    hashTable["two"] = 2;
    hashTable["three"] = 3;

    for (const auto& pair : hashTable) {
        std::cout << "Hash Table element: " << pair.first << " -> " << pair.second << std::endl;
    }
}

// When to use them
void whenToUse() {
    std::cout << "Use Arrays when you need fast access to elements by index and the size is fixed." << std::endl;
    std::cout << "Use Linked Lists when you need efficient insertions and deletions at any position." << std::endl;
    std::cout << "Use Stacks when you need LIFO (Last In, First Out) access to elements." << std::endl;
    std::cout << "Use Queues when you need FIFO (First In, First Out) access to elements." << std::endl;
    std::cout << "Use Graphs to represent networks of nodes and edges, such as social networks or maps." << std::endl;
    std::cout << "Use Hash Tables when you need fast access to elements by key." << std::endl;
}


// Algorithms
// Sorting, Searching, Graph Traversal, Dynamic Programming, Greedy Algorithms, Divide and Conquer, Backtracking, etc.
#include <algorithm>
#include <numeric>
#include <functional>
// Binary Search
bool binarySearch(const std::vector<int>& arr, int target) {
    int left = 0;
    int right = arr.size() - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) {
            return true;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return false;
}

void demonstrateBinarySearch() {
    std::vector<int> arr = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int target = 5;
    bool found = binarySearch(arr, target);
    if (found) {
        std::cout << "Element " << target << " found in the array." << std::endl;
    } else {
        std::cout << "Element " << target << " not found in the array." << std::endl;
    }
}



// Chap 8: C++ program to demonstrate the use of multithreading
#include <iostream>
#include <thread>
void printNumbers(int start, int end) {
    for (int i = start; i <= end; ++i) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

void demonstrateMultithreading() {
    std::thread t1(printNumbers, 1, 5);    // Create a thread
    std::thread t2(printNumbers, 6, 10);    // Create another thread

    t1.join();    // Join the first thread
    t2.join();    // Join the second thread
}



#include <iostream>
int main() {
    def_vars();
    def_pointers();
    def_operators();
    def_control_structures();
    def_functions();
    demonstratePolymorphism();
    demonstrateStatics();
    demonstrateFileHandling();
    demonstrateExceptionHandling();
    demonstrateDynamicMemory();
    demonstrateVector();
    demonstrateArray();
    demonstrateLinkedList();
    demonstrateStack();
    demonstrateQueue();
    demonstrateGraph();
    demonstrateHashTable();
    whenToUse();
    demonstrateAcess();
    demonstrateBinarySearch();
    demonstrateMultithreading();


    return 0;
}