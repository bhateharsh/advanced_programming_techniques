#include <iostream>
#include <vector>
#include "simple_string.h"
#include "array.h"

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//NOTE THIS IS NOT A COMPLETE LISTING OF TESTS THAT WILL BE RUN ON YOUR CODE
//Just a sample to help get you started and give you an idea of how i'll be testing
//Above each test gives the counts for std::vector and the solution i've written for your array
//As well as checking totals ensure your array doesn't leak memory
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//comment/uncomment these lines to enable tests
// #define TEST_PUSH_BACK_NEW_VEC
// #define TEST_CLEAR
// #define TEST_PUSH_FRONT_VEC
#define TEST_PUSH_FRONT_WITH_RESERVE
// #define TEST_POP_BACK
// #define TEST_INITIALIZER_LIST
// #define TEST_POP_FRONT
// #define TEST_COPY_CONSTRUCTOR
// #define TEST_RESERVE
// #define TEST_FRONT_REF
// #define TEST_BACK_REF
// #define TEST_BRACKET_REF
// #define TEST_SIZE
// #define TEST_ERASE
// #define TEST_INSERT

using std::vector;
//test your code here

int main() {

#ifdef TEST_RESERVE
{
    std::cout << "Vector" << std::endl;
    simple_string a("Goober");
    vector<simple_string> vec({a,a});
    std::cout<<"Vector Capacity: "<<vec.capacity()<<std::endl;
    simple_string::initialize_counts();
    vec.reserve(10);
    simple_string::print_counts();
    std::cout << "Array" << std::endl;
    array<simple_string> arr({a,a});
    simple_string::initialize_counts();
    arr.reserve(10);
    simple_string::print_counts();
    arr.display();
}
#endif

#ifdef TEST_CLEAR
    //Vector                    Array
    //Default: 0                Default: 0
    //Create: 0                 Create: 0
    //Copy: 0                   Copy: 0ece6122 github templated array
    //Assign: 0                 Assign: 0
    //Destruct: 2               Destruct: 2
    //Move Construct: 0         Move Construct: 0
    //Move Assign: 0            Move Assign: 0

    {
        std::cout << "Vector" << std::endl;
        simple_string a("Goober");
        vector<simple_string> vec;
        vec.reserve(5);
        vec.push_back(a);
        vec.push_back(a);
        simple_string::initialize_counts();
        vec.clear();
        simple_string::print_counts();

        std::cout << "Array" << std::endl;
        array<simple_string> arr;
        arr.reserve(5);
        arr.push_back(a);
        arr.push_back(a);
        simple_string::initialize_counts();
        arr.clear();
        simple_string::print_counts();
    }
#endif

#ifdef TEST_POP_FRONT
    //Vector                Array
    //Default: 0            Default: 0
    //Create: 0             Create: 0
    //Copy: 0               Copy: 0
    //Assign: 0             Assign: 0
    //Destruct: 1           Destruct: 1
    //Move Construct: 0     Move Construct: 0
    //Move Assign: 2        Move Assign: 2

    {
        simple_string a("Goober");
        simple_string b("Gabber");
        simple_string c("Gupper");

        std::cout << "Vector" << std::endl;
        vector<simple_string> vec;
        vec.push_back(a);
        vec.push_back(b);
        vec.push_back(c);
        simple_string::initialize_counts();
        //note: std::vec does not have pop_front
        vec.erase(vec.begin());
        simple_string::print_counts();

        std::cout << "Array" << std::endl;
        array<simple_string> arr;
        arr.push_back(a);
        arr.push_back(b);
        arr.push_back(c);
        arr.display();
        simple_string::initialize_counts();
        arr.pop_front();
        simple_string::print_counts();
        arr.display();
    }

#endif

#ifdef TEST_POP_BACK
    //Vector                Array
    //Default: 0            Default: 0
    //Create: 0             Create: 0
    //Copy: 0               Copy: 0
    //Assign: 0             Assign: 0
    //Destruct: 1           Destruct: 1
    //Move Construct: 0     Move Construct: 0
    //Move Assign: 0        Move Assign: 0
    {
        simple_string a("Goober");


        std::cout << "Vector" << std::endl;
        vector<simple_string> vec;
        vec.push_back(a);
        simple_string::initialize_counts();
        vec.pop_back();
        simple_string::print_counts();


        std::cout << "Array" << std::endl;
        array<simple_string> arr;
        arr.push_back(a);
        arr.push_back(a);
        arr.push_back(a);
        arr.display();
        simple_string::initialize_counts();
        arr.pop_back();
        simple_string::print_counts();
        arr.display();
    }
#endif

#ifdef TEST_PUSH_FRONT_WITH_RESERVE
    //Vector                Array
    //Default: 0            Default: 0
    //Create: 0             Create: 0
    //Copy: 1               Copy: 1
    //Assign: 0             Assign: 0
    //Destruct: 0           Destruct: 0
    //Move Construct: 0     Move Construct: 0
    //Move Assign: 0        Move Assign: 0

    {
        simple_string a("Goober");

        simple_string::initialize_counts();
        std::cout << "Vector" << std::endl;
        vector<simple_string> vec;
        vec.reserve(2);
        vec.insert(vec.begin(), a);
        simple_string::print_counts();

        simple_string::initialize_counts();
        std::cout << "Array" << std::endl;
        array<simple_string> arr;
        arr.reserve(2);
        arr.push_front(a);
        simple_string::print_counts();
    }
#endif

#ifdef TEST_PUSH_FRONT_VEC
    //Vector                Array
    //Default: 0            Default: 0
    //Create: 0             Create: 0
    //Copy: 1               Copy: 1
    //Assign: 0             Assign: 0
    //Destruct: 2           Destruct: 2
    //Move Construct: 2     Move Construct: 2
    //Move Assign: 0        Move Assign: 0

    {
        simple_string a("Bla");
        simple_string b("Foob");
        std::cout << "Vector" << std::endl;

        vector<simple_string> vec;
        vec.push_back(a);
        vec.push_back(a);
        simple_string::initialize_counts();
        //note std::vector doesn't have a push_front
        vec.insert(vec.begin(), b);
        simple_string::print_counts();

        std::cout << "Array" << std::endl;
        array<simple_string> arr;
        arr.push_back(a);
        arr.push_back(a);
        simple_string::initialize_counts();
        arr.push_front(b);
        simple_string::print_counts();
        arr.display();
    }

#endif

#ifdef TEST_PUSH_BACK_NEW_VEC

    //Push back new vec with no reserve
    //
    //Vector                Array
    //Default: 0            Default: 0
    //Create: 0             Create: 0
    //Copy: 1               Copy: 1
    //Assign: 0             Assign: 0
    //Destruct: 0           Destruct: 0
    //Move Construct: 0     Move Construct: 0
    //Move Assign: 0        Move Assign: 0
    {
        simple_string a("Yay");

        std::cout << "Vector" << std::endl;

        vector<simple_string> vec;
        vec.reserve(5);
        simple_string::initialize_counts();
        vec.push_back(a);
        vec.push_back(a);
        vec.push_back(a);
        simple_string::print_counts();

        std::cout << "Array" << std::endl;
        simple_string::initialize_counts();
        array<simple_string> arr(5);
        arr.push_back(a);
        arr.push_back(a);
        arr.push_back(a);
        simple_string::print_counts();
        arr.display();
    }
#endif

#ifdef TEST_INITIALIZER_LIST

    //Test initializer list
    //
    //Vector                  Array
    //Default: 0              Default: 0
    //Create: 0               Create: 0
    //Copy: 4                 Copy: 4
    //Assign: 0               Assign: 0
    //Destruct: 2             Destruct: 2
    //Move Construct: 0       Move Construct: 0
    //Move Assign: 0          Move Assign: 0

    {
        simple_string a;
        simple_string b;
        simple_string c;

        std::cout << "Vector" << std::endl;
        simple_string::initialize_counts();
        vector<simple_string> vec({a, b});
        simple_string::print_counts();

        std::cout << "Array" << std::endl;
        simple_string::initialize_counts();
        array<simple_string> arr({a, b});
        simple_string::print_counts();
    }
#endif

#ifdef TEST_COPY_CONSTRUCTOR
    //Vector                Array
    //Default: 0            Default: 0
    //Create: 0             Create: 0
    //Copy: 0               Copy: 0
    //Assign: 0             Assign: 0
    //Destruct: 1           Destruct: 1
    //Move Construct: 0     Move Construct: 0
    //Move Assign: 2        Move Assign: 2

    {
        simple_string a("Goober");
        simple_string b("Gabber");
        simple_string c("Gupper");

        std::cout << "Vector" << std::endl;
        vector<simple_string> vec({a,b,c});
        simple_string::initialize_counts();
        vector<simple_string> vec2(vec);
        simple_string::print_counts();

        std::cout << "Array" << std::endl;
        
        array<simple_string> arr({a,b,c});
        simple_string::initialize_counts();
        array<simple_string> arr2(arr);
        simple_string::print_counts();
    }

#endif

#ifdef TEST_FRONT_REF
{
    simple_string a("Hello");
    simple_string b("World");
    simple_string c("Okay");

    std::cout<<"Vector"<<std::endl;
    vector<simple_string> vec;
    vec.push_back(a);
    vec.push_back(b);
    vec.push_back(c);
    simple_string::initialize_counts();
    std::cout<<"vec.front() = "<<vec.front()<<std::endl;
    simple_string::print_counts();

    std::cout<<"Array"<<std::endl;
    array<simple_string> arr;
    arr.push_back(a);
    arr.push_back(b);
    arr.push_back(c);
    simple_string::initialize_counts();
    std::cout<<"arr.front() = "<<arr.front()<<std::endl;
    simple_string::print_counts();
}
#endif

#ifdef TEST_BACK_REF
{
    simple_string a("Hello");
    simple_string b("World");
    simple_string c("Okay");

    std::cout<<"Vector"<<std::endl;
    vector<simple_string> vec;
    vec.push_back(a);
    vec.push_back(b);
    vec.push_back(c);
    simple_string::initialize_counts();
    std::cout<<"vec.back() = "<<vec.back()<<std::endl;
    simple_string::print_counts();

    std::cout<<"Array"<<std::endl;
    array<simple_string> arr;
    arr.push_back(a);
    arr.push_back(b);
    arr.push_back(c);
    simple_string::initialize_counts();
    std::cout<<"arr.back() = "<<arr.back()<<std::endl;
    simple_string::print_counts();
}
#endif
#ifdef TEST_BRACKET_REF
{
    simple_string a("Hello");
    simple_string b("World");
    simple_string c("Okay");

    std::cout<<"Vector"<<std::endl;
    vector<simple_string> vec;
    vec.push_back(a);
    vec.push_back(b);
    vec.push_back(c);
    simple_string::initialize_counts();
    std::cout<<"vec[1] = "<<vec[1]<<std::endl;
    simple_string::print_counts();

    std::cout<<"Array"<<std::endl;
    array<simple_string> arr;
    arr.push_back(a);
    arr.push_back(b);
    arr.push_back(c);
    simple_string::initialize_counts();
    std::cout<<"arr[1] = "<<arr[1]<<std::endl;
    simple_string::print_counts();

    std::cout<<"Overloading Check"<<std::endl;  
    std::cout<<"Vector"<<std::endl;
    simple_string::initialize_counts();
    vec[1] = c;
    simple_string::print_counts();
    std::cout<<"Assigned vec[1] = "<<vec[1]<<std::endl;

    std::cout<<"Array"<<std::endl;
    simple_string::initialize_counts();
    arr[1] = c;
    simple_string::print_counts();
    std::cout<<"Assigned arr[1] = "<<arr[1]<<std::endl;
}
#endif

#ifdef TEST_SIZE
{
    simple_string a("Boo");
    
    vector<simple_string> vec;
    array<simple_string> arr;

    std::cout<<"Empty Size"<<std::endl;
    std::cout<<"vec.size() = "<<vec.size()<<std::endl;
    std::cout<<"arr.length() = "<<arr.length()<<std::endl;

    std::cout<<"Initialized Size"<<std::endl;
    vec.push_back(a);
    vec.push_back(a);
    vec.push_back(a);
    arr.push_back(a);
    arr.push_back(a);
    arr.push_front(a);
    std::cout<<"vec.size() = "<<vec.size()<<std::endl;
    std::cout<<"arr.length() = "<<arr.length()<<std::endl;

    std::cout<<"Popping and checking"<<std::endl;
    vec.pop_back();
    vec.pop_back();
    arr.pop_back();
    arr.pop_front();
    std::cout<<"vec.size() = "<<vec.size()<<std::endl;
    std::cout<<"arr.length() = "<<arr.length()<<std::endl;

}
#endif

#ifdef TEST_BEGIN
{
    simple_string a("bao");
    simple_string b("fao");
    simple_string c("gao");

    vector<simple_string> vec({a,b,c});
    std::cout<<"Vector"<<std::endl;
    std::cout<<"vec.begin() = "<<*vec.begin()<<std::endl;
    // std::cout<<"vec.end() = "<<*(vec.end()-1)<<std::endl;

    array<simple_string> arr({a,b,c});
    std::cout<<"Array"<<std::endl;
    std::cout<<"arr.begin() = "<<*arr.begin()<<std::endl;
    // std::cout<<"arr.end() = "<<*(--arr.end())<<std::endl;
    std::cout<<"arr.begin()++ = "<<*(++arr.begin())<<std::endl;
    arr.display();
}
#endif

#ifdef TEST_ERASE
{
    simple_string a("Boo");
    simple_string b("Hoo");
    simple_string c("Goo");
    simple_string d("Doo");

    std::cout<<"Vector"<<std::endl;
    vector<simple_string> vec({a,b,c,d});
    simple_string::initialize_counts();
    vec.erase(vec.begin()+1);
    vec.erase(vec.begin()+1);
    simple_string::print_counts();

    std::cout<<"Array"<<std::endl;
    array<simple_string> arr({a,b,c,d});
    arr.display();
    simple_string::initialize_counts();
    arr.erase(arr.begin()++);
    arr.display();
    arr.erase(arr.begin()++);
    arr.display();
    simple_string::print_counts();
    
}
#endif

#ifdef TEST_INSERT
{
    simple_string a("Boop");
    simple_string b("the");
    simple_string c("heckin");
    simple_string d("snoot");

    std::cout<<"Vector"<<std::endl;
    vector<simple_string> vec({a,c,c,d});
    vec.reserve(10);
    simple_string::initialize_counts();
    vec.insert(--vec.end(), b);
    simple_string::print_counts();
    for (int i = 0; i < vec.size(); i++){
        std::cout<<"vec["<<i<<"] = "<<vec[i]<<std::endl;
    }
    std::cout<<"Array"<<std::endl;
    array<simple_string> arr({a,c,c,d});
    arr.reserve(10);
    simple_string::initialize_counts();
    arr.insert(b, --arr.end());
    simple_string::print_counts();
    arr.display();
    std::cout<<"Checking Increment/Decrement"<<std::endl;
    // vec.erase(vec.end()+1);
    // arr.erase(arr.begin()++);

    // std::cout<<"Vector"<<std::endl;
    // simple_string::initialize_counts();
    // vec.insert(vec.begin()+1, b);
    // simple_string::print_counts();

    // std::cout<<"Array"<<std::endl;
    // simple_string::initialize_counts();
    // arr.insert(b, arr.begin()++);
    // simple_string::print_counts();
    // arr.display();


}
#endif
    return 0;
}