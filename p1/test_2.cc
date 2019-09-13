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
#define TEST_PUSH_BACK_NEW_VEC
// #define TEST_CLEAR
// #define TEST_PUSH_FRONT_VEC
// #define TEST_PUSH_FRONT_WITH_RESERVE
// #define TEST_POP_BACK
// #define TEST_INITIALIZER_LIST
// #define TEST_POP_FRONT
// #define TEST_1
// #define TEST_2
// #define TEST_3
// #define TEST_4

using std::vector;
void Test1();
void Test2();
void Test3();
void Test4();
//test your code here

int main() {

#ifdef TEST_CLEAR
    //Vector                    Array
    //Default: 0                Default: 0
    //Create: 0                 Create: 0
    //Copy: 0                   Copy: 0
    //Assign: 0                 Assign: 0
    //Destruct: 2               Destruct: 2
    //Move Construct: 0         Move Construct: 0
    //Move Assign: 0            Move Assign: 0

    {
        std::cout << "Test Clear\n";
        std::cout << "Vector" << std::endl;
        simple_string a("Goober");
        vector<simple_string> vec;
        vec.push_back(a);
        vec.push_back(a);
        simple_string::initialize_counts();
        vec.clear();
        simple_string::print_counts();

        std::cout << "Array" << std::endl;
        array<simple_string> arr;
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
        std::cout << "\nPop Front\n";
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
        simple_string::initialize_counts();
        //arr.pop_front();
        arr.erase(arr.begin());
        simple_string::print_counts();
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
        std::cout << "\nPop Back\n";
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
        simple_string::initialize_counts();
        arr.pop_back();
        simple_string::print_counts();
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
        std::cout << "\nPush Front\n";
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
        std::cout << "\nPush Front Vec\n";
        simple_string a;
        simple_string b("Foob");
        std::cout << "Vector" << std::endl;

        vector<simple_string> vec;
        vec.push_back(a);
        vec.push_back(a);
        simple_string::initialize_counts();
        //note std::vector doesn't have a push_front
        vec.insert(vec.begin(), a);
        simple_string::print_counts();

        std::cout << "Array" << std::endl;
        array<simple_string> arr;
        arr.push_back(a);
        arr.push_back(a);
        simple_string::initialize_counts();
        arr.push_front(b);
        //arr.insert(b, arr.begin());
        simple_string::print_counts();
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
        std::cout << "\nPush Back New Vector\n";
        simple_string a;

        std::cout << "Vector" << std::endl;
        simple_string::initialize_counts();
        vector<simple_string> vec;
        vec.push_back(a);
        simple_string::print_counts();

        std::cout << "Array" << std::endl;
        simple_string::initialize_counts();
        array<simple_string> arr;
        arr.push_back(a);
        simple_string::print_counts();
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
        std::cout << "\nInitializer List\n";
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

#ifdef TEST_1
    Test1();
#endif

#ifdef TEST_2
    Test2();
#endif

#ifdef TEST_3
    Test3();
#endif

#ifdef TEST_4
    Test4();
#endif

    return 0;
}

void Test1(){
    std::cout << "\n\nTesting Push Functionalities. (Test 1)\n\n";

    std::cout << "Array\n";
    simple_string::initialize_counts();
    array<simple_string> v;
    for(int i = 10; i > 0; --i){
        std::string str = std::to_string(i);
        const char *c = str.c_str();
        v.push_front(simple_string(c));
    }
    for (int i = 0; i < 10; ++i){
        std::string str = std::to_string(i+11);
        const char *c = str.c_str();
        v.push_back(simple_string(c));
    }
    std::cout << "Test 1 results: \n";

    for (size_t i = 0; i < v.length(); ++i){
        std::cout << v[i] << ", ";
    }
    std::cout << "\n";
    simple_string::print_counts();

    std::cout << "Vector\n";
    simple_string::initialize_counts();

    vector<simple_string> vec;
    for(int i = 10; i > 0; --i){
        std::string str = std::to_string(i);
        const char *c = str.c_str();
        vec.insert(vec.begin(), simple_string(c));
    }
    for (int i = 0; i < 10; ++i){
        std::string str = std::to_string(i+11);
        const char *c = str.c_str();
        vec.push_back(simple_string(c));
    }
    std::cout << "Test 1 results: \n";

    for (size_t i = 0; i < vec.size(); ++i){
        std::cout << vec[i] << ", ";
    }
    std::cout << "\n";
    simple_string::print_counts();

    std::cout << "\n *Test should count from 1 to 20\n";

}

void Test2(){
    std::cout << "\n\nTesting Pop Functionalities. (Test 2)\n\n";

    std::cout << "Array\n";
    simple_string::initialize_counts();
    array<simple_string> v;

    for (int i = 1; i < 20; i=i+2){
        std::string str = std::to_string(i);
        const char *c = str.c_str();
        v.push_back(simple_string(c));
    }
    for (int i = 20; i > 0; i = i-2){
        std::string str = std::to_string(i);
        const char *c = str.c_str();
        v.push_back(simple_string(c));
    }

    std::cout << "Test 2 results: \n";

    bool front = true;
    while(!v.empty()){
        if(front){
            std::cout << v.front() << ", ";
            v.pop_front();
            front = false;
        }else{
            std::cout << v.back() << ", ";
            v.pop_back();
            front = true;
        }
    }
    std::cout << "\n";
    simple_string::print_counts();

    std::cout << "Vector\n";
    simple_string::initialize_counts();
    vector<simple_string> vec;

    for (int i = 1; i < 20; i=i+2){
        std::string str = std::to_string(i);
        const char *c = str.c_str();
        vec.push_back(simple_string(c));
    }
    for (int i = 20; i > 0; i = i-2){
        std::string str = std::to_string(i);
        const char *c = str.c_str();
        vec.push_back(simple_string(c));
    }

    std::cout << "Test 2 results: \n";

    front = true;
    while(!vec.empty()){
        if(front){
            std::cout << vec.front() << ", ";
            vec.erase(vec.begin());
            front = false;
        }else{
            std::cout << vec.back() << ", ";
            vec.pop_back();
            front = true;
        }
    }
    std::cout << "\n";
    simple_string::print_counts();


    std::cout << "\n *Test should count from 1 to 20\n";
    
}

void Test3(){

    simple_string::initialize_counts();
    std::cout << "\n\nTesting Constructor Functionalities. (Test 3)\n\n";
    std::cout << "Array\n";
    std::string str1 = "[Passed] ";
    const char *c1 = str1.c_str();
    std::string str2 = "Constructor Functionality";
    const char *c2 = str2.c_str();
    std::string str3 = "(Init List)\n";
    const char *c3 = str3.c_str();
    array<simple_string> v = {simple_string(c1), 
    simple_string(c2), simple_string(c3)};

    for (size_t i = 0; i < v.length(); ++i){
        std::cout << v[i];
    }

    v.pop_back();
    std::string str4 = "(Copy)\n";
    const char *c4 = str4.c_str();
    v.push_back(simple_string(c4));

    array<simple_string> v1 = v;
    
    for (size_t i = 0; i < v1.length(); ++i){
        std::cout << v1[i];
    }

    std::string str5 = "(Move + index assignment)\n";
    const char *c5 = str5.c_str();
    v1[2] = simple_string(c5);

    array<simple_string> v2 = std::move(v1);

    if(v1.empty()){
        for (size_t i = 0; i < v2.length(); ++i){
            std::cout << v2[i];
        }
    }else{
        std::cout << "Move constructor failed.\n";
    }

    array<simple_string> v3(5, simple_string(c1));
    for (size_t i = 0; i < v3.length(); ++i){
        std::cout << v3[i];
    }
    std::cout << "5 Passes = [Passed] Clone Constructor\n";

    v.clear();
    if(v.empty()){
        std::cout << "[Passed] Clear Function\n";
    }
    simple_string::print_counts();



    std::cout << "Vector\n";
    simple_string::initialize_counts();

    vector<simple_string> vec = {simple_string(c1), 
    simple_string(c2), simple_string(c3)};

    for (size_t i = 0; i < vec.size(); ++i){
        std::cout << vec[i];
    }

    vec.pop_back();

    vec.push_back(simple_string(c4));

    vector<simple_string> vec1 = vec;
    
    for (size_t i = 0; i < vec1.size(); ++i){
        std::cout << vec1[i];
    }

    vec1[2] = simple_string(c5);

    vector<simple_string> vec2 = std::move(vec1);

    if(vec1.empty()){
        for (size_t i = 0; i < vec2.size(); ++i){
            std::cout << v2[i];
        }
    }else{
        std::cout << "Move constructor failed.\n";
    }

    vector<simple_string> vec3(5, simple_string(c1));
    for (size_t i = 0; i < vec3.size(); ++i){
        std::cout << v3[i];
    }
    std::cout << "5 Passes = [Passed] Clone Constructor\n";

    vec.clear();
    if(vec.empty()){
        std::cout << "[Passed] Clear Function\n";
    }

    simple_string::print_counts();


}

void Test4(){
    simple_string::initialize_counts();
    std::cout << "\n\nTesting Iterator Functionalities. (Test 4)\n\n";

    std::cout << "Array\n";
    array<simple_string> v;
    for (int i = 0; i < 10; ++i){
        std::string str = std::to_string(i);
        const char *c = str.c_str();
        v.push_back(simple_string(c));
    }

    array_iterator<simple_string> it = v.begin();
    while(it != v.end()){
        std::cout << *it++ << ", ";
    }
    std::cout << "\n*Test should count from 0 to 9.\n";

    array_iterator<simple_string> it1 = v.end();
    size_t i = 0;
    for (it = v.begin(); it != v.end(); ++it){
        if (i++ == 4) {
            it1 = it;
            break;
        }
    }
    std::string str = "[Passed] Insert Function\n";
    const char *c = str.c_str();
    v.insert(simple_string(c), it1);
    std::cout << *it1;

    v.erase(it1);

    array_iterator<simple_string> it2 = v.begin();
    std::cout << "Insert fx missing if test passes\n";
    for(; it2 != v.end(); ++it2){
        std::cout << *it2 << ", ";
    }
    std::cout << std::endl;

    if(it1 != it2){
        std::cout << "[Passed] Inequality Test\n";
    }

    if (!(it1 == it2)){
        std::cout << "[Passed] Equality Test\n";
    }

    simple_string::print_counts();

    simple_string::initialize_counts();

    std::cout << "Vector\n";
    vector<simple_string> vec;
    for (int i = 0; i < 10; ++i){
        std::string str = std::to_string(i);
        const char *c = str.c_str();
        vec.push_back(simple_string(c));
    }

    vector<simple_string>::iterator itvec = vec.begin();
    while(itvec != vec.end()){
        std::cout << *itvec++ << ", ";
    }
    std::cout << "\n*Test should count from 0 to 9.\n";

    vector<simple_string>::iterator itvec1 = vec.end();
    size_t ii = 0;
    for (itvec = vec.begin(); itvec != vec.end(); ++itvec){
        if (ii++ == 4) {
            itvec1 = itvec;
            break;
        }
    }

    vec.insert(itvec1, simple_string(c));
    std::cout << *itvec1;

    vec.erase(itvec1);

    vector<simple_string>::iterator itvec2 = vec.begin();
    std::cout << "Insert fx missing if test passes\n";
    for(; itvec2 != vec.end(); ++itvec2){
        std::cout << *itvec2 << ", ";
    }
    std::cout << std::endl;

    if(itvec1 != itvec2){
        std::cout << "[Passed] Inequality Test\n";
    }

    if (!(itvec1 == itvec2)){
        std::cout << "[Passed] Equality Test\n";
    }
    simple_string::print_counts();


}
