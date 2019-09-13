#include <iostream>
#include "vector3.h"

int main(){
    Vector3 a;
    Vector3 b(3,4,5);
    Vector3 c(5);
    Vector3 d = a+b;
    Vector3 e;
    std::cout<<"Vector a:"<<"x: "<<a.x<<"; y: "<<a.y<<"; z:"<<a.z<<"\n";
    std::cout<<"Vector b:"<<"x: "<<b.x<<"; y: "<<b.y<<"; z:"<<b.z<<"\n";
    std::cout<<"Vector c:"<<"x: "<<c.x<<"; y: "<<c.y<<"; z:"<<c.z<<"\n";
    //Vector-Vector Operations
    std::cout<<"Addition of b and a yeilds:\n";
    std::cout<<"x = "<<d.x<<"; y = "<<d.y<<"; z = "<<d.z<<"\n";
    std::cout<<"Subtraction of b and c yeilds: \n";
    d = b - c;
    std::cout<<"x = "<<d.x<<"; y = "<<d.y<<"; z = "<<d.z<<"\n";
    std::cout<<"Multiplication of b and c yeilds: \n";
    d = b * c;
    std::cout<<"x = "<<d.x<<"; y = "<<d.y<<"; z = "<<d.z<<"\n";
    std::cout<<"Division of b and c yeilds: \n";
    d = b / c;
    std::cout<<"x = "<<d.x<<"; y = "<<d.y<<"; z = "<<d.z<<"\n";
    //Vector-Scalar Operations
    std::cout<<"Addition of b and 10 yeilds:\n";
    d = b + 10;
    std::cout<<"x = "<<d.x<<"; y = "<<d.y<<"; z = "<<d.z<<"\n";
    std::cout<<"Subtraction of b and 10 yeilds: \n";
    d = b - 10;
    std::cout<<"x = "<<d.x<<"; y = "<<d.y<<"; z = "<<d.z<<"\n";
    std::cout<<"Multiplication of b and 10 yeilds: \n";
    d = b * 10;
    std::cout<<"x = "<<d.x<<"; y = "<<d.y<<"; z = "<<d.z<<"\n";
    std::cout<<"Division of b and 10 yeilds: \n";
    d = b / 10;
    std::cout<<"x = "<<d.x<<"; y = "<<d.y<<"; z = "<<d.z<<"\n";
    //Vector-Vector Special Operations
    std::cout<<"Dot product of b and c: "<<(b|c)<<"\n";
    std::cout<<"Cross product of b and c: \n";
    d = b^c;
    std::cout<<"x = "<<d.x<<"; y = "<<d.y<<"; z = "<<d.z<<"\n";
    //Vector-Vector Short Operations
    b += c;
    std::cout<<"b+=c: "<<b.x<<"; "<<b.y<<";  "<<b.z<<"\n";
    b -= c;
    std::cout<<"b-=c: "<<b.x<<"; "<<b.y<<";  "<<b.z<<"\n";
    b *= c;
    std::cout<<"b*=c: "<<b.x<<"; "<<b.y<<";  "<<b.z<<"\n";
    b /= c;
    std::cout<<"b/=c: "<<b.x<<"; "<<b.y<<";  "<<b.z<<"\n";
    //Vector increment
    std::cout<<"Vector b is: x = "<<b.x<<"; "<<b.y<<";  "<<b.z<<"\n";
    e = b++;
    std::cout<<"Vector b is: x = "<<b.x<<"; "<<b.y<<";  "<<b.z<<"\n";
    std::cout<<"Vector e(b++) is: x = "<<e.x<<"; "<<e.y<<";  "<<e.z<<"\n";
    e = ++b;
    std::cout<<"Vector b is: x = "<<b.x<<"; "<<b.y<<";  "<<b.z<<"\n";
    std::cout<<"Vector e(++b) is: x = "<<e.x<<"; "<<e.y<<";  "<<e.z<<"\n";
    e = b--;
    std::cout<<"Vector b is: x = "<<b.x<<"; "<<b.y<<";  "<<b.z<<"\n";
    std::cout<<"Vector e(b--) is: x = "<<e.x<<"; "<<e.y<<";  "<<e.z<<"\n";
    e = --b;
    std::cout<<"Vector b is: x = "<<b.x<<"; "<<b.y<<";  "<<b.z<<"\n";
    std::cout<<"Vector e(--b) is: x = "<<e.x<<"; "<<e.y<<";  "<<e.z<<"\n";
    
    //Vector-Vector Comparison
    if (a == a){
        std::cout<<"Vector A and B are equal\n";
    }
    if (b != a){
        std::cout<<"Vector A and B are not equal\n";
    }
    return 0;
}