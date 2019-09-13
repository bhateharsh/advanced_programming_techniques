#include "vector3.h"

//Constructor Declarations
Vector3::Vector3(float xyz){
    ///Set all components to xyz
    x = xyz;
    y = xyz;
    z = xyz;
}
Vector3::Vector3(float x, float y, float z):x(x), y(y), z(z)
{
    ///set component by names
}

//Vector-Vector Operator
Vector3 Vector3::operator+(const Vector3& rhs) const{
    ///component-wise add
    Vector3 res;
    res.x = x + rhs.x;
    res.y = y + rhs.y;
    res.z = z + rhs.z;
    return res; 
}
Vector3 Vector3::operator-(const Vector3& rhs) const{
    ///component-wise subtract
    Vector3 res;
    res.x = x - rhs.x;
    res.y = y - rhs.y;
    res.z = z - rhs.z;
    return res; 
}
Vector3 Vector3::operator*(const Vector3& rhs) const{
    ///component-wise multiplication
    Vector3 res;
    res.x = x * rhs.x;
    res.y = y * rhs.y;
    res.z = z * rhs.z;
    return res; 
}
Vector3 Vector3::operator/(const Vector3& rhs) const{
    ///component-wise division
    Vector3 res;
    res.x = x / rhs.x;
    res.y = y / rhs.y;
    res.z = z / rhs.z;
    return res; 
}

//Vector-Scalar Operator
Vector3 Vector3::operator+(float rhs) const{
    ///add rhs to each component
    Vector3 res;
    res.x = x + rhs;
    res.y = y + rhs;
    res.z = z + rhs;
    return res;
}
Vector3 Vector3::operator-(float rhs) const{
    ///subtract rhs from each component
    Vector3 res;
    res.x = x - rhs;
    res.y = y - rhs;
    res.z = z - rhs;
    return res;
}
Vector3 Vector3::operator*(float rhs) const{
    ///multiply each component by rhs
    Vector3 res;
    res.x = x * rhs;
    res.y = y * rhs;
    res.z = z * rhs;
    return res;
}
Vector3 Vector3::operator/(float rhs) const{
    ///divide each component by rhs
    Vector3 res;
    res.x = x / rhs;
    res.y = y / rhs;
    res.z = z / rhs;
    return res;
}

//Vector-Vector Special Operations
float Vector3::operator|(const Vector3& rhs) const{
    ///dot product
    return (x*rhs.x + y*rhs.y + z*rhs.z);
}
Vector3 Vector3::operator^(const Vector3& rhs) const{
    ///cross product
    Vector3 cross_prod;
    cross_prod.x = (y*rhs.z) - (z*rhs.y);
    cross_prod.y = (z*rhs.x) - (x*rhs.z);
    cross_prod.z = (x*rhs.y) - (y*rhs.x);
    return cross_prod;
}

//Vector-Vector Short Operators
Vector3& Vector3::operator+=(const Vector3& rhs){
    ///Add Shorthand
    this->x = this->x + rhs.x;
    this->y = this->y + rhs.y;
    this->z = this->z + rhs.z;
    return *this;
}
Vector3& Vector3::operator-=(const Vector3& rhs){
    ///Subtract Shorthand
    this->x = this->x - rhs.x;
    this->y = this->y - rhs.y;
    this->z = this->z - rhs.z;
    return *this;
}
Vector3& Vector3::operator*=(const Vector3& rhs){
    ///Multiplication Shorthand
    this->x = this->x * rhs.x;
    this->y = this->y * rhs.y;
    this->z = this->z * rhs.z;
    return *this;
}
Vector3& Vector3::operator/=(const Vector3& rhs){
    ///Division Shorthand
    this->x = this->x / rhs.x;
    this->y = this->y / rhs.y;
    this->z = this->z / rhs.z;
    return *this;
}

//Vector Increment/Decrement Operator
//Vector Incrememnt
Vector3& Vector3::operator++(){
    ///Skeleton Increment
    float temp = this->x;
    this->x = this->z;
    this->z = this->y;
    this->y = temp;
    return *this;
}
Vector3 Vector3::operator++(int __unused){
    ///Postfix operator
    Vector3 result (this->x, this->y, this->z);
    ++(*this);
    return result;
}
//Vector Decrement
Vector3& Vector3::operator--(){
    ///Skeleton Decrement
    float temp = this->x;
    this->x = this->y;
    this->y = this->z;
    this->z = temp;
    return *this;
}
Vector3 Vector3::operator--(int __unused){
    ///Prefix operator
    Vector3 result (this->x, this->y, this->z);
    --(*this);
    return result;
}
//Vector-Vector Comparisons
bool Vector3::operator==(const Vector3& rhs) const{
    ///Component-wise equality
    return ((x == rhs.x) && (y == rhs.y) && (z == rhs.z));
}
bool Vector3::operator!=(const Vector3& rhs) const{
    ///Component-wise inequality
    return ((x != rhs.x) || (y != rhs.y) || (z != rhs.z));
}
