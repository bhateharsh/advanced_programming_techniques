#ifndef VECTOR3_H
#define VECTOR3_H

struct Vector3{
    //Vector Components
    float x;
    float y;
    float z;
    //Constructors
    Vector3() = default;                                //Default constructor
    Vector3(float xyz);                                 //set x,y and z to xyz
    Vector3(float x, float y, float z);                  //set component by name
    //Vector-Vector Operator
    Vector3 operator+(const Vector3& rhs) const;              //component-wise add
    Vector3 operator-(const Vector3& rhs) const;              //component-wise subtract
    Vector3 operator*(const Vector3& rhs) const;              //component-wise multiplication
    Vector3 operator/(const Vector3& rhs) const;              //component-wise division
    //Vector-Scalar Operator
    Vector3 operator+(float rhs) const;                       //add rhs to each component
    Vector3 operator-(float rhs) const;                       //subtract rhs from each component
    Vector3 operator*(float rhs) const;                       //multiply each component by rhs
    Vector3 operator/(float rhs) const;                       //divide each component by rhs
    //Vector-Vector Special Operations
    float operator|(const Vector3& rhs) const;                //dot product
    Vector3 operator^(const Vector3& rhs) const;              //cross product
    //Vector-Vector Short Operators
    Vector3& operator+=(const Vector3& rhs);            //component-wise add
    Vector3& operator-=(const Vector3& rhs);            //component-wise subtraction
    Vector3& operator*=(const Vector3& rhs);            //component-wise multiplication
    Vector3& operator/=(const Vector3& rhs);            //component-wise division
    //Vector Increment/Decrement Operator
    /// Vector3++ and ++Vector3 rotate xyz to the right
    /// i.e. make x = z, y = x, z = y
    Vector3& operator++();
    Vector3 operator++(int __unused);
    /// Vector3-- and --Vector3 rotate xyz to the left
    /// i.e. make x = y, y = z, z = x
    Vector3& operator--();
    Vector3 operator--(int __unused);
    //Vector-Vector Comparison
    bool operator==(const Vector3& rhs) const;                //component-wise equality
    bool operator!=(const Vector3& rhs) const;                //component-wise inequality
};

#endif