//
// Created by Pranav on 15/06/24.
//

#ifndef NATIVE_DATACONTAINERVALUE_H
#define NATIVE_DATACONTAINERVALUE_H
class DataContainerValue {
public:
    virtual void* getPtr() = 0; // Returns a pointer to the underlying data structure
    virtual void setValPtr(void* ptr) = 0; // Sets the underlying data structure pointer
};
#endif //NATIVE_DATACONTAINERVALUE_H
