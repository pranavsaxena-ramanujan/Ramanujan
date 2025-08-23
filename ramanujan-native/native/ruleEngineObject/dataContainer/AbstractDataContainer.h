#ifndef ABSTRACTDATACONTAINER_H
#define ABSTRACTDATACONTAINER_H



#include "array/ArrayValue.h"
#include "DataContainerValue.h"
#include <string>

class ArrayValue;

class AbstractDataContainer {
public:
    DataContainerValue* ptr;

    AbstractDataContainer(DataContainerValue* inPtr) : ptr(inPtr) {}

    virtual std::string getId() = 0;

    // This would give the pointer to underlying datastructure. Using this ptr, we can change object
    // of that datastructure on that location without doing any other ops.
    // important for function call optimization.
    virtual DataContainerValue* getVal()
    {
        return ptr;
    }

    virtual  ~AbstractDataContainer() = default;
};

#endif // ABSTRACTDATACONTAINER_H