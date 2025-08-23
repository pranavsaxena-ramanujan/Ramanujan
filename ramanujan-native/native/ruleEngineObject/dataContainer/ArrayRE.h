//
// Created by pranav on 22/3/24.
//

#ifndef NATIVE_ARRAYRE_H
#define NATIVE_ARRAYRE_H

#include "DataContainerRE.h"
#include "array/ArrayValue.h"
#include "iostream"

#include "stack"

class ArrayRE: public RuleEngineInputUnits, public AbstractDataContainer {
private:
    Array* array;

    std::string dataType;

public:
    ArrayValDataContainer* arrayValue;
    std::string name, frameCount;

    ArrayRE(Array *array) : AbstractDataContainer(new ArrayValDataContainer(array, this->id)), arrayValue((ArrayValDataContainer*)ptr) {
        this->array = array;

        this->id = array->id;
        this->name = array->name;
        this->frameCount = array->frameCount;
    }

    void setFields(std::unordered_map<std::string, RuleEngineInputUnits *> *map) override {

    }

    void process() override {

    }

    void destroy() {
        if (arrayValue->arrayValue) {
            arrayValue->arrayValue->destroy();
            delete arrayValue->arrayValue;
            arrayValue->arrayValue = nullptr;
        }
    }

    std::string getId() override {
        return std::string();
    }
};
#endif //NATIVE_ARRAYRE_H
