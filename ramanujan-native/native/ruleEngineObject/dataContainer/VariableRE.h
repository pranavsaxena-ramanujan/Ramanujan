//
// Created by pranav on 22/3/24.
//

#ifndef NATIVE_VARIABLERE_H
#define NATIVE_VARIABLERE_H

#include "Variable.hpp"
#include "../ruleEngineObject/ConstantRE.h"
#include "DataContainerRE.h"
#include "../processing/ProcessingResult.hpp"
#include "array/ArrayValue.h"
#include <stack>
#include <cmath>

class DoublePtr : public DataContainerValue{

public:
    double value;

    DoublePtr(double value = 0.0) : value(value) {}

    DoublePtr(DoublePtr& inDoublePtr)
    {
        value = inDoublePtr.value;
    }

    void* getPtr() override {
        return &value;
    }

    void setValPtr(void* ptr) override {
        value = *((double*)(ptr));
    }
};


class VariableRE : public AbstractDataContainer, public RuleEngineInputUnits {
    Variable *variable;

    bool added = false;

public:
    std::string name, frameCount;
    VariableRE(Variable *variable) : AbstractDataContainer(&value) {
        this->variable = variable;

        id = variable->id;
        name = variable->name;
        double  val = variable->value;
        if(std::isnan(val)) {
            val = 0;
        }
        value.value = val;
        frameCount = variable->frameCount;

    }

    void destroy() override {

    }

    void setFields(std::unordered_map<std::string, RuleEngineInputUnits *> *map) override {

    }

    void process() override {

    }

    std::string getId() override {
        return id;
    }

    DoublePtr value;
};

class ConstantRE : public AbstractDataContainer, public RuleEngineInputUnits {
public:
    DoublePtr doublePtr;
    ConstantRE(Constant* constant) : AbstractDataContainer(&doublePtr), doublePtr(constant->value)
    { }

    void destroy() {
    }

    void setFields(std::unordered_map<std::string, RuleEngineInputUnits *> *map) override {

    }

    std::string getId() override {
        return id;
    }

    void process() override {
    }
};
#endif //NATIVE_VARIABLERE_H
