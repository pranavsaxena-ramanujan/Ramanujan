#ifndef NATIVE_REDEFINEARRAYCOMMANDRE_H
#define NATIVE_REDEFINEARRAYCOMMANDRE_H

#include "RuleEngineInputUnits.hpp"
#include <vector>
#include <string>

class RedefineArrayCommandRE : public RuleEngineInputUnits {
public:
    std::string arrayId;
    std::vector<int> newDimensions;
    // Optionally, initial values can be added here

    RedefineArrayCommandRE(const std::string& arrayId, const std::vector<int>& newDimensions)
        : arrayId(arrayId), newDimensions(newDimensions) {}

    void process(); // To be implemented: logic to redefine the array in memory
};

#endif // NATIVE_REDEFINEARRAYCOMMANDRE_H
