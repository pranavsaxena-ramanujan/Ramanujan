//
// Created by pranav on 28/3/24.
//

#include "FunctionCommandRE.h"
#include "dataContainer/ArrayRE.h"
#include "dataContainer/VariableRE.h"
#include "DebugPoint.h"
#include <list>

#include <limits>
#include <random>

FunctionCommandRE::FunctionCommandRE(FunctionCall* functionCommand, FunctionCallRE* functionInfo) {
    this->functionCommandInfo = functionCommand;
    this->functionInfoRE = functionInfo;
}

/*
 * How will we process it?
 * 1. functionInfo has the information about the function. It has the arguments and the first command.
 * 2. it has the allVariablesInMethod.
 * 3. when the function has to be started, we would have to put the values to the arg variables only, and other as 0.0
 * 4. when the function is popped, we would have to transfer the value to calling method variables, and the de-set the args in the current function.
 *    This step is to be done with care, as in the case of recursive function, the value of the variable should not be lost.
 */

void FunctionCommandRE::setFields(std::unordered_map<std::string, RuleEngineInputUnits *> *map) {
    functionInfoRE->setFields(map);

    argSize = functionCommandInfo->argumentsSize;

    firstCommand = functionInfoRE->commmandRe;
    if (firstCommand == nullptr) {
        firstCommand = dynamic_cast<CommandRE *>(getFromMap(map, functionInfoRE->functionCall->firstCommandId));
    }

    /*
     * Use unified DataContainerValue approach for both variables and arrays.
     */

    std::list<DataContainerValue*> methodCalledContainersList;
    std::list<DataContainerValue*> methodCallingContainersList;

    for(int i = 0; i < functionInfoRE->argSize; i++) {
        // Get DataContainerValue from called function arguments
        AbstractDataContainer* calledArg = (functionInfoRE->arguments[i]);
        DataContainerValue* calledContainer = calledArg->getVal();

        // Get DataContainerValue from calling function arguments  
        AbstractDataContainer* callingArg =
                (AbstractDataContainer*)(map->at(functionCommandInfo->arguments[i]));
        DataContainerValue* callingContainer = callingArg->getVal();
        
        if (calledContainer && callingContainer) {
            methodCalledContainersList.push_back(calledContainer);
            methodCallingContainersList.push_back(callingContainer);
            
            // Track array name mappings for debugging
            #ifdef DEBUG_BUILD
            ArrayRE* calledArrayRE = dynamic_cast<ArrayRE*>(functionInfoRE->arguments[i]);
            ArrayRE* callingArrayRE = dynamic_cast<ArrayRE*>(map->at(functionCommandInfo->arguments[i]));
            if (calledArrayRE && callingArrayRE) {
                arrayNameMethodMap.insert(std::make_pair(callingArrayRE->name, calledArrayRE->name));
            }
            #endif
        }
    }

    // Store containers for parameter handling
    methodCalledContainers = new DataContainerValue*[argSize];
    methodCallingContainers = new DataContainerValue*[argSize];

    for(int i = 0; i < argSize; i++) {
        methodCalledContainers[i] = methodCalledContainersList.front();
        methodCalledContainersList.pop_front();
        
        methodCallingContainers[i] = methodCallingContainersList.front();
        methodCallingContainersList.pop_front();
    }

    // Handle all variables in method (parameters + locals)
    totalContainerCount = functionInfoRE->functionCall->allVariablesInMethodSize;
    methodAllContainers = new DataContainerValue*[totalContainerCount];
    methodAllContainerCurrentVals = new DataContainerValue*[totalContainerCount];

    for(int i = 0; i < totalContainerCount; i++) {
        AbstractDataContainer* container = (functionInfoRE->allVariablesInMethod[i]);
        methodAllContainers[i] = container->getVal();
    }

    // Store stack values for restoration
    containerStackCurrent = new DataContainerValue*[argSize];
}

void FunctionCommandRE::process() {
    // ==================== DEBUG SETUP ====================
#ifdef DEBUG_BUILD
    // Get debug point for tracking function call execution
    std::shared_ptr<DebugPoint> debugPoint = debugger->getDebugPointToBeCommitted();
    
    // Record array name mappings for debugging purposes
    // This helps track which calling arrays correspond to which function parameters
    for(auto it = arrayNameMethodMap.begin(); it != arrayNameMethodMap.end(); it++) {
        debugPoint->addArrayInFuncCall(it->first, it->second);
    }
#endif

    // ==================== PHASE 1: PARAMETER SETUP AND CONTAINER CONTEXT SAVING ====================
    
    /*
     * For each parameter passed to the function:
     * 1. Save the current value of the called function's parameter container (for stack restoration)
     * 2. Copy the calling function's argument value to the called function's parameter
     * 3. Save the current state of the parameter container for later restoration
     */
    for (int i = 0; i < argSize; i++) {
#ifdef DEBUG_BUILD
        // Record the current function container value for debugging (if it's a variable)
        VariableRE* varContainer = dynamic_cast<VariableRE*>(methodCallingContainers[i]);
        if (varContainer) {
            debugPoint->addCurrentFuncVal(static_cast<DoublePtr*>(varContainer->getVal())->value);
        }
#endif
        // Save the current value of the parameter in the called function (stack save)
        containerStackCurrent[i] = methodCalledContainers[i];



        methodCalledContainers[i]->setValPtr(methodCallingContainers[i]->getPtr());
        
        // Copy the argument value from calling context to called function parameter
        // This will be handled through the unified DataContainerValue interface
        
        // Save the current value of the parameter container for restoration after function execution
        methodAllContainerCurrentVals[i] = methodAllContainers[i];
    }

#ifdef DEBUG_BUILD
    debugger->commitDebugPoint();
#endif

    /*
     * For local containers in the function (non-parameters):
     * Save their current values so they can be restored after function execution.
     * These containers start from index argSize (after the parameters).
     */
    for(int i = argSize; i < totalContainerCount; i++) {
        methodAllContainerCurrentVals[i] = methodAllContainers[i];
    }

    // ==================== PHASE 2: FUNCTION BODY EXECUTION ====================
    
    /*
     * Execute the function body by traversing the command chain.
     * Each command returns the next command to execute, forming a linked execution chain.
     * Execution continues until there are no more commands (nullptr).
     */
    CommandRE* command = firstCommand;
    while(command != nullptr) {
        command = command->get();  // Execute current command and get next command
    }

    // ==================== PHASE 3: CONTEXT RESTORATION AND CLEANUP ====================
    
    /*
     * CRITICAL: We must restore the calling context regardless of function execution outcome.
     * This is essential for recursive functions and proper stack management.
     * 
     * Example scenario requiring restoration:
     * func(a,b) {
     *      c = a + 1
     *      func(c,b)    // Recursive call modifies 'a'
     *      d = a + 1    // Without restoration, 'a' would have value of 'c'
     * }
     */

    // First, update our stack copies with the final values from the called function context
    for(int i = 0; i < argSize; i++) {
        containerStackCurrent[i] = methodCalledContainers[i];
    }

    // ==================== PHASE 4: CONTAINER RESTORATION ====================
    
    /*
     * DETAILED CONTAINER RESTORATION EXPLANATION:
     * 
     * The restoration process is critical for maintaining proper function call semantics,
     * especially for recursive functions. Using the unified DataContainerValue approach
     * simplifies this process by treating variables and arrays uniformly through the
     * common DataContainerValue interface, eliminating the need for type-specific handling.
     */
    
    /*
     * Parameter Container Restoration Loop:
     * For each parameter container (i = 0 to argSize-1):
     */
    for(int i = 0; i < argSize; i++) {
        /*
         * Step 1: Restore the function parameter to its saved state
         * Use the unified DataContainerValue approach
         */
        DataContainerValue* calledContainer = methodCalledContainers[i];
        DataContainerValue* savedVal = methodAllContainerCurrentVals[i];
        methodCalledContainers[i]->setValPtr(savedVal->getPtr());
        
        /*
         * Step 2: Restore the calling context container to its stack-saved value
         */
        DataContainerValue* callingContainer = methodCallingContainers[i];
        DataContainerValue* stackVal = containerStackCurrent[i];
        if (stackVal) {
            callingContainer->setValPtr(stackVal->getPtr());
        }
    }
    
    /*
     * Local Container Restoration Loop:
     * For local containers (non-parameters) (i = argSize to totalContainerCount-1):
     * 
     * WHAT HAPPENS:
     * These are containers declared within the function that are not parameters.
     * 
     * WHY NEEDED:
     * If a function calls itself recursively, local containers need to be restored
     * to their pre-call state to prevent interference between recursive calls.
     */
    for(int i = argSize; i < totalContainerCount; i++) {
        /*
         * Restore local container to its pre-function-execution value
         * Use the unified DataContainerValue approach
         */
        DataContainerValue* localContainer = methodAllContainers[i];
        DataContainerValue* savedVal = methodAllContainerCurrentVals[i];
        
        if (localContainer && savedVal) {
            localContainer->setValPtr((savedVal->getPtr()));
        }
    }

    // Function execution complete - calling context fully restored
}

void BuiltInFunctionsImpl::setFields(std::unordered_map<std::string, RuleEngineInputUnits *> *map) {
    // Call the base class unified setFields first
    FunctionCommandRE::setFields(map);
    
    // For backward compatibility, also populate the legacy arrays used by built-in functions
    // NOTE: Built-in functions still require type-specific handling for legacy interface compatibility
    std::list<double *> methodArgVariableAddrList;
    std::list<ArrayValue **> methodArgArrayAddrList;

    varCount = 0;
    arrCount = 0;

    for (int i = 0; i < functionCommandInfo->argumentsSize; i++) {
        auto arg = map->at(functionCommandInfo->arguments[i]);
        // Legacy type checking required for built-in function interface compatibility
        if (dynamic_cast<ArrayRE *>(arg) != nullptr) {
            arrCount++;
            ArrayRE* arrayRE = (ArrayRE*) arg;
            ArrayValue* arrayValuePtr = arrayRE->arrayValue->arrayValue;
            methodArgArrayAddrList.push_back(&arrayValuePtr);
        } else {
            varCount++;
            // Use unified DataContainerValue approach where possible
            AbstractDataContainer* container = (AbstractDataContainer*) arg;
            DoublePtr* doublePtr = static_cast<DoublePtr*>(container->getVal());
            methodArgVariableAddrList.push_back(&doublePtr->value);
        }
    }

    methodArgVariableAddr = new double *[varCount];
    methodArgArrayAddr = new ArrayValue **[arrCount];

    for (int i = 0; i < varCount; i++) {
        methodArgVariableAddr[i] = methodArgVariableAddrList.front();
        methodArgVariableAddrList.pop_front();
    }

    for (int i = 0; i < arrCount; i++) {
        methodArgArrayAddr[i] = methodArgArrayAddrList.front();
        methodArgArrayAddrList.pop_front();
    }
}

void NINF::process() {
    if(varCount == 1) {
        *methodArgVariableAddr[0] = -std::numeric_limits<double>::infinity();
    }

    if(arrCount == 1)
    {
        ArrayValue** arrayValue = methodArgArrayAddr[0];
        for(int i = 0; i < (*arrayValue)->totalSize; i++) {
            (*arrayValue)->val[i] = -std::numeric_limits<double>::infinity();
        }
    }
}

void PINF::process() {
    if(varCount == 1) {
        *methodArgVariableAddr[0] = std::numeric_limits<double>::infinity();
    }

    if(arrCount == 1)
    {
        ArrayValue** arrayValue = methodArgArrayAddr[0];
        for(int i = 0; i < (*arrayValue)->totalSize; i++) {
            (*arrayValue)->val[i] = std::numeric_limits<double>::infinity();
        }
    }
}

static std::random_device rd;  // Non-deterministic random seed
static std::mt19937 gen(rd()); // Mersenne Twister engine
static std::uniform_real_distribution<> dis(0.0, 1.0);

void RAND::process() {
    if(varCount == 1) {
        *methodArgVariableAddr[0] = dis(gen);
    }

    if(arrCount == 1)
    {
        ArrayValue** arrayValue = methodArgArrayAddr[0];
        for(int i = 0; i < (*arrayValue)->totalSize; i++) {
            (*arrayValue)->val[i] = dis(gen);
        }
    }
}

// All variable based built-in methods:
void ABS::process() {
    if(varCount == 1) {
        *methodArgVariableAddr[0] = std::abs(*methodArgVariableAddr[0]);
    }
}

void SIN::process() {
    if(varCount == 1) {
        *methodArgVariableAddr[0] = std::sin(*methodArgVariableAddr[0]);
    }
}

void COS::process() {
    if(varCount == 1) {
        *methodArgVariableAddr[0] = std::cos(*methodArgVariableAddr[0]);
    }
}

void TAN::process() {
    if(varCount == 1) {
        *methodArgVariableAddr[0] = std::tan(*methodArgVariableAddr[0]);
    }
}

void ASIN::process() {
    if(varCount == 1) {
        *methodArgVariableAddr[0] = std::asin(*methodArgVariableAddr[0]);
    }
}

void ACOS::process() {
    if(varCount == 1) {
        *methodArgVariableAddr[0] = std::acos(*methodArgVariableAddr[0]);
    }
}

void ATAN::process() {
    if(varCount == 1) {
        *methodArgVariableAddr[0] = std::atan(*methodArgVariableAddr[0]);
    }
}

void FLOOR::process() {
    if(varCount == 1) {
        *methodArgVariableAddr[0] = std::floor(*methodArgVariableAddr[0]);
    }
}

void CEIL::process() {
    if(varCount == 1) {
        *methodArgVariableAddr[0] = std::ceil(*methodArgVariableAddr[0]);
    }
}

void EXP::process() {
    if(varCount == 1) {
        *methodArgVariableAddr[0] = std::exp(*methodArgVariableAddr[0]);
    }
}

void SQRT::process() {
    if(varCount == 1) {
        *methodArgVariableAddr[0] = std::sqrt(*methodArgVariableAddr[0]);
    }
}

void POW::process() {
    if(varCount == 2) {
        *methodArgVariableAddr[0] = std::pow(*methodArgVariableAddr[0], *methodArgVariableAddr[1]);
    }
}