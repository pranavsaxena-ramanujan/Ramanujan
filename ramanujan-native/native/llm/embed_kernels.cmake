# Converts kernels.cl into a C char array (MSVC limits string literals, so no raw string).
file(READ "${INPUT}" HEX_CONTENT HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," BYTES "${HEX_CONTENT}")
file(WRITE "${OUTPUT}" "// Generated from kernels.cl; do not edit.\nstatic const char kKernelSource[] = {${BYTES}0x00};\n")
