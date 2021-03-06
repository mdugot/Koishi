#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <fstream>
#include <queue>
#include <vector>
#include <stack>
#include <map>
#include <unordered_map>
#include <set>
#include <list>
#include <string>
#include <iostream>
#include <exception>
#include <random>
#include <initializer_list>
#include <algorithm>
#include "color.h"
#define NPOS std::string::npos
#define NL "\n"
#define PRINT_ERROR true
#define ERROR if (PRINT_ERROR) std::cerr
#define PRINT_DEBUG true
#define DEBUG if (PRINT_DEBUG) std::cerr
#define PRINT_OUT true
#define OUT if (PRINT_OUT) std::cout
#define LIST std::list
#define STRING std::string
#define ABS(x) (x) < 0 ? -(x) : (x)
#define FLOAT double
#define STRING_TO_FLOAT std::stod

#endif
