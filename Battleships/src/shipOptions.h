#ifndef SHIPOPTIONS_H_INCLUDED
#define SHIPOPTIONS_H_INCLUDED

#include <stdbool.h>
#include "shipStructs.h"

typedef struct {
    char up[20];
    char right[20];
    char down[20];
    char left[20];
    char confirm[20];
}options;

extern options op;

bool loadOptions();

#endif // SHIPOPTIONS_H_INCLUDED
