#define _CRT_SECURE_NO_WARNINGS

#include "shipOptions.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

bool loadOptions() {
    FILE* fp;
    char opti[10];

    fp = fopen("options.txt", "r");

    if(fp == NULL)
        return false;

    for(int i = 0; i < 5; i++) {
        fscanf(fp, "%s", opti);

        switch(i) {
            case 0: strcpy(op.up, opti); break;
            case 1: strcpy(op.right, opti); break;
            case 2: strcpy(op.down, opti); break;
            case 3: strcpy(op.left, opti); break;
            case 4: strcpy(op.confirm, opti); break;
        }
    }
    printf("%s", op.up);
    printf("%s", op.right);
    printf("%s", op.down);
    printf("%s", op.left);
    printf("%s", op.confirm);


    fclose(fp);
    return true;
}

