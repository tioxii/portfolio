#include "shipFunc.h"

#ifndef PGSIZE
#define PGSIZE 100
#endif // PGSIZE
#include "shipStructs.h"

void isItPlaceable(playGround* player1) {
    for(int i = 0; i < PGSIZE; i++) {
        player1[i].placeable = true;
    }


    for(int i = 0; i < PGSIZE; i++) {
        if(player1[i].ship) {
            player1[i].placeable = false;
            if(i - 10 > -1)
                player1[i - 10].placeable = false;
            if(i + 10 < 100)
                player1[i + 10].placeable = false;
            if((i - 1)/10 == i/10)
                player1[i - 1].placeable = false;
            if((i + 1)/10 == i/10)
                player1[i + 1].placeable = false;

        }
    }
}

int checkShip(playGround* player1, ships* normalShips) {
    int a;

    if(!(normalShips->battleship)) {
        if(!(normalShips->cruiser)) {
            if(!(normalShips->destroyer)) {
                if(!(normalShips->submarine)) {
                    normalShips->submarine--;
                    a = -1;
                } else {
                    normalShips->submarine--;
                    a = 2;
                }
            } else {
                normalShips->destroyer--;
                a = 3;
            }
        } else {
            normalShips->cruiser--;
            a = 4;
        }
    } else {
        normalShips->battleship--;
        a = 5;
    }

    return a;
}
