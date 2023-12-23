#include "shipAI.h"

#include <stdio.h>
#include <stdbool.h>

#ifndef PGSIZE
#define PGSIZE 100
#endif // PGSIZE

int placeShip(playGround* player1, int sL, int spot) {
    wT ai = {true, true, true, true};
    int g;

    //Tests every possibility of placing a ship in one spot;
    for(int i = 0; i < sL; i++) {
        //Up
        if(spot - i*10 > -1) {
            if(!(player1[spot - i*10].placeable))
                ai.up = false;
        } else {
            ai.up = false;
        }
        //Down
        if(spot + i*10 < 100) {
            if(!(player1[spot + i*10].placeable))
                ai.down = false;
        } else {
            ai.down = false;
        }
        //Right
        if((spot + i)/10 == spot/10) {
            if(!(player1[spot + i].placeable))
                ai.right = false;
        } else {
            ai.right = false;
        }
        //Left
        if((spot - i)/10 == spot/10) {
            if(!(player1[spot - i].placeable))
                ai.left = false;
        } else {
            ai.left = false;
        }
    }
    printf(" Funktioniert bis hier hin\n");

    printf(ai.up ? " true" : " false");
    printf(ai.right ? " true" : " false");
    printf(ai.down ? " true" : " false");
    printf(ai.left ? " true" : " false");
    printf("\n");

    //Tests if there is a possible way
    if(!(ai.left || ai.up || ai.right || ai.down)) {
        player1[spot].placeable = false;
        return -1;
    }

    g = rand() % 4;

    printf(" %d\n", g);

    while(1) {
        g = g % 4;

        //Up
        if(g == 0) {
            if(ai.up) {
                for(int i = 0; i < sL; i++) {
                    player1[spot - 10*i].ship = true;
                }
                break;
            } else {
                g++;
            }
        }
        //Right
        if(g == 1) {
            if(ai.right) {
                for(int i = 0; i < sL; i++) {
                    player1[spot + i].ship = true;
                }
                break;

            } else {
                g++;
            }
        }
        //Down
        if(g == 2) {
            if(ai.down) {
                for(int i = 0; i < sL; i++) {
                    player1[spot + 10*i].ship = true;
                }
                break;

            } else {
                g++;
            }
        }
        //Left
        if(g == 3) {
            if(ai.left) {
                for(int i = 0; i < sL; i++) {
                    player1[spot - i].ship = true;
                }
                break;

            } else {
                g++;
            }
        }
    }

    return 0;
}

int randomShip(playGround* player1) {
    int a = 0;
    int b,c,d;
    uint32_t* storage;

    printf(" RandomShip:");

    for(int i = 0; i < PGSIZE; i++) {
        if(player1[i].placeable)
            a++;
    }

    printf(" %d", a);
    storage = (uint32_t*) malloc(a*sizeof(uint32_t));

    if(storage == NULL)
        printf("\n Error in randomShip()\n");

    b = 0;

    for(int i = 0; i < PGSIZE; i++) {
        if(player1[i].placeable) {
           storage[b] = i;
           b++;
        }
    }
    printf(" %d", b);

    c = rand() % a;

    printf(" %d", c);

    d = storage[c];
    printf(" %d\n", d);

    free(storage);
    return d;
}

bool setShipsAI(playGround* player1, ships* normalShips) {
    int spot;
    int sL;
    int b = 1;
    int n = 0;
    bool d;


    d = true;

    while(1) {
        spot = randomShip(player1);
        sL = checkShip(player1, normalShips);
        printf(" ShipLength: %d\n", sL);
        if(sL == -1)
            break;
        b = placeShip(player1, sL, spot);
        printf(" Rt-Value placeShip: %d\n", b);
        if(b == -1) {
            if(sL == 5)
                normalShips->battleship++;
            if(sL == 4)
                normalShips->cruiser++;
            if(sL == 3)
                normalShips->destroyer++;
            if(sL == 2)
                normalShips->submarine++;
        }
        isItPlaceable(player1);
        n++;
        if(n == 100) {
            printf(" Error with randomShipsAI\n");
            d = false;
            break;
        }
    }
    return d;
}
