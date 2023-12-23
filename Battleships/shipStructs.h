#ifndef SHIPSTRUCTS_H_INCLUDED
#define SHIPSTRUCTS_H_INCLUDED

#include <SDL.h>
#include <stdbool.h>

typedef struct {
    bool shot;
    bool ship;
    bool marked;
    bool placeable;
    SDL_Rect rect;
}playGround;

typedef struct {
    int battleship;
    int cruiser;
    int destroyer;
    int submarine;
}ships;

typedef struct {
    bool up;
    bool right;
    bool down;
    bool left;
}wT;

#endif // SHIPSTRUCTS_H_INCLUDED
