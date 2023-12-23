#define _CRT_SECURE_NO_WARNINGS

#include "SDLship.h"
#include "shipOptions.h"
#include "saveGame.h"
#include "shipStructs.h"
#include <stdbool.h>
#include <stdio.h>
#include <SDL.h>

options op;

void mainMenu() {


}

int main(int argc, char* args[]) {
    bool a = false;
    SDL_Window* window;
    SDL_Renderer *renderer;

    //Initializes SDL
    if(SDL_Init(SDL_INIT_EVERYTHING) != 0)
        printf(" SDL_Init failed!\n");
    else
        printf(" SDL_Init was successful!\n");

    //Initializes window & renderer
    window = SDL_CreateWindow("hi", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 1000, 500, SDL_WINDOW_OPENGL);
    if(window == NULL)
        printf(" SDL_CreateWindow failed");
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    a = loadOptions();

    if(a)
        startGame(renderer, window);
    return 0;

    //Destroying window and renderer
    SDL_DestroyWindow(window);
    SDL_DestroyRenderer(renderer);

    //Quit SDL
    SDL_Quit();
}
