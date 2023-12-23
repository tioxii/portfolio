#include "SDLship.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <SDL.h>
#include <time.h>
#include "shipOptions.h"
#include "shipStructs.h"
#include "saveGame.h"
#include "shipAI.h"
#include "shipFunc.h"

#ifndef PGSIZE
#define PGSIZE 100
#endif // PGSIZE

options op;

void resetShipsPlayer(playGround* player2) {
    for(int i = 0; i < PGSIZE; i++) {
        player2[i].ship = false;
    }
}

bool setShipsPlayer(SDL_Renderer* renderer, playGround* player1, playGround* player2, int spot, int sL, ships* pl1) {
    bool pg = true, running = true;
    bool up = false, right = false, down = false, left = false;
    SDL_Event event;
    wT p1 = {true, true, true, true};


    for(int i = 0; i < sL; i++) {
        //Up
        if(spot - i*10 > -1) {
            if(!(player2[spot - i*10].placeable))
                p1.up = false;
        } else {
            p1.up = false;
        }
        //Down
        if(spot + i*10 < 100) {
            if(!(player2[spot + i*10].placeable))
                p1.down = false;
        } else {
            p1.down = false;
        }
        //Right
        if((spot + i)/10 == spot/10) {
            if(!(player2[spot + i].placeable))
                p1.right = false;
        } else {
            p1.right = false;
        }
        //Left
        if((spot - i)/10 == spot/10) {
            if(!(player2[spot - i].placeable))
                p1.left = false;
        } else {
            p1.left = false;
        }
    }

    //Testing if there is a possible way
    if(!(p1.up || p1.right || p1.down || p1.left)) {
        running = false;
        if(sL == 5)
            pl1->battleship++;
        if(sL == 4)
            pl1->cruiser++;
        if(sL == 3)
            pl1->destroyer++;
        if(sL == 2)
            pl1->submarine++;
    }
    //Marking the possible ways
    if(p1.up) {
        for(int i = 0; i < sL; i++)
            player2[spot - 10*i].marked = true;
    }
    if(p1.right) {
        for(int i = 0; i < sL; i++)
            player2[spot + i].marked = true;
    }
    if(p1.down) {
        for(int i = 0; i < sL; i++)
            player2[spot + 10*i].marked = true;
    }
    if(p1.left) {
        for(int i = 0; i < sL; i++)
            player2[spot - i].marked = true;
    }

    while(running) {
        while(SDL_PollEvent(&event)) {
            if(event.type == SDL_QUIT) {
                running = false;
                pg = false;
            }
            if(event.type == SDL_KEYDOWN) {
                const char* key = SDL_GetKeyName(event.key.keysym.sym);
                printf(" SetShip: %s \n", key);

                if(strcmp(key, "L") == 0) {
                    running = false;
                }
                if(strcmp(key, "Escape") == 0) {
                    running = false;
                    pg = false;
                }
                if(strcmp(key, op.up) == 0 && p1.up) {
                    up = true;
                    running = false;
                }
                if(strcmp(key, op.right) == 0 && p1.right) {
                    right = true;
                    running = false;
                }
                if(strcmp(key, op.down) == 0 && p1.down) {
                    down = true;
                    running = false;
                }
                if(strcmp(key, op.left) == 0 && p1.left) {
                    left = true;
                    running = false;
                }
            }
        }

        for(int j = 0; j < PGSIZE; j++) {
            //Player1
            if(player1[j].shot) {
                if(player1[j].ship) {
                    SDL_SetRenderDrawColor(renderer, 0, 0, 180, 255);
                    SDL_RenderFillRect(renderer, &player1[j].rect);
                } else {
                    SDL_SetRenderDrawColor(renderer, 69, 79, 255, 180);
                    SDL_RenderFillRect(renderer, &player1[j].rect);
                }
            } else {
                SDL_SetRenderDrawColor(renderer, 55, 55, 55, 25);
                SDL_RenderFillRect(renderer, &player1[j].rect);
            }
            //Player2
            if(player2[j].ship) {
                SDL_SetRenderDrawColor(renderer, 25, 25, 25, 180);
                SDL_RenderFillRect(renderer, &player2[j].rect);
            } else {
                if(player2[j].marked) {
                    SDL_SetRenderDrawColor(renderer, 255, 255, 0, 255);
                    SDL_RenderFillRect(renderer, &player2[j].rect);
                } else {
                SDL_SetRenderDrawColor(renderer, 69, 79, 255, 180);
                SDL_RenderFillRect(renderer, &player2[j].rect);
                }
            }


        }
        //Show Renderer
        SDL_RenderPresent(renderer);
    }

    if(up) {
        for(int i = 0; i < sL; i++)
            player2[spot - 10*i].ship = true;
    }
    if(right) {
        for(int i = 0; i < sL; i++)
            player2[spot + i].ship = true;
    }
    if(down) {
        for(int i = 0; i < sL; i++)
            player2[spot + 10*i].ship = true;
    }
    if(left) {
        for(int i = 0; i < sL; i++)
            player2[spot - i].ship = true;
    }

    for(int i = 0; i < PGSIZE; i++) {
        player2[i].marked = false;
    }

    isItPlaceable(player2);

    return pg;
}

bool setShips(playGround* player1, playGround* player2, SDL_Window* window, SDL_Renderer* renderer) {
    bool running = true;
    SDL_Event event;
    SDL_Rect sel;
    int i = 0, sL;
    bool pg = true, pg2 = true;
    ships p1 = {1, 2, 3, 4};

    SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);

    while(running) {
        while(SDL_PollEvent(&event)) {
            if(event.type == SDL_QUIT) {
                running = false;
                pg = false;
            }
            if(event.type == SDL_KEYDOWN) {
                const char* key = SDL_GetKeyName(event.key.keysym.sym);
                printf(" SetShip: %s \n", key);

                if(strcmp(key, "S") == 0) {
                    running = false;
                }
                if(strcmp(key, "Escape") == 0) {
                    running = false;
                    pg = false;
                }
                if(strcmp(key, op.left) == 0) {
                    i--;
                    if((i + 1) % 10 == 0)
                        i += 10;
                }
                if(strcmp(key, op.right) == 0) {
                    i++;
                    if(i % 10 == 0)
                        i -= 10;
                }
                if(strcmp(key, op.down) == 0) {
                    i += 10;
                    if(i > 99)
                        i -= 100;
                }
                if(strcmp(key, op.up) == 0) {
                    i -= 10;
                    if(i < 0)
                        i += 100;
                }
                if(strcmp(key, "Z") == 0) {
                    resetShipsPlayer(player2);
                    p1.battleship = 1;
                    p1.cruiser = 2;
                    p1.destroyer = 3;
                    p1.submarine = 4;
                    setShipsAI(player2, &p1);
                    running = false;
                }
                if(strcmp(key, "R") == 0) {
                    resetShipsPlayer(player2);
                    p1.battleship = 1;
                    p1.cruiser = 2;
                    p1.destroyer = 3;
                    p1.submarine = 4;
                    isItPlaceable(player2);
                }
                if(strcmp(key, op.confirm) == 0) {
                    sL = checkShip(player2, &p1);
                    if(sL == -1)
                        running = false;
                    pg2 = setShipsPlayer(renderer, player1, player2, i, sL, &p1);
                    running = pg2;
                    pg = pg2;
                }
            }
        }
        if(p1.submarine == 0)
            running = false;

        //Renderer
        SDL_SetRenderDrawColor(renderer, 0, 0, 255, 180);
        SDL_RenderClear(renderer);
        //Draw & Check

        for(int j = 0; j < PGSIZE; j++) {
            //Player1
            if(player1[j].shot) {
                if(player1[j].ship) {
                    SDL_SetRenderDrawColor(renderer, 0, 0, 180, 255);
                    SDL_RenderFillRect(renderer, &player1[j].rect);
                } else {
                    SDL_SetRenderDrawColor(renderer, 69, 79, 255, 180);
                    SDL_RenderFillRect(renderer, &player1[j].rect);
                }
            } else {
                SDL_SetRenderDrawColor(renderer, 55, 55, 55, 25);
                SDL_RenderFillRect(renderer, &player1[j].rect);
            }
            //Player2
            if(player2[j].ship) {
                SDL_SetRenderDrawColor(renderer, 25, 25, 25, 180);
                SDL_RenderFillRect(renderer, &player2[j].rect);
            } else {
                SDL_SetRenderDrawColor(renderer, 69, 79, 255, 180);
                SDL_RenderFillRect(renderer, &player2[j].rect);
                }

            if(i == j) {
                sel.x = player2[j].rect.x;
                sel.y = player2[j].rect.y;
                sel.w = player2[j].rect.w;
                sel.h = player2[j].rect.h;

                SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255);
                SDL_RenderDrawRect(renderer, &sel);
            }
        }
        //Show Renderer
        SDL_RenderPresent(renderer);
    }
    return pg;
}

int randomAI(playGround* player2) {
    int a,b,c;
    int k = 0;
    uint32_t* storage;

    printf(" turnAI: ");

    //Looks how many squares are unknown to the AI
    for(int i = 0; i < PGSIZE; i++) {
        if(!(player2[i].shot))
            k++;
    }
    printf(" %d",k);

    //for security
    if(k == 0)
        return(-1);

    //Creates an Array where all unknown squares are getting stored
    storage = (uint32_t*) malloc(k*sizeof(uint32_t));
    if(storage == NULL) {
        printf(" Fehler beim Speicher \n ");
        return 0;
    }
    b = 0;

    for(int i = 0; i < PGSIZE; i++) {
        if(!(player2[i].shot)) {
            storage[b] = i;
            b++;
        }
    }
    printf(" %d", b);

    //Creates a random value between -1 < c < 100 and takes the corresponding Element from the Array
    c = rand() % k;
    printf(" %d", c);

    a = storage[c];
    printf(" %d\n", a);

    if(player2[a].shot)
        printf(" Fehler! bei random! \n");


    free(storage);
    return a;
}

void playGame(playGround* player1, playGround* player2, SDL_Window* window, SDL_Renderer* renderer) {
    bool running = true;
    bool turnPlayer;
    SDL_Event event;
    SDL_Rect sel;
    int i = 0;
    int a;
    int countP1, countP2;

    turnPlayer = rand() % 2;

    while(running) {
        while(SDL_PollEvent(&event)) {
            if(event.type == SDL_QUIT) {
                running = false;
            }
            if(event.type == SDL_KEYDOWN) {
                const char* key = SDL_GetKeyName(event.key.keysym.sym);
                printf(" PlayGame: %s \n", key);

                if(strcmp(key, "Escape") == 0) {
                    running = false;
                }
                if(turnPlayer) {
                    if(strcmp(key, op.left) == 0) {
                        i--;
                        if((i + 1) % 10 == 0)
                            i += 10;
                    }
                    if(strcmp(key, op.right) == 0) {
                        i++;
                        if(i % 10 == 0)
                            i -= 10;
                    }
                    if(strcmp(key, op.down) == 0) {
                        i += 10;
                        if(i > 99)
                            i -= 100;
                    }
                    if(strcmp(key, op.up) == 0) {
                        i -= 10;
                        if(i < 0)
                            i += 100;
                    }
                    if(strcmp(key, op.confirm) == 0) {
                        if(!(player1[i].shot)) {
                            player1[i].shot = true;
                            if(!(player1[i].ship))
                                turnPlayer = false;
                        }
                    }
                    if(strcmp(key, "S") == 0) {
                        saveGames(player1, player2);
                    }
                    if(strcmp(key, "V") == 0) {
                        for(int l = 0; l < PGSIZE; l++)
                            player1[l].shot = true;
                    }
                }
            }
        }

        //Renderer
        SDL_SetRenderDrawColor(renderer, 0, 0, 255, 180);
        SDL_RenderClear(renderer);
        //Draw & Check
        countP1 = 0;
        countP2 = 0;

        for(int j = 0; j < PGSIZE; j++) {
            //Player1
            if(player1[j].shot) {
                if((player1[j].ship)) {
                    SDL_SetRenderDrawColor(renderer, 25, 25, 25, 180);
                    SDL_RenderFillRect(renderer, &player1[j].rect);
                    countP1 += 1;
                } else {
                    SDL_SetRenderDrawColor(renderer, 69, 79, 255, 180);
                    SDL_RenderFillRect(renderer, &player1[j].rect);
                    /*if(!(player1[j].placeable)) {
                        SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
                        SDL_RenderFillRect(renderer, &player1[j].rect);
                    } */
                }
            } else {
                SDL_SetRenderDrawColor(renderer, 55, 55, 55, 25);
                SDL_RenderFillRect(renderer, &player1[j].rect);
            }
            if(i == j) {
                sel.x = player1[j].rect.x;
                sel.y = player1[j].rect.y;
                sel.w = player1[j].rect.w;
                sel.h = player1[j].rect.h;

                SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255);
                SDL_RenderDrawRect(renderer, &sel);
            }
            //Player2
            if(player2[j].shot) {
                if(player2[j].ship) {
                    SDL_SetRenderDrawColor(renderer, 25, 25, 25, 180);
                    SDL_RenderFillRect(renderer, &player2[j].rect);
                    countP2 += 1;
                } else {
                    SDL_SetRenderDrawColor(renderer, 69, 79, 255, 180);
                    SDL_RenderFillRect(renderer, &player2[j].rect);
                }
                SDL_SetRenderDrawColor(renderer, 255,0,0,255);
                SDL_RenderDrawLine(renderer, player2[j].rect.x, player2[j].rect.y, player2[j].rect.x + 29, player2[j].rect.y + 29);
                SDL_RenderDrawLine(renderer, player2[j].rect.x + 29, player2[j].rect.y, player2[j].rect.x, player2[j].rect.y + 29);
            } else {
                if(player2[j].ship) {
                    SDL_SetRenderDrawColor(renderer, 25, 25, 25, 180);
                    SDL_RenderFillRect(renderer, &player2[j].rect);
                } else {
                    SDL_SetRenderDrawColor(renderer, 69, 79, 255, 180);
                    SDL_RenderFillRect(renderer, &player2[j].rect);
                }
            }
        }
        //Show Renderer
        SDL_RenderPresent(renderer);

        //Computer
        if(!(turnPlayer)) {
            a = randomAI(player2);
            if(a == -1) {
                running = false;
                printf("PlayGame: Whoops! That was an unexpected Ending.\n");
            }
            player2[a].shot = true;
            if(!(player2[a].ship))
                turnPlayer = true;
        }
        //Game End?
        /*if(countP1 == 30) {
            running = false;
            printf(" Game End: You Win!\n");
        }
        if(countP2 == 30) {
            running = false;
            printf(" Game End; You Lost!\n");
        }*/
    }
}

void startGame(SDL_Renderer* renderer, SDL_Window* window) {
    bool pg,pg1;
    ships normalShips = {1, 2, 3, 4};

    //Gets Memory for Arrays
    playGround* player1 = (playGround*) malloc(PGSIZE*sizeof(playGround));
    playGround* player2 = (playGround*) malloc(PGSIZE*sizeof(playGround));

    if(player1 != NULL && player2 != NULL) {
        printf(" Memory allocated for player 1 & player 2\n");
    }
    //Initializes rand()-function
    srand(time(NULL));

    //Initializes allocated Arrays
    for(int i = 0; i < PGSIZE; i++) {
        player1[i].shot = false;
        player1[i].ship = false;
        player1[i].marked = false;
        player1[i].placeable = true;
        player1[i].rect.x = (i%10)*30 + (i%10)*10 + 60;
        player1[i].rect.y = (i/10)*30 + (i/10)*10 + 10;
        player1[i].rect.h = 30;
        player1[i].rect.w = 30;

        player2[i].shot = false;
        player2[i].ship = false;
        player2[i].marked = false;
        player2[i].placeable = true;
        player2[i].rect.x = (i%10)*30 + (i%10)*10 + 550;
        player2[i].rect.y = (i/10)*30 + (i/10)*10 + 10;
        player2[i].rect.h = 30;
        player2[i].rect.w = 30;
    }

    pg1 = setShipsAI(player1, &normalShips);
    pg = setShips(player1, player2, window, renderer);
    if(pg && pg1)
        playGame(player1, player2, window, renderer);


    //Release Structure Arrays
    free(player1);
    free(player2);
}
