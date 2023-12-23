#define _CRT_SECURE_NO_WARNINGS

#include "saveGame.h"

#include <stdio.h>

void saveGames(playGround* player1, playGround* player2) {
    FILE* fp;

    fp = fopen("saveGame.txt", "w");

    if(fp == NULL)
        return;

    fprintf(fp, "Player 1:\n");

    //ships.shot
    for(int i = 0; i < 100; i++) {
        if(i % 10 == 0)
            fprintf(fp, "\n");
        if(player1[i].ship)
            fprintf(fp, "1 ");
        else
            fprintf(fp, "0 ");
    }

    fprintf(fp, "\n");

    for(int i = 0; i < 100; i++) {
        if(i % 10 == 0)
            fprintf(fp, "\n");
        if(player1[i].shot)
            fprintf(fp, "1 ");
        else
            fprintf(fp, "0 ");
    }

    fprintf(fp, "\n");
    fprintf(fp, "\n");

    fprintf(fp, "Player 2:\n");

    for(int i = 0; i < 100; i++) {
        if(i % 10 == 0)
            fprintf(fp, "\n");
        if(player1[i].ship)
            fprintf(fp, "1 ");
        else
            fprintf(fp, "0 ");
    }

    fprintf(fp, "\n");

    for(int i = 0; i < 100; i++) {
        if(i % 10 == 0)
            fprintf(fp, "\n");
        if(player1[i].shot)
            fprintf(fp, "1 ");
        else
            fprintf(fp, "0 ");
    }

    fclose(fp);
}
/*void loadGame(playGround* player1, playGround player2) {
    FILE* fp;

    fp = fopen("saveGame.txt", "r");

}*/
