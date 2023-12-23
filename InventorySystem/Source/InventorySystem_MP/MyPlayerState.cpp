// Fill out your copyright notice in the Description page of Project Settings.


#include "MyPlayerState.h"
#include "Net/UnrealNetwork.h"

AMyPlayerState::AMyPlayerState()
{
    InventoryComponent = CreateDefaultSubobject<UItemInventoryComponent>(TEXT("InventoryComponent"));
    bReplicates = true;
}

void AMyPlayerState::GetLifetimeReplicatedProps(TArray<FLifetimeProperty>& OutLifetimeProps) const
{
    Super::GetLifetimeReplicatedProps(OutLifetimeProps);
    
    DOREPLIFETIME_CONDITION(AMyPlayerState, InventoryComponent, COND_OwnerOnly);
}
