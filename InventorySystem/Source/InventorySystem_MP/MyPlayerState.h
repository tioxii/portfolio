// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/PlayerState.h"
#include "ItemInventoryComponent.h"
#include "MyPlayerState.generated.h"



/**
 * 
 */
UCLASS()
class INVENTORYSYSTEM_MP_API AMyPlayerState : public APlayerState
{
	GENERATED_BODY()
	
public:
	AMyPlayerState();

	void GetLifetimeReplicatedProps(TArray<FLifetimeProperty>& OutLifetimeProps) const override;

protected:
	UPROPERTY(Replicated)
 	class UItemInventoryComponent* InventoryComponent;

public:
	UFUNCTION(BlueprintCallable, Category = "Inventory")
	UItemInventoryComponent* GetInventoryComponent() const { return InventoryComponent; };
};
