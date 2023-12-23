// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Net/UnrealNetwork.h"
#include "InventoryItem.h"
#include "Engine/DataTable.h"
#include "ItemInventoryComponent.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE(FRefreshInventory);

UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent), Blueprintable, BlueprintType )
class INVENTORYSYSTEM_API UItemInventoryComponent : public UActorComponent
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this component's properties
	UItemInventoryComponent();

protected:
	// Called when the game starts
	virtual void BeginPlay() override;

	virtual void BeginDestroy() override;

	virtual void GetLifetimeReplicatedProps(TArray<FLifetimeProperty>& OutLifetimeProps) const override;

	UFUNCTION()
	void OnRep_Items();

	virtual void UpdateVariablesAndNotifyChanges();

	void UpdateWeight();

public:	
	// Called every frame
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

	/*
	* Inventory Functions
	*/
	UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category = "Inventory")
	bool CanAddItem(const class UInventoryItem* item) const;

	UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category = "Inventory")
	bool AddItem(class UInventoryItem* item);

	UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category = "Inventory")
	bool CanRemoveItem(const class UInventoryItem* item) const;

	UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category = "Inventory")
	bool RemoveItem(class UInventoryItem* item);

	UFUNCTION(BlueprintCallable, Category = "Inventory")
	bool ExistsID(const FName itemID) const;

	UFUNCTION(BlueprintCallable, Category = "Inventory")
	class UInventoryItem* GetItemByID(const FName itemID) const;

	/*
	* Inventory Variables
	*/
	UPROPERTY(BlueprintReadOnly, Category = "Inventory", ReplicatedUsing=OnRep_Items, meta = (AllowPrivateAccess = "true"))
	TArray<class UInventoryItem*> InventoryItems;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Inventory", meta = (AllowPrivateAccess = "true"))
	class UDataTable* ItemDataTable;

	UPROPERTY(BlueprintAssignable, Category = "Inventory")
	FRefreshInventory RefreshInventory;

	/*
	* Number of different Items in Inventory
	*/
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Replicated, Category = "Inventory")
	int32 MaxInventorySize = 100;

	UPROPERTY(BlueprintReadWrite, Replicated, Category = "Inventory")
	int32 CurrentInventorySize = 0;
	
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Inventory")
	bool bUseWeight = false;

	UPROPERTY(EditAnywhere, Replicated, BlueprintReadOnly, Category = "Inventory", meta = (EditCondition = "bUseWeight"))
	float MaxInventoryWeight = 100.0f;

	UPROPERTY(BlueprintReadWrite, Replicated, Category = "Inventory", meta = (EditCondition = "bUseWeight"))
	float CurrentInventoryWeight = 0.0f;

	/*
	* Max Stack Size
	*/
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Inventory")
	bool bUseMaxStackSize = false;
};
