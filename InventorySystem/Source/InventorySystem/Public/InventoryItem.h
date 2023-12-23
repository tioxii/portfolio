// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "Net/UnrealNetwork.h"
#include "Engine/DataTable.h"
#include "InventoryItem.generated.h"

USTRUCT(BlueprintType, Blueprintable)
struct FInventoryItemData : public FTableRowBase
{
	GENERATED_BODY()

public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Inventory")
	FString ItemDisplayName;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Inventory")
	FString ItemDescription;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Inventory")
	UTexture2D* ItemIcon;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Inventory")
	TSubclassOf<class UInventoryItem> ItemClass;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Inventory")
	int32 MaxQuantity = 64;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Inventory")
	bool bStackable = false;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Inventory")
	bool bConsumable = false;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Inventory")
	float Weight = 0.0f;
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE(FRefreshQuantity);

UCLASS( ClassGroup=(Custom), Blueprintable, BlueprintType)
class INVENTORYSYSTEM_API UInventoryItem : public UObject
{
	GENERATED_BODY()

public:

	virtual void GetLifetimeReplicatedProps(TArray<FLifetimeProperty>& OutLifetimeProps) const override;

	virtual bool IsSupportedForNetworking() const override;

	UFUNCTION(BlueprintNativeEvent, Category = "Inventory")
	void OnRep_Quantity();

	UFUNCTION(BlueprintNativeEvent, Category = "Inventory")
	void Copy(const UInventoryItem* item);

	UFUNCTION(BlueprintNativeEvent, Category = "Inventory")
	bool Equals(const UInventoryItem* item) const;

	UFUNCTION(BlueprintNativeEvent, Category = "Inventory")
	void AddQuantity(int32 amount);

	UFUNCTION(BlueprintNativeEvent, Category = "Inventory")
	void RemoveQuantity(int32 amount);

	UPROPERTY(BlueprintReadOnly, Category = "Inventory", Replicated)
	class UItemInventoryComponent* Inventory;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Inventory", Replicated, meta = (ExposeOnSpawn = true))
	FName ItemID;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Inventory", ReplicatedUsing=OnRep_Quantity, meta = (ExposeOnSpawn = true))
	int32 Quantity = 1;

	UPROPERTY(BlueprintAssignable, Category = "Inventory")
	FRefreshQuantity RefreshQuantity;
};
