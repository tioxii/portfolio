// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "ItemInventoryComponent.h"
#include "InventoryWidget.h"
#include "InputMappingContext.h"
#include "InventoryItem.h"
#include "PlayerInventoryComponent.generated.h"

USTRUCT(BlueprintType)
struct FEquipmentSlot
{
	GENERATED_BODY()

public:
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Equipment")
	FName SlotName;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Equipment")
	TSubclassOf<class UInventoryItem> ItemClass;

	UPROPERTY(BlueprintReadWrite, Category = "Equipment")
	UInventoryItem* Item;
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FEquipmentSlotChanged, FName, SlotName);

DECLARE_DYNAMIC_MULTICAST_DELEGATE(FRefreshEquipment);

/**
 * PlayerInventoryComponent
 */
UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class INVENTORYSYSTEM_API UPlayerInventoryComponent : public UItemInventoryComponent
{
	GENERATED_BODY()
private:
	
	// UI
	UPROPERTY(EditAnywhere, Category = "UI", meta = (AllowPrivateAccess = "true"))
	TSubclassOf<class UInventoryWidget> InventoryWidgetClass;

	class UInventoryWidget* inventoryWidget;

	// Equipment Slots
	UPROPERTY(ReplicatedUsing=OnRep_Equipment, EditAnywhere, BlueprintReadOnly, Category = "Equipment", meta = (AllowPrivateAccess = "true"))
	TArray<FEquipmentSlot> EquipmentSlots;

protected:

	virtual void BeginPlay() override;

	virtual void BeginDestroy() override;

	UFUNCTION(BlueprintCallable, Category = "Equipment")
	TArray<UInventoryItem*> GetEquipedItems() const;

	UFUNCTION(BlueprintCallable, Category = "Equipment")
	UInventoryItem* GetEquipedItemBySlotName(FName slot) const;

	UFUNCTION()
	void OnRep_Equipment();

	virtual void GetLifetimeReplicatedProps(TArray<FLifetimeProperty>& OutLifetimeProps) const override;

public:

	// InventoryItem Functions
	UFUNCTION(Server, Reliable, WithValidation, BlueprintCallable, Category = "Inventory")
	void TransferInventoryItem(UInventoryItem* InventoryItem, UItemInventoryComponent* From, UItemInventoryComponent* To);

	UFUNCTION(Server, Reliable, BlueprintCallable, Category = "Equipment")
	void TransferInventoryItemToEquipment(UInventoryItem* inventoryItem, FName slot);

	UFUNCTION(Server, Reliable, BlueprintCallable, Category = "Equipment")
	void TransferEquipmentItemToInventory(FName Slot);

	UFUNCTION(Server, Reliable, BlueprintCallable, Category = "Inventory")
	void DropItem(UInventoryItem* Item, UItemInventoryComponent* From);

	UPROPERTY(BlueprintAssignable, Category = "Equipment")
	FEquipmentSlotChanged OnEquipmentSlotChanged;

	UPROPERTY(BlueprintAssignable, Category = "Equipment")
	FRefreshEquipment OnRefreshEquipment;

	// UI
	UFUNCTION(BlueprintCallable, Category = "UI")
	void OpenInventory();
};
