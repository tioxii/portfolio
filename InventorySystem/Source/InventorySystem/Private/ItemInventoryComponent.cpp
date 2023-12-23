// Fill out your copyright notice in the Description page of Project Settings.


#include "ItemInventoryComponent.h"

// Sets default values for this component's properties
UItemInventoryComponent::UItemInventoryComponent()
{
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	PrimaryComponentTick.bCanEverTick = false;
	bReplicateUsingRegisteredSubObjectList = true;

#if WITH_EDITORONLY_DATA
	// Load ItemDataTable.
	if (ItemDataTable == nullptr) 
	{
		static ConstructorHelpers::FObjectFinder<UDataTable> ItemDataTableObject(TEXT("/Script/Engine.DataTable'/Game/Inventory/InventoryItemData.InventoryItemData'"));
		if (ItemDataTableObject.Succeeded()) 
		{
			ItemDataTable = ItemDataTableObject.Object;
		}
		else 
		{
			UE_LOG(LogTemp, Warning, TEXT("ItemDataTable not found!"));
		}
	}
#endif
}


// Called when the game starts
void UItemInventoryComponent::BeginPlay()
{
	Super::BeginPlay();

	//Add InventoryItem to SubObjectList
	for(UInventoryItem* item : InventoryItems)
	{
		AddReplicatedSubObject(item);
		CurrentInventorySize++;
	}
}


void UItemInventoryComponent::BeginDestroy()
{
	Super::BeginDestroy();

	//Remove InventoryItem from SubObjectList
	for(UInventoryItem* item : InventoryItems)
	{
		RemoveReplicatedSubObject(item);
	}
}


// Called every frame
void UItemInventoryComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
}


void UItemInventoryComponent::GetLifetimeReplicatedProps(TArray<FLifetimeProperty>& OutLifetimeProps) const
{
	Super::GetLifetimeReplicatedProps(OutLifetimeProps);
	
	DOREPLIFETIME_CONDITION_NOTIFY(UItemInventoryComponent, InventoryItems, COND_None, REPNOTIFY_Always);
	DOREPLIFETIME(UItemInventoryComponent, CurrentInventorySize);
	DOREPLIFETIME(UItemInventoryComponent, MaxInventorySize);

	DOREPLIFETIME(UItemInventoryComponent, CurrentInventoryWeight);
	DOREPLIFETIME(UItemInventoryComponent, MaxInventoryWeight);
}


void UItemInventoryComponent::UpdateVariablesAndNotifyChanges()
{
	CurrentInventorySize = InventoryItems.Num();
	UpdateWeight();
	if (GetNetMode() != NM_DedicatedServer && GetNetMode() != NM_Client)  OnRep_Items(); 
}


//Notify Inventory has changed
void UItemInventoryComponent::OnRep_Items()
{	
	if (RefreshInventory.IsBound()) RefreshInventory.Broadcast();
}


// Check if Item can be added to Inventory
bool UItemInventoryComponent::CanAddItem_Implementation(const class UInventoryItem* inventoryItem) const
{
	if (inventoryItem == nullptr) return false;
	
	if (this->ExistsID(inventoryItem->ItemID)) return true;

	if (CurrentInventorySize + 1 <= MaxInventorySize) return true;

	if (bUseWeight) //If Weight is enabled
	{	
		FInventoryItemData* itemData = ItemDataTable->FindRow<FInventoryItemData>(inventoryItem->ItemID, FString(""));
		if (itemData->Weight * inventoryItem->Quantity + CurrentInventoryWeight <= MaxInventoryWeight) return true;
	}
	else
	{
		return true;
	}

	UE_LOG(LogTemp, Warning, TEXT("Can't add Item!"));
	return false;
}


//Add ItemStack to Inventory
bool UItemInventoryComponent::AddItem_Implementation(class UInventoryItem* inventoryItem)
{	
	if(!this->CanAddItem(inventoryItem)) return false;

	//Check if Item is already in Inventory and create new object or add Quantity
	if (this->ExistsID(inventoryItem->ItemID))
	{
		this->GetItemByID(inventoryItem->ItemID)->AddQuantity(inventoryItem->Quantity);
	}
	else
	{
		UInventoryItem* newItem = NewObject<UInventoryItem>(this);
		newItem->Copy(inventoryItem);
		InventoryItems.Add(newItem);
		AddReplicatedSubObject(newItem);
	}

	UpdateVariablesAndNotifyChanges();
	return true;
}


bool UItemInventoryComponent::CanRemoveItem_Implementation(const UInventoryItem* inventoryItem) const
{
	if(inventoryItem == nullptr) return false;

	if(this->InventoryItems.Contains(inventoryItem)) return true;

	if(this->ExistsID(inventoryItem->ItemID))
	{
		UInventoryItem* item = this->GetItemByID(inventoryItem->ItemID);
		if(item->Quantity >= inventoryItem->Quantity) return true;
	}

	return false;
}


//Remove InventoryItem from Inventory
bool UItemInventoryComponent::RemoveItem_Implementation(class UInventoryItem* inventoryItem)
{	
	if (!this->CanRemoveItem(inventoryItem)) return false;

	//Remove InventoryItem from Inventory
	if (this->InventoryItems.Contains(inventoryItem))
	{
		InventoryItems.Remove(inventoryItem);
		RemoveReplicatedSubObject(inventoryItem);
		inventoryItem->ConditionalBeginDestroy();
	} 
	else 
	{
		UInventoryItem* item = this->GetItemByID(inventoryItem->ItemID);
		if (item == nullptr) return false;

		item->RemoveQuantity(inventoryItem->Quantity);
		if (item->Quantity <= 0)
		{
			InventoryItems.Remove(item);
			RemoveReplicatedSubObject(item);
			item->ConditionalBeginDestroy();
		}
	}

	UpdateVariablesAndNotifyChanges();
	return true;
}


//Check if Inventory contains InventoryItem with corresponding Item.
bool UItemInventoryComponent::ExistsID(const FName itemID) const
{
	for (UInventoryItem* item : InventoryItems)
	{
		if (item->ItemID == itemID) return true;
	}
	return false;
}


UInventoryItem* UItemInventoryComponent::GetItemByID(const FName itemID) const
{
	for (UInventoryItem* item : InventoryItems)
	{
		if (item->ItemID == itemID) return item;
	}
	return nullptr;	
}


void UItemInventoryComponent::UpdateWeight()
{	
	if (!bUseWeight) return;

	CurrentInventoryWeight = 0;
	for (UInventoryItem* item : InventoryItems)
	{
		FInventoryItemData* itemData = ItemDataTable->FindRow<FInventoryItemData>(item->ItemID, FString(""));
		CurrentInventoryWeight += item->Quantity * itemData->Weight;
	}
}