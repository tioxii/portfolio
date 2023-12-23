// Fill out your copyright notice in the Description page of Project Settings.


#include "InventoryItem.h"
#include "ItemInventoryComponent.h"

void UInventoryItem::GetLifetimeReplicatedProps(TArray<FLifetimeProperty>& OutLifetimeProps) const
{
    Super::GetLifetimeReplicatedProps(OutLifetimeProps);

    DOREPLIFETIME(UInventoryItem, ItemID);
    DOREPLIFETIME_CONDITION_NOTIFY(UInventoryItem, Quantity, COND_None, REPNOTIFY_Always);
}

bool UInventoryItem::IsSupportedForNetworking() const
{
    return true;
}

void UInventoryItem::OnRep_Quantity_Implementation()
{   
    if (RefreshQuantity.IsBound()) 
    {
        RefreshQuantity.Broadcast();
    }
}

void UInventoryItem::Copy_Implementation(const UInventoryItem* item)
{
    ItemID = item->ItemID;
    Quantity = item->Quantity;
}

bool UInventoryItem::Equals_Implementation(const UInventoryItem* item) const
{
    return ItemID == item->ItemID;
}

void UInventoryItem::AddQuantity_Implementation(int32 amount)
{
    Quantity += amount;
}

void UInventoryItem::RemoveQuantity_Implementation(int32 amount)
{
    if (Quantity >= amount)
    {
        Quantity -= amount;
    }
}