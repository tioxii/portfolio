// Fill out your copyright notice in the Description page of Project Settings.

#include "PlayerInventoryComponent.h"
#include "EnhancedInputSubsystems.h"

void UPlayerInventoryComponent::BeginPlay()
{
    Super::BeginPlay();

    for (FEquipmentSlot& Slot : EquipmentSlots)
    {
        if (IsValid(Slot.Item)) AddReplicatedSubObject(Slot.Item);
    }
}

void UPlayerInventoryComponent::BeginDestroy()
{
    Super::BeginDestroy();

    for (FEquipmentSlot& Slot : EquipmentSlots)
    {
        if (IsValid(Slot.Item)) RemoveReplicatedSubObject(Slot.Item);
    }
}


void UPlayerInventoryComponent::GetLifetimeReplicatedProps(TArray<FLifetimeProperty>& OutLifetimeProps) const
{
    Super::GetLifetimeReplicatedProps(OutLifetimeProps); 
    DOREPLIFETIME_CONDITION(UPlayerInventoryComponent, EquipmentSlots, COND_OwnerOnly);
}


void UPlayerInventoryComponent::OnRep_Equipment()
{
    OnRefreshEquipment.Broadcast();
}


bool UPlayerInventoryComponent::TransferInventoryItem_Validate(class UInventoryItem* InventoryItem, class UItemInventoryComponent *From, class UItemInventoryComponent *To)
{
    if (From == nullptr || To == nullptr)
    {
        UE_LOG(LogTemp, Warning, TEXT("PlayerInventoryComponent::transferItemStack: from or to is null"))
        return false;
    }
    if (InventoryItem == nullptr)
    {
        UE_LOG(LogTemp, Warning, TEXT("PlayerInventoryComponent::transferItemStack: itemStack is null"))
        return false;
    }
    if (From == To)
    {
        UE_LOG(LogTemp, Warning, TEXT("PlayerInventoryComponent::transferItemStack: from and to are the same"))
        return false;
    }
    if (From->InventoryItems.Find(InventoryItem) == INDEX_NONE)
    {
        UE_LOG(LogTemp, Warning, TEXT("PlayerInventoryComponent::transferItemStack: itemStack is not in from"))
        return false;
    }
    return true;
}


/*
*   Transfer an item from one inventory to another.
*/
void UPlayerInventoryComponent::TransferInventoryItem_Implementation(class UInventoryItem* InventoryItem, class UItemInventoryComponent *From, class UItemInventoryComponent *To)
{
    if(To->CanAddItem(InventoryItem) && From->CanRemoveItem(InventoryItem))
    {
        From->RemoveItem(InventoryItem);
        To->AddItem(InventoryItem);
    }       
}


/*
* Transfer an item from one inventory to another.
*/
void UPlayerInventoryComponent::DropItem_Implementation(class UInventoryItem* Item, UItemInventoryComponent* From)
{
    if (From->CanRemoveItem(Item))
    {
        From->RemoveItem(Item);
    }
}


/*
*   Functions for equipping items.
*/
TArray<UInventoryItem*> UPlayerInventoryComponent::GetEquipedItems() const
{
    TArray<UInventoryItem*> EquipedItems = TArray<UInventoryItem*>();
    for (auto EquipmentSlot : EquipmentSlots)
    {
        if (EquipmentSlot.Item != nullptr)
        {
            EquipedItems.Add(EquipmentSlot.Item);
        }
    }
    return EquipedItems;
}


UInventoryItem* UPlayerInventoryComponent::GetEquipedItemBySlotName(FName Slot) const
{
    for (auto slotItem : EquipmentSlots)
    {
        if (slotItem.SlotName == Slot)
        {   
            UE_LOG(LogTemp, Display, TEXT("Slot found: %s"), *Slot.ToString()); 
            return slotItem.Item;
        }
    }
    return nullptr;
}


void UPlayerInventoryComponent::TransferInventoryItemToEquipment_Implementation(UInventoryItem* Item, FName Slot)
{   
    for (FEquipmentSlot& EquipmentSlot : EquipmentSlots)
    {   
        if (EquipmentSlot.SlotName == Slot && EquipmentSlot.ItemClass == Item->GetClass())
        {
            EquipmentSlot.Item = Item;
            RemoveItem(Item);
            AddReplicatedSubObject(Item);
            return;
        }
    }
}


void UPlayerInventoryComponent::TransferEquipmentItemToInventory_Implementation(FName Slot)
{
    for (FEquipmentSlot& EquipmentSlot : EquipmentSlots)
    {
        if (EquipmentSlot.SlotName == Slot && IsValid(EquipmentSlot.Item))
        {
            UInventoryItem* Item = EquipmentSlot.Item;
            EquipmentSlot.Item = nullptr;
            RemoveReplicatedSubObject(Item);
            AddItem(Item);
            return;
        }
    }
}


/*
*   Create an InventoryWidget and add it to the viewport.
*/
void UPlayerInventoryComponent::OpenInventory()
{   
    if (inventoryWidget == nullptr)
    {
        inventoryWidget = CreateWidget<UInventoryWidget>(GetWorld(), InventoryWidgetClass);
        if(inventoryWidget == nullptr)
        {
            UE_LOG(LogTemp, Warning, TEXT("PlayerInventoryComponent::openInventory: inventoryWidget is null"))
            return;
        }
        inventoryWidget->inventory = this;
        inventoryWidget->AddToViewport();
        
        
        UE_LOG(LogTemp, Warning, TEXT("PlayerInventoryComponent::openInventory"))

        APlayerController* playerController = Cast<APlayerController>(Cast<APawn>(GetOwner())->Controller);
        if (playerController == nullptr)
        {
            UE_LOG(LogTemp, Warning, TEXT("PlayerInventoryComponent::openInventory: playerController is null"))
            return;
        }

        FInputModeGameAndUI InputMode;
        InputMode.SetLockMouseToViewportBehavior(EMouseLockMode::DoNotLock);
        playerController->SetInputMode(InputMode);
        playerController->bShowMouseCursor = true;
    }
    else
    {
        inventoryWidget->RemoveFromParent();
        inventoryWidget = nullptr;
        UE_LOG(LogTemp, Warning, TEXT("PlayerInventoryComponent::closeInventory"))

        APlayerController* playerController = Cast<APlayerController>(Cast<APawn>(GetOwner())->Controller);
        if (playerController == nullptr)
        {
            UE_LOG(LogTemp, Warning, TEXT("PlayerInventoryComponent::closeInventory: playerController is null"))
            return;
        }

        FInputModeGameOnly InputMode;
        playerController->SetInputMode(InputMode);
        playerController->bShowMouseCursor = false;
    }
}