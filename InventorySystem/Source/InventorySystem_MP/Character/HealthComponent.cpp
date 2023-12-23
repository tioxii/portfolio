// Fill out your copyright notice in the Description page of Project Settings.


#include "Character/HealthComponent.h"
#include "Character/MyCharacter.h"

// Sets default values for this component's properties
UHealthComponent::UHealthComponent()
{
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	PrimaryComponentTick.bCanEverTick = false;

	SetIsReplicatedByDefault(true);

	AbilitySystemComponent = nullptr;
	HealthSet = nullptr;
}

void UHealthComponent::InitializeWithAbilitySystem(UAbilitySystemComponent* InASC)
{
	if (InASC != AbilitySystemComponent || AbilitySystemComponent == nullptr) AbilitySystemComponent = InASC;;
	
	if (!AbilitySystemComponent)
	{
		UE_LOG(LogTemp, Warning, TEXT("UHealthComponent::InitializeWithAbilitySystem() - AbilitySystemComponent is nullptr!"));
		return;
	}

	HealthSet = AbilitySystemComponent->GetSet<UHealthSet>();
	if (!HealthSet)
	{
		UE_LOG(LogTemp, Warning, TEXT("UHealthComponent::InitializeWithAbilitySystem() - HealthSet is nullptr!"));
		return;
	}

	AbilitySystemComponent->GetGameplayAttributeValueChangeDelegate(HealthSet->GetHealthAttribute()).AddUObject(this, &UHealthComponent::HandleHealthChanged);
	AbilitySystemComponent->GetGameplayAttributeValueChangeDelegate(HealthSet->GetMaxHealthAttribute()).AddUObject(this, &UHealthComponent::HandleMaxHealthChanged);
}


void UHealthComponent::UninitializeWithAbilitySystem()
{
	AbilitySystemComponent = nullptr;
	HealthSet = nullptr;

	AbilitySystemComponent->GetGameplayAttributeValueChangeDelegate(HealthSet->GetHealthAttribute()).RemoveAll(this);
}

void UHealthComponent::HandleHealthChanged(const FOnAttributeChangeData& Data)
{
	OnHealthChanged.Broadcast(Data.NewValue, Data.OldValue);
}

void UHealthComponent::HandleMaxHealthChanged(const FOnAttributeChangeData& Data)
{
	OnMaxHealthChanged.Broadcast(Data.NewValue, Data.OldValue);
}


float UHealthComponent::GetHealth() const
{
	if (HealthSet) return HealthSet->GetHealth();
	
	return 0.0f;
}

float UHealthComponent::GetMaxHealth() const
{
	if (HealthSet) return HealthSet->GetMaxHealth();

	return 0.0f;
}

float UHealthComponent::GetHealthPercentage() const
{
	if (HealthSet) return HealthSet->GetHealth() / HealthSet->GetMaxHealth();

	return 0.0f;
}