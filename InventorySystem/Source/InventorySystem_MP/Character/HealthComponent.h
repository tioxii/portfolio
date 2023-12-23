// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "AbilitySystem/Attributes/HealthSet.h"
#include "HealthComponent.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FHealth_AttributeChanged, float, NewValue, float, OldValue);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FHealth_MaxAttributeChanged, float, NewValue, float, OldValue);

UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class INVENTORYSYSTEM_MP_API UHealthComponent : public UActorComponent
{
	GENERATED_BODY()

public:	

	UHealthComponent();

	virtual void InitializeWithAbilitySystem(UAbilitySystemComponent* InASC);

	virtual void UninitializeWithAbilitySystem();

	virtual void HandleHealthChanged(const FOnAttributeChangeData& Data);

	virtual void HandleMaxHealthChanged(const FOnAttributeChangeData& Data);

	UFUNCTION(BlueprintCallable, Category="Health")
	float GetHealth() const;

	UFUNCTION(BlueprintCallable, Category = "Health")
	float GetMaxHealth() const;

	UFUNCTION(BlueprintCallable, Category = "Health")
	float GetHealthPercentage() const;

	UPROPERTY(BlueprintAssignable)
	FHealth_AttributeChanged OnHealthChanged;

	UPROPERTY(BlueprintAssignable)
	FHealth_MaxAttributeChanged OnMaxHealthChanged;

protected:
	
	UPROPERTY()
	TObjectPtr<class UAbilitySystemComponent> AbilitySystemComponent;

	UPROPERTY()
	TObjectPtr<const class UHealthSet> HealthSet;
};
