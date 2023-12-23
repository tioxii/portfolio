// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "AbilitySystem/Attributes/BaseAttributeSet.h"
#include "ResistanceSet.generated.h"

/**
 * 
 */
UCLASS()
class INVENTORYSYSTEM_MP_API UResistanceSet : public UBaseAttributeSet
{
	GENERATED_BODY()

public:

	UResistanceSet();

	ATTRIBUTE_ACCESSORS(UResistanceSet, PhysicalArmor);
	ATTRIBUTE_ACCESSORS(UResistanceSet, MagicResistance);

private:

	// Resistance against any physical damage
	UPROPERTY(BlueprintReadOnly, Category = "Resistance", meta = (AllowPrivateAccess = "true"))
	FGameplayAttributeData PhysicalArmor;

	// Resistance against any magical damage
	UPROPERTY(BlueprintReadOnly, Category = "Resistance", meta = (AllowPrivateAccess = "true"))
	FGameplayAttributeData MagicResistance;
};
