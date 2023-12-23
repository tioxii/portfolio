// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Interface.h"
#include "InteractInterface.generated.h"

// This class does not need to be modified.
UINTERFACE(MinimalAPI)
class UInteractInterface : public UInterface
{
	GENERATED_BODY()
};

/**
 * 
 */
class INTERACTIONSYSTEM_API IInteractInterface
{
	GENERATED_BODY()

	// Add interface functions to this class. This is the class that will be inherited to implement this interface.
public:

	UFUNCTION(BlueprintNativeEvent)
	void Trigger_Interaction_Client(class AActor* Interactor);

	UFUNCTION(BlueprintNativeEvent)
	bool Trigger_Interaction_Server(class AActor* Interactor);

 	bool Trigger_Interaction_Server_Implementation(class AActor* Interactor);

	UFUNCTION(BlueprintNativeEvent)
	void Cancel_Interaction(class AActor* Interactor);

	UFUNCTION(BlueprintNativeEvent)
	void OnFocus(class AActor* Interactor);

	UFUNCTION(BlueprintNativeEvent)
	void OnLostFocus(class AActor* Interactor);
};
