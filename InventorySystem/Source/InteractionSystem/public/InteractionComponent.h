// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "InteractionComponent.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE(FOnExistingInteractDelegate);

UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class INTERACTIONSYSTEM_API UInteractionComponent : public UActorComponent
{
	GENERATED_BODY()

public:	
	// Sets default values for this component's properties
	UInteractionComponent();

protected:
	// Called when the game starts
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

	UPROPERTY(BlueprintReadWrite, Category = "Interaction")
	TArray<class AActor*> InteractableActors;

	UFUNCTION(BlueprintCallable, Category = "Interaction")
	void AddInteractableActor(class AActor* Actor);

	UFUNCTION(BlueprintCallable, Category = "Interaction")
	void RemoveInteractableActor(class AActor* Actor);

	UFUNCTION(BlueprintCallable, Category = "Interaction")
	void ClearInteractableActors();

	UFUNCTION(BlueprintCallable, Category = "Interaction")
	void Trigger_Interaction();

	UFUNCTION(BlueprintCallable, Category = "Interaction")
	void Cancel_Interaction();

	UFUNCTION(Server, Reliable, Category = "Interaction")
	void Trigger_ServerInteraction(class AActor* Interactor);

	UPROPERTY()
	class AActor* CurrentFocusedActor;

	UPROPERTY(BlueprintAssignable, Category = "Interaction")
	FOnExistingInteractDelegate OnExistingInteract;
};
