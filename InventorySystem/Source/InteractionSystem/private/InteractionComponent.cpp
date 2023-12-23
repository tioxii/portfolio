// Fill out your copyright notice in the Description page of Project Settings.


#include "InteractionComponent.h"
#include "InteractInterface.h"
#include "Kismet/GameplayStatics.h"

// Sets default values for this component's properties
UInteractionComponent::UInteractionComponent()
{
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	PrimaryComponentTick.bCanEverTick = true;

	// ...
}


// Called when the game starts
void UInteractionComponent::BeginPlay()
{
	Super::BeginPlay();

	// ...
	
}


// Called every frame
void UInteractionComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	APlayerController* PlayerController = UGameplayStatics::GetPlayerController(GetWorld(), 0);

	if (!IsValid(PlayerController))
	{
		return;
	}

	AActor* FocusedActor = nullptr;
	float MinDistanceToScreenCenter = 0;
	for (AActor* Actor : InteractableActors)
	{
		if (!IsValid(Actor))
		{
			continue;
		}

		FVector2D ScreenLocation;
		bool bSuccess = PlayerController->ProjectWorldLocationToScreen(Actor->GetActorLocation(), ScreenLocation);

		UGameViewportClient* ViewportClient = GetWorld()->GetGameViewport();
		if (!IsValid(ViewportClient))
		{
			return;
			UE_LOG(LogTemp, Error, TEXT("ViewportClient is nullptr"));
		}
		FIntPoint ViewportSize = ViewportClient->Viewport->GetSizeXY();
		FIntPoint ViewportCenter = ViewportSize / 2;

		float DistanceToScreenCenter = FVector2D::Distance(ScreenLocation, FVector2D(ViewportCenter));
		if (FocusedActor == nullptr || DistanceToScreenCenter < MinDistanceToScreenCenter)
		{
			FocusedActor = Actor;
			MinDistanceToScreenCenter = DistanceToScreenCenter;
		}
	}

	if (FocusedActor != nullptr)
	{
		if (FocusedActor != CurrentFocusedActor)
		{
			if (CurrentFocusedActor != nullptr)
			{
				IInteractInterface::Execute_OnLostFocus(CurrentFocusedActor, this->GetOwner());
			}

			IInteractInterface::Execute_OnFocus(FocusedActor, this->GetOwner());
			CurrentFocusedActor = FocusedActor;
		}
	}
	else
	{
		if (CurrentFocusedActor != nullptr)
		{
			IInteractInterface::Execute_OnLostFocus(CurrentFocusedActor, this->GetOwner());
			CurrentFocusedActor = nullptr;
		}
	}
}

void UInteractionComponent::AddInteractableActor(AActor* Actor)
{	
	if (!IsValid(Actor))
	{
		return;
	}

	if (!Actor->Implements<UInteractInterface>())
	{
		return;
	}

	InteractableActors.Add(Actor);
}

void UInteractionComponent::RemoveInteractableActor(AActor* Actor)
{
	if (!IsValid(CurrentFocusedActor)) 
	{
		return;
	}

	if (InteractableActors.Contains(Actor))
	{
		InteractableActors.Remove(Actor);
	}
}

void UInteractionComponent::ClearInteractableActors()
{
	InteractableActors.Empty();
}

void UInteractionComponent::Trigger_Interaction()
{
	if (!IsValid(CurrentFocusedActor))
	{
		return;
	}

	if (IInteractInterface::Execute_Trigger_Interaction_Server(CurrentFocusedActor, this->GetOwner()))
	{
		Trigger_ServerInteraction(CurrentFocusedActor);
	}
	
	if (IsValid(CurrentFocusedActor))
	{
		IInteractInterface::Execute_Trigger_Interaction_Client(CurrentFocusedActor, this->GetOwner());
	}
}

void UInteractionComponent::Cancel_Interaction()
{

}

void UInteractionComponent::Trigger_ServerInteraction_Implementation(AActor* FocusedActor)
{	
	if (!IsValid(FocusedActor))
	{
		return;
	}

	if (FocusedActor->IsOverlappingActor(this->GetOwner()))
	{
		IInteractInterface::Execute_Trigger_Interaction_Server(FocusedActor, this->GetOwner());
	}
}
