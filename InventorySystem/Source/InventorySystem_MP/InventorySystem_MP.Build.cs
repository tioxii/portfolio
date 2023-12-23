// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;

public class InventorySystem_MP : ModuleRules
{
	public InventorySystem_MP(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore", "HeadMountedDisplay", "EnhancedInput", "InteractionSystem", "InventorySystem", "QuestSystem", "GameplayAbilities", "GameplayTags", "GameplayTasks" });
		PublicIncludePaths.AddRange(new string[] { "InventorySystem_MP" });
	}
}
