# Portfolio
This repository consists of projects I have done during my time at the University or in my free time.

## Battleships
This a project I did during my first semester at University. The task was to create something in C slightly bigger than our usual weekly tasks and apply everything we have learned so far. I decided to implement [Battleships](https://en.wikipedia.org/wiki/Battleship_(game)) using [SDL2](https://github.com/libsdl-org/SDL).
I settled for Battleships since video games are one my passion. Since this was a first semester project the code is not good, not checked for security issues and is definitly not an example on how I would code today. If you want to run the code nontheless, click [here](Battleships/README.md) for more information.

## Metric Consensus Process - Bachelor's Thesis
This project was part of my bachelor's thesis. I was supposed model the metric consenus process by implementing a simulation software, which I have done in Java. 
The metric consensus process is a specifc type of consensus process (https://en.wikipedia.org/wiki/Consensus_(computer_science)). In the metric consensus process nodes try to converge to a single opinion based on the metric distances between nodes. The opinion in the metric consenus process is point in space and could represent the current location of the node.
The process itself is simulated in rounds. In each round a node can ask/sample two or more nodes about their opinion/location. The asking node itself can then decide based on the sample of opinions/location which opinion/location it should adapt.
The goal of this simulation was to find out how many rounds, depending on the input, it would take so that all nodes converge to the same opinion. I was also supposed to find out if there are any differences in the metric consensus to already existing consensus processes.
For more information about the metric consensus process and how to run the software click [here](metric-consensus/README.md).

## Inventory System UE5
This more of a sideproject I did during free time. I wanted to learn more about Unreal Engine and decided to implement an RPG like Inventory System.
The UI doesn't look good and mechanics are still very basic, but it works in multiplayer.
Players can pick up items and drop them. They can also equip them in a fitting slot and exchange them with other players by passing them through a chest.
It took me quiet a few iterations to find a design that makes it easy to add new items and still make it possible to be used in a multiplayer scenario.

## CUDA-Kernels
During GPU-Architectures course, we were supposed to choose an algorithm and adapt it for the [H100](https://www.nvidia.com/de-de/data-center/h100/) from NVIDIA using the the new feature [Thread Blocks Clusters](https://www.nvidia.com/de-de/data-center/h100/) that comes with it. By now, Thread Block Clusters are also avaiable with 50-Series and the corresponding [Blackwell](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)-Architecture. I chose the [kmeans](https://de.wikipedia.org/wiki/K-Means-Algorithmus) algorithm for the assignment. For more information on the requirements and how to build it click [here](cuda/Readme.md).