# The Metric Consensus Process

## How to run & general information
Everything in this simulation software is done via the (consensus.properties)[consensus.properties] file.
The simulation software itself is written in a modular way, so that it is very easy to add new custom classes. A simulation process consists of three components:

* *Configuration*: Can be considered as the initial opionion or starting position for each simulation process.
* *Dynamics*: The update rule of each node in each simulation round.
* *Termination criterion*: Indicates at which point the simulation process should terminate. This is espcially important, when the consensus does not converge to one opionion, but just moves closer together.

To select a setting simply write classname as it is written in the Java classfile at the corresponding position in the config file. The name must match lower and uper case. If the are any additional parameters requried they have to be annotated, like in the following example for the *Beta-Closest-Node-Dynamics*.
```
@Parameter(isParameter = true, name = "Beta")
public double beta;
```
The name is later used in the consensus.prop file to target the variable. To run the application, just run it like any other java application. The simulation software will take every information needed from the consensus.properties file.

## Possible Settings
This is a list of possible settings a user can choose from by default. Some of the settings do not work together and can result in an infinite loop.

### Configurations
* *Circle*: All nodes are aligned in a circle. The user needs to provide the number of points the circle has.
* *Full-Circle*: All nodes are on the a circle.
* *Random-Nodes*: Random starting positions.

### Dynamics
* *Closest-Node-Dynamics*: Asks two random nodes for their opinion/position and selects the opinion that is the closest.
* *β-Closest-Node-Dynamics*: With the probability of *β* applies the *Closest-Node-Dynamics*, otherwise the *Voter-Dynamics*.
* *Closest-To-Mean-Dynamics*: Asks two other nodes for their opinion/position and calculates the mean including itslef. Jumps to the node which is the closest to the calculated mean.
* *Mean-Value-Dynamics*: Asks two other nodes for their opinion/position and calculates the mean including itslef. Adapts the calculated mean as its own opinion/position. This dynamics has to be paired up with the *Epsilon-Termination*, otherwise the simulation could end up in an infinite loop.
* *Voter-Dynamics*: Adapts the opinion/position of a random node.

### Termination Criteria
* *Consensus-Termination*: Terminates, when all nodes converged to the same opinion/position.
* *Almost-Consensus-Termination*: Used with byzantine nodes, to test if the honest nodes can still converge to the same opinion/position.
* *Epsilon-Termination*: Terminates when all nodes are within a circle of radius epsilon.
* *Number-Of-Cluster-Termination*: Terminates when only the given number opinions remain.
* *Fifity-Percent-Termination*: Used with byzantine nodes and terminates the process, when over 50% of nodes adapted a faulty opinion.

