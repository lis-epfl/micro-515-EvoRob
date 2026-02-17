# MICRO-515

**Goal:** Design your own evolutionary robotics experiment and present your findings in the final poster session  
**How:** In 4 exercises, we will explore different concepts of evolutionary robotics experimentation.  

---

## Introduction

Large programming projects are often modularised in different components to develop and adjust a project pipeline quickly. In the upcoming exercises, we will (re)build an evolutionary pipeline for robot evolution in [MuJoCo](https://mujoco.readthedocs.io). MuJoCo is a fast general-purpose physics engine commonly used in robotics research and industry (e.g. [OpenAI](https://openai.com/index/openai-gym-beta/)). Mujoco will be used to evaluate the performance of individuals (specifically robots) in our Evolutionary Algorithm (EA). We will interface the EA with our simulator using [gymnasium environments](https://gymnasium.farama.org/) that hold a specific interface to interact with the simulator. On the black-end Gym environments offer additional software infrastructure (like memory management and asynchronous parallelisation) for big (robot) experiments. 

At the end of these exercises you will have created a fully functional evolutionary robotics pipeline that you can also use for real-world robotics projects. These are the learning objectives of this exercises:

- Hands-on experience with commonly used evolutionary algorithms and deep reinforcement learning algorithms in robotics.  
- Proficiency with state-of-the art software tools like Gym environments and the physics engine MuJoCo.   
- Ability to design and build an evolutionary experiment

**Final examination:** poster session


## The EvoRob software pipeline

In the next section we will outline a general concept of the EvoRob pipeline and its main components. The flow of information in an EvoRob experiment with the software framework is presented below.  
![evorob overview](doc/imgs/evorob_overview.png)

An EvoRob experiment is divided into two main components: the **EA**, and the **World**. The **EA** is an umbrella for any evolutionary algorithm used
to optimise a vector of numbers (e.g. CMA, ES, NSGA-II etc.). The **World** may be any environment we would like to optimise within. During the exercises
we will change parts of the software to our needs (e.g. change the optimiser, or the controller). This division into components enables us to quickly
change (i.e. hotswap) these parts of our framework.

For consistency with our framework, the source code is separated in a  [📁evorob/algorithms](evorob/algorithms), and  [📁evorob/world](evorob/world),
folder. A third  [📁evorob/utils](evorob/utils) folder contains several functionalities like filesystem management
and geometric matrix manipulation.

### EA

The **EA** handles the evolutionary process. These back-end functionalities are separated as a Python class (see overview below). More information about classes can be found here: [https://docs.python.org/3/tutorial/classes.html](https://docs.python.org/3/tutorial/classes.html). The **EA** class is a general description of our evolutionary algorithm, providing the core routines for our evolutionary process, such as the generation of new populations and the mutation of our genotype.  
Concrete algorithms designed by you have to follow this convention to interface with the **World** and our population. These class functions should be customized for different implementations of **EA**s. The dots indicate the possibility of custom functions that are specific for your implementation (for example ES requires additional mutation functions for you to plug in).  For convenience, we also add utility functions like the loading and saving of checkpoints or logging of current fitness.

```
class EA():

    def __init__(self, n_pop, n_params, opts: dict = EA_opts, output_dir: str = "./results"):
        """
        Evolutionary Algorithm

        :param n_pop: population size
        :param n_params: number of parameters
        :param opts: algorithm options
        :param output_dir: output directory
        """

    def ask(self):
        """Generates a new population based on the current one."""
   
    def tell(self, solutions, function_values, save_checkpoint=True):
       """Updates the current population given the individuals and fitnesses."""
   
    def initialise_x0(self, num_parameters):
        """Initialises the first population."""
   
    def save_checkpoint(self):
        ...
   
    def load_checkpoint():
        ...
```

### World

The **World** handles the computation of the fitness per individual. The **World** translates a vector of numbers to a phenotype (e.g. a vector of numbers
representing segment lengths \-genotype- are translated into a robot \-phenotype-), which is used to evaluate an individual . Evaluation results  in
a fitness value(s), which is sent to the **EA**. For each exercise, the class functions should be customized for different implementations of **World**s.
```
class World():

    def __init__(self):
        """World class"""

    def geno2pheno(self, genotype):
        """Transcribes genotype input and returns phenotype"""

    def evaluate_individual(self, save_checkpoint):
        """Test current individual and returns fitness"""
```

## Exercises

In the following exercises, we will develop the building blocks of the pipeline. We will publish each exercise on a different branch. Throughout the manual, we will ask conceptual questions to check whether you understand the framework.

| You will find programming questions in boxes (like this). |
| :---- |

The corresponding sections in the code are indicated with `#TODO`s.


## Exercise timeline

### [Install the software](doc/installation.md) (19.02.2026)

### [Introduction: Customize MuJoCo simulators, and implement your own EA](doc/introduction0.md) (19.02.2026)
**Goal:** Introduction to the MuJoCo simulator, build and test your own EA.  
**How:** Play with the MuJoCo simulator. Familiarise yourself with the Passive Dynamic Walker environment, and integrate your own EA to optimise the morphology.  
**Short description:** A warm-up exercise to get acquainted with the software, where you will add and remove objects in the simulator, and build your very own EA class. At the end, we take a first try at morphology optimization by evolving a Passive Dynamic Walker in a custom MuJoCo **World**.  

### [Challenge 1: Evolving control with openAI gym - flat terrain](doc/challenge1.md) (05.03.2026)
**Goal:** Evolve a neural network controller and compare with PPO.  
**How:** Use Multi-Layer Perceptrons (MLPs) on agents in a gym environment.  
**Short description:** In this challenge, we recreate the Ant environment for robot learning. Using ready-made libraries we will optimize an MLP, and introduce the OpenAI gymnasium interface (referred to as gym environments). We will load different EA libraries using `ask()` and `tell()` interface, and compare with Reinforcement Learning algorithms.

### [Challenge 2: Multi-Objective optimization - two terrains](doc/challenge2.md) (19.03.2026)
**Goal:** Evolve a general control strategy for different environments.  
**How:** Build your own NSGA-II and train on flat and icy terrain.  
**Short description:**  In this challenge, Build our own multi-objective EA to test different control strategies in a flat, and icy terrain. We will compare the performance of single vs. multi-objective controllers, and analyse the different strategies that emerge at the Pareto front.

### [Challenge 3: Co-evolve parameterized body and brain - hilly terrain](doc/challenge3.md) (02.04.2026)
**Goal:** Evolve both robot controller and morphology on hilly terrain.  
**How:** Integrate the previous exercises for body and brain co-evolution.  
**Short description:** In this challenge, we adapt both body and brain by optimizing leg-length and different controller parameters. Here, we compare different control strategies: open-loop vs. feed-back vs. adaptive feed-back control and analyse how these impact morphological changes. 

### [Final Presentation](doc/final.md) (23.04.2026 \- **deadline**: T.B.A)
**Goal:** Full robot evolution.  
**How:** Evolve a robot to conquer all previous environments, and a mystery terrain given at the end.  
**Short description:** This is the final project which you will present in a (**graded**) poster session. We highly encourage you to adapt the EvoRob codebase and improve the current setup. Creativity is rewarded.
