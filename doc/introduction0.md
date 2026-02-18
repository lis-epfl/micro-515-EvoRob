
# Introduction0a: Mujoco Basics

**Goal**: Familiarise yourself with the Mujoco physics simulator.  
**How**: Create new objects, modify gravity, create a simple robot by modifying the XML file.  

----------------------------------------------------------------------
MuJoCo is a physics engine that has been widely adopted in AI robotics research, due to its [fast compute time](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6386109). In fact, most OpenAI gym environments use MuJoCo as their simulator on the back-end.

![mujoco](imgs/exercise0a.png)

## Instructions

| Checkout out the introduction0 branch:  `git checkout introduction0` |
| :----                                                      |

Open the [XML file](../resources/exercise0.xml) and have a look at the elements. You will see that it is still pretty empty. The important bits are
- the `option`s section used to define `gravity`
- the `asset`s section - here you can define `texture`s and `material`s which you can reference later
- the `worldbody` section where you define the actual components of the simulation
For a comprehensive overview have a look at the [XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html).

Run [Introductiona.py](../Introduction0a.py) and explore the simulator. So far we have only defined the floor, light, and a single static box. See if you can spawn
two additional objects, e.g., spheres.

Now run the simulator again to verify your changes. By adding a `<freejoint/>` tag inside of the `body` tag you can make your newly created objects dynamic.
Do you observe a change when you run the simulator again? Why / why not?

Finally, see if you can create a very simple robot - it does not have to be actuated.

Tip: look at [body-joint](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint) to connect components. Feel free to use any other components
that might strike your interest.


# Introduction0b: Passive Dynamic Walker

![passive walker](imgs/passive0.gif)

**Goal:** Build and test your own ES & optimise the passive dynamic walker  

**Introduction**
This exercise is a warm-up to understand the information flow of an evolution pipeline. We will use our EvoRob pipeline and build your own implementation
of an ES. The use of simple test-functions is important for debugging.

*Q.0.1*

* ESs are ‘blackbox’ optimisers that require minimal information on the problem. Nevertheless, what information is still needed for the ES to optimise the parameters in the **World**?

*Q.0.2*
```python
class ES:

    def __init__(self, n_pop, n_params, opts: Dict = ES_opts):
        ...

    def ask(self):
        """Generates a new population based on the current one."""
        # TODO

    def tell(self, solutions, function_values, save_checkpoint=True):
        """Updates the current population given the individuals and fitnesses."""
        # TODO

    def initialise_x0(self):
        """Initialises the first population."""
        # TODO

    def update_sigma(self):
        """Update the perturbation strength (sigma)."""
        # TODO

    def sort_and_select_parents(self, population, fitness, num_parents):
        """Sorts the population based on fitness and selects the top individuals as parents."""
        # TODO

    def update_population_mean(self, parent_population, parent_fitness, rank: bool = True):
        """Updates the population mean based on the selected parents and their fitness."""
        # TODO

    def generate_mutated_offspring(self, population_size):
        """Generates a new population by adding Gaussian noise to the current mean."""
        # TODO
```

| The ES class is provided with standard functionalities (such as tracking results, and loading from checkpoints), but still lacks the functionality to optimize properly.|
| :---- |

* The ES class uses the ask/tell interface that receives and provides certain inputs and outputs. What are the inputs and outputs of the ask/tell interface?

* The ES class is provided with automatic bookkeeping during optimisation. We can retrieve the stored data for plotting results. What do the array dimension
  of the \*.npy files correspond with (generations/individuals/genes/fitnesses/…?). Analyse the data obtained with
  \<fitness\_full \= [np.load(...)](https://numpy.org/doc/stable/reference/generated/numpy.load.html)\> how do the dimensions of the fitness variables
  relate to n\_pop, n\_gen, num\_parents?

----------------------------------------------------------------------

In the second part of the exercise, we take a first try at robot morphology optimization by building a custom **World** with [MuJoCo](https://mujoco.readthedocs.io). Here, MuJoCo is used to test and evolve different passive dynamic walker designs.   
[Passive dynamic walkers](https://www.youtube.com/watch?v=HwlKouopjqM) are mechanical systems that locomote down a slope by gravity without any further actuators.
We will optimize the lengths of the leg segments and feet to maximize the distance travelled of our passive dynamic walker. We describe our passive walker
as a sphere with two legs. Each leg is connected through a ‘hip’ joint, and consists of an upper leg connected with a ‘knee’ joint to a lower leg with
foot (no ankle joint). The leg segment lengths (genotype) are parsed into robot description (phenotype) and saved as an .xml file that MuJoCo can read.

![passive walker structure](imgs/passive1.png)

*Q2.0* Look in the step() function of the custom gym environment [evorob/world/envs/passive_walker.py](../evorob/world/envs/passive_walker.py)

* What is the current reward function?

| Improve the reward function |
| :---- |

* In the same step() function a termination clause is built-in to save simulation time. What are the termination criteria?

*Q2.1* The geno2pheno() function translates the genotype (a vector describing the leg segment lengths) to a phenotype (a graph-description of a robot). The graph-description consists of the (x,y,z) Cartesian coordinates that represent nodes (points), and a connectivity matrix that defines how nodes are connected (connectivity\_matrix). You can imagine the graph as a set of nodes (whose locations are defined by points) whose connections are rigid bodies (defined by a connectivity matrix).  
The figure above shows a graph-description for our passive dynamic walker: a base rigid body at the top, with nodes as circles and leg segments ( e.g. right\_up\_leg) as lines. The points, (x,y,z)-locations, are changed during optimization as we evolve the leg segment lengths. How nodes are connected remains the same, thus the connectivity\_matrix remains the same.

* The  visualise\_individual() function, takes a genotype as input and simulates the passive dynamic walker. Inside of this function the geno2pheno() function produces the points and connectivity\_matrix that describe the robot. Play with the inputs of the visualise\_individual() function to see if you understand what is going on inside geno2pheno(). Hint: print the points output.

We automatically create robot descriptions that MuJoCo can read during evolution. Based on the points and connectivity\_matrix information our `PassiveWalkerRobot` class writes a robot description file (`PassiveWalkerRobot.xml`) that can be loaded by MuJoCo. The file is saved in a temporary directory. If you want to change the location the file is written to you can modify the `temp_dir` attribute in `PassiveWalkerWorld`.

* In the same file we load slope geometry as a worldbody. Can you find the slope angle of this geometry?

Open a simulation instance of MuJoCo by running the following command in the command line: `python -m mujoco.viewer`

* Use drag-and-drop to load the default walker world file 📂[/evorob/world/robot/assets/walker_world.xml](..//evorob/world/robot/assets/walker_world.xml) file into the simulator. Can you change the slope in this file:
  * Color
  * Length
  * Width

Return the properties of the slope back to its original settings.

| Improve the geno2pheno mapping (hint: can you reduce the number of parameters in the genotype) |
| :---- |

*Q2.2* The `PassiveWalkerRobot.xml` that our `PassiveWalkerRobot` class generates describes our simulation model. We can plug this xml in our custom `PassiveWalkerWorld` environment to calculate a fitness value. This provides the same interface  with your EA as our previous exercises. Now that we have the **World** we can plug in our previously developed **EA**.

| Adapt your ES to optimize for the new genotype |
| :---- |

* Does the best solution make sense?

| Make a video of the best walker |
| :---- |

* Is the best solution different?

Q2.3 

| Can you change the slope height? HINT: also look into the walker\_world.xml |
| :---- |

* How much further can you go?
