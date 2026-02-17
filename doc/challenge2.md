# Challenge 2: Evolving Multi-objective Neural Controllers in Two-Terrain Scenario

![Challenge 1 Ant Banner](imgs/challenge_1_ant_banner.png)

In Challenge 2, you will use multi-objective evolutionary algorithms to evolve specialist and generalist neural controller to let an abstract four-legged robot learn to locomote on a flat surface with different friction conditions as fast as possible. 

## Learning Goals

<a><img src="imgs/ant_locomote.gif" width="200" align="right" /></a>

In this exercise, you will ...
- implement the baseline multi-objective evolutionary algorithm __Non-sorting Genetic Algorithm II__ proposed by [Deb et al. 2002](https://doi.org/10.1109/4235.996017).
- compare single-objective specialist controllers to controllers on the Pareto front evolved by your multi-objective algorithm.


## Exercise 2: NSGA-II implementation

NSGA-II is a popular baseline algorithm for multi-objective optimization. The following figure illustrates the working principle of NSGA-II. Checkout the original publication [Deb et al. 2002](https://doi.org/10.1109/4235.996017) and the lecture material to familiarize with the concept of NSGA-II. 

<div align="center">
  <img src="imgs/nsga.png" width="80%">
</div>

1. Find `nsga.py` in the `evorob` codebase ([evorob/algorithms/nsga.py](/evorob/algorithms/nsga.py)). You will find the following structure:

```python
class NSGAII():

    def ask(self):
        if self.current_gen==0:
            new_population = self.initialise_x0()
        else:
            new_population = self.create_children(self.n_pop)
        new_population = np.clip(new_population, self.min, self.max)
        return new_population

    def tell(self, solutions, function_values, save_checkpoint=True):
        parents_population, parents_fitness = self.sort_and_select_parents(
            solutions, function_values, self.n_parents
        )

    def initialise_x0(self):
        ...

    def create_children(self, population_size):
        ...

    def sort_and_select_parents(self, solutions, function_values, n_parents):
        ...
        return solutions[draw_ind], function_values[draw_ind]

    def dominates(self, individual, other_individual) -> bool:
        ...

    def fast_nondominated_sort(self, fitness):
        ...
        return pareto_fronts, population_rank
```

2. Implement the `dominates`-function according to the specified formula. The dominates function receives the scalar fitness values in an array for two individuals. It returns `True` when the first individual given to the function dominates the second individual based on the given fitness values. Run 

> **Hint:** Remember, a solution dominates another if:
>    1. $\forall i \in {1, \dots, k}, f_i(x) \leq f_i(y)$ --> Solution ($x$) is no worse than Solution ($y$) in all objectives.
>    2. $\exists i \in {1, \dots, k}, f_i(x) < f_i(y)$ --> Solution ($x$) is strictly better than Solution ($y$) in at least one objective.

3. Now execute the main running script (`python3 Exercise2.py`) and see if your `dominates`-function is well-defined.

4. Implement the `fast_nondominated_sort`-function. The sorting is usually done in two parts, where first all individuals are compared against each other. 

## Exercise 2: Single vs. Multi-objective Evolution
...

# Challenge 2 Submission Details

The given exercise should provide an fundamental understanding of practical aspects of evolutionary robotics and concepts explained in the lecture. We encourage you to experiment with different evolutionary algorithms, hyperparameter settings and perform modifications to control architecture and learning curriculum in order to evolve the fastest flat-terrain Ant in __MICRO-515__ history!

<div align="center">
  <img src="imgs/ant_caricature.png" width="60%">
</div>

To master Challenge 2, submit:
- the weights and code of your final controller. The controller should be compatabile with the `controller` interface presented in the exercises.
- a video rendering of your Ant.
- a fitness graph showing the evolution of fitness over generations.
- a textfile `README.md` describing shortly the selected algorithm, environment design and controller (max. 300 words).

Provide all documents in a zipped folder with the following naming convention: `2026_micro_515_SCIPER_TEAMNAME_LASTNAME1_LASTNAME2.zip`.

We will compare all submissions and publish the results on a leaderboard with the provided teamname at submission. While you might adapt the reward function, the final fitness evaluation should be performed on the default `Ant-v5` environment as provided in the evaluation script.


# Questions?

If some parts of your code are not working or you have general questions, do not hesitate to contact your MICRO-515 teaching assistants in the exercise sessions or via e-mail `fuda.vandiggelen@epfl.ch`, `alexander.ertl@epfl.ch`, `alexander.dittrich@epfl.ch`, `hongze.wang@epfl.ch`