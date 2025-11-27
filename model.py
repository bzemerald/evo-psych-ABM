from pathlib import Path

import numpy as np

import mesa
from mesa.discrete_space import OrthogonalVonNeumannGrid
from mesa.discrete_space.property_layer import PropertyLayer
from agents import SugarscapeAgent


# Helper Functions
def flatten(list_of_lists):
    """
    helper function for model datacollector for trade price
    collapses agent price list into one list
    """
    return [item for sublist in list_of_lists for item in sublist]


def geometric_mean(list_of_prices):
    """
    find the geometric mean of a list of prices
    """
    return np.exp(np.log(list_of_prices).mean())



class Sugarscape(mesa.Model):
    """
    Manager class to run basic Sugarscape with sugar-only agents
    """

    def __init__(
        self,
        width=50,
        height=50,
        initial_population=200,
        endowment_min=25,
        endowment_max=50,
        metabolism_min=1,
        metabolism_max=5,
        vision_min=1,
        vision_max=5,
        seed=None,
    ):
        super().__init__(seed=seed)
        # grid size
        self.width = width
        self.height = height

        self.running = True

        # grid
        self.grid = OrthogonalVonNeumannGrid(
            (self.width, self.height), torus=False, random=self.random
        )

        # datacollector (simple: just count agents; add more if you want)
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "#Agents": lambda m: len(m.agents),
            },
            agent_reporters={
                "Sugar": lambda a: a.sugar,
            },
        )

        # sugar landscape from file
        self.sugar_distribution = np.genfromtxt(
            Path(__file__).parent / "sugar-map.txt"
        )

        self.grid.add_property_layer(
            PropertyLayer.from_data("sugar", self.sugar_distribution)
        )

        # create agents
        SugarscapeAgent.create_agents(
            self,
            initial_population,
            self.random.choices(self.grid.all_cells.cells, k=initial_population),
            sugar=self.rng.integers(
                endowment_min, endowment_max, (initial_population,), endpoint=True
            ),
            metabolism=self.rng.integers(
                metabolism_min, metabolism_max, (initial_population,), endpoint=True
            ),
            vision=self.rng.integers(
                vision_min, vision_max, (initial_population,), endpoint=True
            ),
        )

    def step(self):
        """
        Step function for sugar-only Sugarscape:
        1) regenerate sugar
        2) move/eat/die for agents
        3) collect data
        """
        # regenerate sugar (1 unit per step up to landscape cap)
        self.grid.sugar.data = np.minimum(
            self.grid.sugar.data + 1, self.sugar_distribution
        )

        # step agents
        agent_shuffle = self.agents_by_type[SugarscapeAgent].shuffle()

        for agent in agent_shuffle:
            agent.move()
            agent.eat()
            agent.maybe_die()

        # collect model/agent data
        self.datacollector.collect(self)

    def run_model(self, step_count=1000):
        for _ in range(step_count):
            self.step()