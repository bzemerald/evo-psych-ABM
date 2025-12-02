from pathlib import Path
from random import Random

import numpy as np

import mesa
from mesa.discrete_space import Cell, OrthogonalVonNeumannGrid
from mesa.discrete_space.property_layer import PropertyLayer
from agents import SugarscapeAgent, AgentParams, AgentLogicProtocol
from genetic import *
from mapmaker import *
from typing import Generator


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

class SugarGrid(OrthogonalVonNeumannGrid):
    def __init__(self, 
                dimensions,
                regen_amount: float,
                regen_chance: float,
                ini_sugar_map: np.ndarray | None = None,
                map_generator: Generator | None = None,
                new_map_cycle: int = 0,
                new_map_transition: int = 0,
                torus: bool = False, 
                capacity: float | None = None, 
                random: Random | None = None, 
                ) -> None:
        super().__init__(dimensions, torus, capacity, random)
        assert ini_sugar_map is not None or map_generator is not None
        self.regen_amount, self.regen_chance = regen_amount, regen_chance
        self.sugar_capacity = ini_sugar_map
        self.map_gen = map_generator
        if self.map_gen is not None:
            if ini_sugar_map is None:
                self.sugar_capacity = next(self.map_gen)
            if new_map_cycle < new_map_transition:
                raise ValueError("Transition period should be shorter than the cycle period.".join(
                    f"\n Got {new_map_transition} for transition and {new_map_cycle} for cycle instead."
                ))
            self._next_map = next(self.map_gen)
            self.cycle, self.transition = new_map_cycle, new_map_transition
            self._cycle_counter = 0

        self.add_property_layer(
            PropertyLayer.from_data("sugar", self.sugar_capacity)
        )
    
    def step(self):
        self._try_regen()
        self._map_cycle_step()
        

    def _map_cycle_step(self):
        if self.map_gen is None or self._next_map is None: 
            print(1)
            return
        self._cycle_counter += 1
        if self._cycle_counter > self.cycle - self.transition:
            remaining = self.cycle - self._cycle_counter
            self._shift_capacity_to(self._next_map, remaining)
        if self._cycle_counter == self.cycle:
            self._cycle_counter = 0
            try: self._next_map = next(self.map_gen)
            except StopIteration:
                self.map_gen = None
                self._next_map = None
                return


    def _try_regen(self):
        mask = rng.random(self.sugar.data.shape) < self.regen_chance
        self.sugar.data = np.minimum(
            self.sugar.data + self.regen_amount * mask, self.sugar_capacity
        )

    def _shift_capacity_to(self, next_map: np.ndarray, in_steps: int):

        diff = next_map - self.sugar_capacity
        total_diff = np.sum(np.abs(diff))
        if in_steps <= 0 or total_diff == 0:
            return

        num_changes = int(total_diff / in_steps)
        if num_changes <= 0:
            num_changes = 1 

        current = self.sugar_capacity

        for _ in range(num_changes):
            diff = next_map - current

            # 找到所有还不相等的位置
            flat_diff = diff.ravel()
            nonzero_idx = np.flatnonzero(flat_diff)
            if nonzero_idx.size == 0:
                break  # 已经和 next_map 一样了

            # 按 |diff| 大小加权随机选择一个位置
            weights = np.abs(flat_diff[nonzero_idx].astype(float))
            weights_sum = weights.sum()
            if weights_sum == 0:
                break
            probs = weights / weights_sum

            chosen_flat = np.random.choice(nonzero_idx, p=probs)
            i, j = np.unravel_index(chosen_flat, diff.shape)

            # 朝着 next_map 的方向移动一步（+1 或 -1）
            step = np.sign(diff[i, j]).astype(int)
            current[i, j] += step


class Sugarscape(mesa.Model):
    """
    Manager class to run basic Sugarscape with sugar-only agents
    """

    def __init__(
        self,
        empty_genome: EmptyLociCollection,
        agent_logics: AgentLogicProtocol,
        
        # Grid params
        width: int = 50,
        height: int = 50,
        grid_size: int | None = None,
        regen_amount: int = 1,
        regen_chance: float = 1,
        map_generator: Generator | None = None,
        new_map_cycle: int = 10,
        new_map_transition: int = 4,
        ini_sugar_map=None,

        # Initialization params
        initial_population: int = 200,
        endowment: int = 10,

        # Agent params
        initial_sugar: int = 0,
        reproduction_age: int = 10,
        reproduction_check_radius: int = 1,
        reproduction_cooldown: int = 5,
        max_age: int = 80,
        max_sugar:int = 50,
        max_children: int = 100,
        vision_min: int = 1,
        vision_max: int = 5,
        metabolism_min: int = 1,
        metabolism_max: int = 5,
        mutation_rate: float = 0.05,
        num_alleles: int = 5,
        seed=None,
    ):
        super().__init__(seed=seed)
        if seed:
            set_seed(seed) # sync up with numpy's rng in the genetic module

        # allow a single slider to control both width and height
        if grid_size is not None:
            width = height = int(grid_size)

        # grid
        self.width, self.height = width, height

        # default map generator depends on grid size if none is provided
        if map_generator is None:
            map_generator = spiky(
                self.width,
                self.height,
                spike_size=10,
                spike_decrement=4,
                spike_freq=0.01,
            )

        self.grid = SugarGrid(
            (self.width, self.height), 
            regen_amount=regen_amount, 
            regen_chance=regen_chance, 
            ini_sugar_map=None if map_generator is not None else ini_sugar_map,
            map_generator=map_generator,
            new_map_cycle=new_map_cycle,
            new_map_transition=new_map_transition,
            torus=False, 
            random=self.random
        )

        self.empty_genome = empty_genome
        self.gene_names = list(self.empty_genome.by_name.keys())
        self.is_gendered = empty_genome.is_gendered

        # construct AgentParams from keyword arguments
        self.agent_params = AgentParams(
            min_vision=vision_min,
            max_vision=vision_max,
            min_metabolism=metabolism_min,
            max_metabolism=metabolism_max,
            initial_sugar=initial_sugar,
            reproduction_age=reproduction_age,
            reproduction_check_radius=reproduction_check_radius,
            reproduction_cooldown=reproduction_cooldown,
            max_age=max_age,
            max_children=max_children,
            max_sugar=max_sugar
        )
        self.agent_logics = agent_logics

        # trait and mutation parameters (ensure min <= max)
        self.vision_min, self.vision_max = sorted((vision_min, vision_max))
        self.metabolism_min, self.metabolism_max = sorted(
            (metabolism_min, metabolism_max)
        )
        self.mutation_rate = mutation_rate
        # apply mutation rate to all genes
        Gene.mutation_rate = mutation_rate
        # dynamically set num_alleles for vision / metabolism genes
        vision_alleles = int(self.vision_max - self.vision_min + 1)
        metabolism_alleles = int(self.metabolism_max - self.metabolism_min + 1)
        vision_cls = Gene.loci_registry.get("VisionGene")
        if vision_cls is not None:
            vision_cls.set_num_alleles(vision_alleles)
        metabolism_cls = Gene.loci_registry.get("MetabolismGene")
        if metabolism_cls is not None:
            metabolism_cls.set_num_alleles(metabolism_alleles)
        self.running = True
        control_cls = Gene.loci_registry.get("ControlGene")
        if control_cls is not None:
            control_cls.set_num_alleles(num_alleles)
        strategy_cls = Gene.loci_registry.get("StrategyGene")
        if strategy_cls is not None:
            strategy_cls.set_num_alleles(num_alleles)

        # datacollector: collect agent count and allele frequencies for all genes
        model_reporters: dict[str, object] = {
            "#Agents": lambda m: len(m.agents),
        }
        for gene in self.gene_names:
            model_reporters[f"{gene}_freqs"] = (
                lambda m, gene=gene: m.allele_frequencies(gene)
            )

        self.datacollector = mesa.DataCollector(
            model_reporters=model_reporters,
            agent_reporters={
                "Sugar": lambda a: a.sugar,
            },
        )

        # create agents
        karyotypes = (
            (AGENDER_GENOME,)
            if not self.is_gendered
            else (MALE_GENOME, FEMALE_GENOME)
        )
        # Generate a genotype per agent to introduce genetic variation
        ini_genotypes = [
            self.empty_genome.generate_random_genotype(*karyotypes)
            for _ in range(initial_population)
        ]
        SugarscapeAgent.create_agents(
            self,
            initial_population,
            self.random.choices(self.grid.all_cells.cells, k=initial_population),
            genotype=ini_genotypes,
            params=self.agent_params,
            logics=self.agent_logics,
            sugar=endowment,
        )

    def step(self):
        """
        Step function for Sugarscape:
        1) regenerate sugar
        2) move/eat/die for agents
        3) collect data
        """
        # regenerate sugar 
        self.grid.step()

        # step agents
        agent_shuffle = self.agents_by_type[SugarscapeAgent].shuffle()

        for agent in agent_shuffle:
            agent.age_growth()
            agent.move()
            agent.eat()
            agent.attempt_breed()
            agent.maybe_die()

        # collect model/agent data
        self.datacollector.collect(self)

    def run_model(self, step_count=1000):
        for _ in range(step_count):
            self.step()

    def allele_frequencies(self, gene_name: str) -> np.ndarray:
        """
        Return the relative frequencies of all alleles for the given gene
        across the current agent population.

        The result is a 1D numpy array of length `Gene.loci_registry[gene_name].num_alleles`
        whose entries sum to 1 (unless there are no such alleles present, in which case
        a zero array is returned).
        """
        if gene_name not in Gene.loci_registry:
            raise ValueError(f"{gene_name} is not a registered Gene subclass name")

        gene_cls = Gene.loci_registry[gene_name]
        num_alleles = gene_cls.num_alleles
        counts = np.zeros(num_alleles, dtype=float)

        # Count alleles over all agents
        for agent in self.agents_by_type[SugarscapeAgent]:
            for allele in agent.genotype.by_name.get(gene_name, []):
                # allele indices are 1-based; convert to 0-based
                idx = allele.allele - 1
                if 0 <= idx < num_alleles:
                    counts[idx] += 1

        total = counts.sum()
        if total == 0:
            return counts

        return counts / total
