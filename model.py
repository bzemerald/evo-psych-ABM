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
            self.sugar.data + self.regen_amount * mask, self.sugar_capacity # type: ignore
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
        spike_size: int = 9,
        spike_decrement: int = 3,
        spike_freq: float = 0.05,
        base: float = 0.0,
        ini_sugar_map=None,

        # Initialization params
        initial_population: int = 200,
        endowment: int = 10,

        # Agent params
        initial_sugar: int = 0,
        reproduction_age: int = 10,
        reproduction_check_radius: int = 1,
        reproduction_cooldown: int = 5,
        min_reproduction_sugar: float = 20,
        max_age: int = 80,
        max_children: int = 100,
        vision_min: int = 1,
        vision_max: int = 5,
        metabolism_min: int = 1,
        metabolism_max: int = 5,
        mutation_rate: float = 0.05,
        num_alleles: int = 5,
        starvation_punishment:float = 1,
        seed=None,
    ):
        super().__init__(seed=seed)

        # Per-step cultural tracking
        # Stores strategies and net sugar gains from the *previous* step.
        self.last_step_strategies: list[float] = []
        self.last_step_net_gains: list[float] = []
        # Cultural strategy value used by agents in the current step.
        self.cultural_strategy_value: float | None = None

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
                spike_size=spike_size,
                spike_decrement=spike_decrement,
                spike_freq=spike_freq,
                base=base,
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
            min_reproduction_sugar=min_reproduction_sugar,
            max_age=max_age,
            max_children=max_children,
            starvation_punishment=starvation_punishment
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
        for name in self.empty_genome.by_name.keys():
            cls = Gene.loci_registry.get(name)
            if cls is not None:
                cls.set_num_alleles(num_alleles)

        # datacollector: collect agent count, average genetic strategy, and allele frequencies for all genes
        model_reporters: dict[str, object] = {
            "#Agents": lambda m: len(m.agents),
        }
        model_reporters["avg_genetic_strategy"] = (
            lambda m: np.mean(
                [m.agent_logics.genetic_strategy(a) for a in m.agents_by_type[SugarscapeAgent]]
            )
            if len(m.agents_by_type[SugarscapeAgent]) > 0
            else 0.0
        )
        model_reporters["avg_cultural_strategy"] = (
            lambda m: float(m.cultural_strategy_value)
            if getattr(m, "cultural_strategy_value", None) is not None
            else 0.0
        )
        model_reporters["avg_total_strategy"] = (
            lambda m: np.mean(
                [m.agent_logics.strategy(a) for a in m.agents_by_type[SugarscapeAgent]]
            )
            if hasattr(m.agent_logics, "strategy")
            and len(m.agents_by_type[SugarscapeAgent]) > 0
            else 0.0
        )
        model_reporters["avg_flexibility_effect"] = (
            lambda m: np.mean(
                [
                    a.genotype.phenotype("FlexibilityGene")
                    for a in m.agents_by_type[SugarscapeAgent]
                ]
            )
            if len(m.agents_by_type[SugarscapeAgent]) > 0
            else 0.0
        )
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

        # Initialize cultural strategy to the average genetic strategy
        # so that the first step has a sensible value.
        agents = self.agents_by_type[SugarscapeAgent]
        if len(agents) > 0 and hasattr(self.agent_logics, "genetic_strategy"):
            self.cultural_strategy_value = float(
                np.mean(
                    [self.agent_logics.genetic_strategy(a) for a in agents]  # type: ignore[attr-defined]
                )
            )
        else:
            self.cultural_strategy_value = 0.5

    def step(self):
        """
        Step function for Sugarscape:
        1) regenerate sugar
        2) move/eat/die for agents
        3) collect data
        """
        # Snapshot strategies and sugar at the beginning of the step
        agents_start = list(self.agents_by_type[SugarscapeAgent])
        has_strategy = hasattr(self.agent_logics, "strategy")
        start_sugar = [a.sugar for a in agents_start]
        start_strategies = (
            [self.agent_logics.strategy(a) for a in agents_start]  # type: ignore[attr-defined]
            if has_strategy
            else []
        )

        # regenerate sugar 
        self.grid.step()

        # step agents
        agent_shuffle = self.agents_by_type[SugarscapeAgent].shuffle()

        for agent in agent_shuffle:
            agent.age_growth() # type: ignore[attr-defined]
            agent.move() # type: ignore[attr-defined]
            agent.eat() # type: ignore[attr-defined]
            agent.attempt_breed() # type: ignore[attr-defined]
            agent.maybe_die() # type: ignore[attr-defined]

        # Update cultural statistics based on previous step performance
        if has_strategy and agents_start:
            net_gains = [a.sugar - s for a, s in zip(agents_start, start_sugar)]
            self.last_step_strategies = start_strategies
            self.last_step_net_gains = net_gains

            # Elite subset + temporal smoothing:
            # 1) sort agents by net gain and take the top fraction as "elites"
            # 2) compute an elite-weighted average strategy
            # 3) update cultural strategy as an exponential moving average
            gains_arr = np.asarray(net_gains, dtype=float)
            if gains_arr.size > 0:
                elite_frac = 0.2
                k = max(1, int(len(gains_arr) * elite_frac))
                idx_sorted = np.argsort(gains_arr)[::-1]  # descending by net gain
                elite_idx = idx_sorted[:k]

                elite_gains = gains_arr[elite_idx]
                elite_strats = [start_strategies[i] for i in elite_idx]

                # Only reward positive performers; fall back to uniform if all <= 0
                weights = np.maximum(elite_gains, 0.0)
                total_w = float(weights.sum())
                if total_w > 0:
                    elite_mean = float(np.dot(elite_strats, weights) / total_w)
                else:
                    elite_mean = float(np.mean(elite_strats))

                # Temporal smoothing: keep cultural distinct from instantaneous genetics
                eta = 0.5
                prev = (
                    float(self.cultural_strategy_value)
                    if self.cultural_strategy_value is not None
                    else elite_mean
                )
                value = (1.0 - eta) * prev + eta * elite_mean
                self.cultural_strategy_value = max(0.0, min(1.0, value))

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
