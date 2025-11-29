from pathlib import Path

import numpy as np

import mesa
from mesa.discrete_space import OrthogonalVonNeumannGrid
from mesa.discrete_space.property_layer import PropertyLayer
from agents import SugarscapeAgent, AgentParams, AgentLogicProtocol
from genetic import *


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
        empty_genome: EmptyLociCollection,
        agent_params: AgentParams,
        agent_logics: AgentLogicProtocol,
        width=50,
        height=50,
        initial_population=200,
        endowment_min=25,
        endowment_max=50,
        vision_min: int = 1,
        vision_max: int = 5,
        metabolism_min: int = 1,
        metabolism_max: int = 5,
        mutation_rate: float = 0.05,
        seed=None,
    ):
        super().__init__(seed=seed)
        if seed:
            set_seed(seed) # sync up with numpy's rng in the genetic module
        # grid size
        self.width, self.height = width, height
        self.empty_genome = empty_genome
        self.gene_names = list(self.empty_genome.by_name.keys())
        self.is_gendered = empty_genome.is_gendered
        self.agent_params, self.agent_logics = agent_params, agent_logics
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

        # grid
        self.grid = OrthogonalVonNeumannGrid(
            (self.width, self.height), torus=False, random=self.random
        )

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

        # sugar landscape from file
        self.sugar_distribution = np.genfromtxt(
            Path(__file__).parent / "sugar-map.txt"
        )

        self.grid.add_property_layer(
            PropertyLayer.from_data("sugar", self.sugar_distribution)
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
            sugar=self.rng.integers(
                endowment_min, endowment_max, (initial_population,), endpoint=True
            )
        )

    def step(self):
        """
        Step function for Sugarscape:
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
