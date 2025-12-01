from model import Sugarscape
from agents import AgentParams, DefaultAgentLogics, SugarscapeAgent
from genetic import EmptyLociCollection, Gene


# ============================================================
# genes
# ============================================================


class VisionGene(Gene):
    """Gene controlling vision."""


class MetabolismGene(Gene):
    """Gene controlling metabolism."""

class StrategyGene(Gene):
    """conbining vison and metabolism"""

class FlexibilityGene(Gene):
    """Controls the weight of nature vs nurture in decision making"""

class ControlGene(Gene):
    """Neutral control gene (no direct effect used in logic)."""


# ============================================================
# Agent logic
# ============================================================


class BasicAgentLogics(DefaultAgentLogics):
    """
    Agent logics that derive metabolism and vision from genotype
    using MetabolismGene and VisionGene.
    """

    @staticmethod
    def _allele_to_int(allele_index: int, min_val: int, max_val: int, num_alleles: int) -> int:
        """Map an allele index (1..num_alleles) to an integer in [min_val, max_val]."""
        if num_alleles <= 1:
            return int(min_val)
        frac = (allele_index - 1) / (num_alleles - 1)
        return int(round(min_val + frac * (max_val - min_val)))

    def _gene_trait(self, agent, gene_name: str, min_val: int, max_val: int) -> int:
        """Compute an integer trait value from the given gene using the agent's genotype."""
        genes = agent.genotype.by_name.get(gene_name, [])
        if not genes:
            return int(round((min_val + max_val) / 2))
        num_alleles = type(genes[0]).num_alleles
        values = [
            self._allele_to_int(g.allele, min_val, max_val, num_alleles) for g in genes
        ]
        trait = int(round(sum(values) / len(values)))
        return max(min_val, min(max_val, trait))

    def metabolism(self, agent) -> float:
        m_min = int(agent.model.metabolism_min)
        m_max = int(agent.model.metabolism_max)
        return self._gene_trait(agent, "MetabolismGene", m_min, m_max)

    def vision(self, agent) -> int:
        v_min = int(agent.model.vision_min)
        v_max = int(agent.model.vision_max)
        return self._gene_trait(agent, "VisionGene", v_min, v_max)
    
    def can_breed(self, agent: SugarscapeAgent) -> bool:
        return agent.sugar > 20 and super().can_breed(agent)

class StrategicAgentLogics(BasicAgentLogics):
    def genetic_strategy(self, agent:SugarscapeAgent) -> float:
        return agent.genotype.phenotype("StrategyGene")
    
    def cultural_strategy(self, agent: SugarscapeAgent) -> float:
        return agent.random.random() #TODO
    
    def strategy(self, agent: SugarscapeAgent) -> float:
        '''
        A float between 0 to 1 to denote the foraging strategy.
        0 -> high vision, low metabolism
        1 -> low vision, high metabolism
        '''
        return self.genetic_strategy(agent)

    def metabolism(self, agent: SugarscapeAgent) -> float:
        minm, maxm = agent.params.min_metabolism, agent.params.max_metabolism
        return minm + (maxm - minm) * self.strategy(agent)

    def vision(self, agent: SugarscapeAgent) -> int:
        minv, maxv = agent.params.min_vision, agent.params.max_vision
        return round(minv + (maxv - minv) * self.strategy(agent))


    

class FlexibleAgentLogics(StrategicAgentLogics):
    def strategy(self, agent: SugarscapeAgent) -> float:
        cweight = agent.genotype.phenotype("FlexibilityGene")
        gweight = 1 - cweight
        return self.cultural_strategy(agent) * cweight + self.genetic_strategy(agent) * gweight
    
    
# ============================================================
# Shared model-level objects
# ============================================================

EMPTY_GENOME = EmptyLociCollection(
    ["ControlGene", "StrategyGene"]
)

AGENT_PARAMS = AgentParams()

AGENT_LOGICS = StrategicAgentLogics()

def create_model(**kwargs) -> Sugarscape:
    """
    Helper to create a Sugarscape model instance.

    By default it wires in:
    - EMPTY_GENOME (with VisionGene, MetabolismGene, ControlGene)
    - AGENT_PARAMS
    - AGENT_LOGICS

    Any keyword arguments override these defaults.
    """
    params = dict(
        empty_genome=EMPTY_GENOME,
        agent_logics=AGENT_LOGICS,
        initial_sugar=AGENT_PARAMS.initial_sugar,
        reproduction_age=AGENT_PARAMS.reproduction_age,
        reproduction_check_radius=AGENT_PARAMS.reproduction_check_radius,
        reproduction_cooldown=AGENT_PARAMS.reproduction_cooldown,
        max_sugar=AGENT_PARAMS.max_sugar,
        max_children=AGENT_PARAMS.max_children,
        max_age=AGENT_PARAMS.max_age,
    )
    params.update(kwargs)
    return Sugarscape(**params)
