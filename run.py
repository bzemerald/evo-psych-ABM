from model import Sugarscape
from agents import AgentParams, DefaultAgentLogics
from genetic import EmptyLociCollection, Gene


# ============================================================
# Constants for gene-driven traits
# ============================================================

VISION_MIN = 1
VISION_MAX = 5

METABOLISM_MIN = 1
METABOLISM_MAX = 5


# ============================================================
# Test genes
# ============================================================


class VisionGene(Gene):
    """Gene controlling vision."""


class MetabolismGene(Gene):
    """Gene controlling metabolism."""


class ControlGene(Gene):
    """Neutral control gene (no direct effect used in logic)."""


# ============================================================
# Agent logic using genes
# ============================================================


class GeneticAgentLogics(DefaultAgentLogics):
    """
    Agent logics that derive metabolism and vision from genotype
    using MetabolismGene and VisionGene.
    """

    def metabolism(self, agent) -> int:
        value = agent.genotype.phenotype("MetabolismGene")
        scaled = METABOLISM_MIN + value * (METABOLISM_MAX - METABOLISM_MIN)
        return max(METABOLISM_MIN, min(METABOLISM_MAX, int(round(scaled))))

    def vision(self, agent) -> int:
        value = agent.genotype.phenotype("VisionGene")
        scaled = VISION_MIN + value * (VISION_MAX - VISION_MIN)
        return max(VISION_MIN, min(VISION_MAX, int(round(scaled))))


# ============================================================
# Shared model-level objects
# ============================================================

# Empty genome with our three test loci
EMPTY_GENOME = EmptyLociCollection(
    ["VisionGene", "MetabolismGene", "ControlGene"]
)

# Default agent parameters
AGENT_PARAMS = AgentParams(
    reproduction_age=10,
    reproduction_check_radius=1,
    reproduction_cooldown=5,
)

# Agent logics instance using genes
AGENT_LOGICS = GeneticAgentLogics()

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
        agent_params=AGENT_PARAMS,
        agent_logics=AGENT_LOGICS,
    )
    params.update(kwargs)
    return Sugarscape(**params)


if __name__ == "__main__":
    # Example: run the model for a fixed number of steps when executed directly.
    model = create_model()
    model.run_model(1000)
