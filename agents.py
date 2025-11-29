from __future__ import annotations
import math

from dataclasses import dataclass
from typing import Protocol

from mesa.discrete_space import CellAgent, Cell

from genetic import (
    Genotype,
    make_gamete,
    combine_gametes,
    AGENDER_GENOME,
    AGENDER_GAMETE,
)


# Helper function
def get_distance(cell_1, cell_2):
    """
    Calculate the Euclidean distance between two positions

    used in .move()
    """

    x1, y1 = cell_1.coordinate
    x2, y2 = cell_2.coordinate
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx**2 + dy**2)


@dataclass(frozen=True)
class AgentParams:
    """
    A dataclass to hold agent parameters.
    """
    initial_sugar: int
    reproduction_age: int
    reproduction_check_radius: int
    reproduction_cooldown: int
    max_sugar: int
    max_children: int
    max_age: int


class AgentLogicProtocol(Protocol):
    """
    Protocol describing the pluggable agent logic behaviour.
    """

    def metabolism(self, agent: SugarscapeAgent) -> float: ...

    def vision(self, agent: SugarscapeAgent) -> int: ...

    def can_breed(self, agent: SugarscapeAgent) -> bool: ...

    def wants_to_breed_with(
        self, agent: SugarscapeAgent, other: SugarscapeAgent
    ) -> bool: ...

    def offspring_genome(
        self, agent: SugarscapeAgent, other: SugarscapeAgent
    ) -> Genotype: ...

    def sugar_donation_to_offspring(
            self, agent: SugarscapeAgent
    ) -> float: ...


@dataclass(frozen=True)
class DefaultAgentLogics(AgentLogicProtocol):
    """
    Default implementation of AgentLogicProtocol.
    """

    def metabolism(self, agent: "SugarscapeAgent") -> int:
        return agent.random.randint(1, 5)

    def vision(self, agent: "SugarscapeAgent") -> int:
        return agent.random.randint(1, 5)

    def can_breed(self, agent: SugarscapeAgent) -> bool:
        has_empty = map(
            lambda c: c.is_empty,
            agent.cell.get_neighborhood(
                agent.params.reproduction_check_radius, include_center=True
            ),
        )
        return (any(has_empty) and 
                agent.age >= agent.params.reproduction_age and
                agent.breed_cooldown == 0 and
                agent.num_children <= agent.params.max_children)

    def wants_to_breed_with(
        self, agent: "SugarscapeAgent", other: "SugarscapeAgent"
    ) -> bool:
        return True

    def offspring_genome(
        self, agent: "SugarscapeAgent", other: "SugarscapeAgent"
    ) -> Genotype:
        gamete1 = make_gamete(agent.genotype, AGENDER_GAMETE)
        gamete2 = make_gamete(other.genotype, AGENDER_GAMETE)
        return combine_gametes(gamete1, gamete2, AGENDER_GENOME)
    
    def sugar_donation_to_offspring(self, agent: SugarscapeAgent) -> float:
        return 0.25 * agent.sugar


class SugarscapeAgent(CellAgent):
    """
    SugarscapeAgent:
    - has sugar and a sugar metabolism
    - moves to harvest sugar
    """

    def __init__(
        self,
        model,
        cell: Cell,
        genotype: Genotype,
        params: AgentParams,
        logics: AgentLogicProtocol,
        sugar: float=0,
    ):
        super().__init__(model)
        self.cell: Cell = cell
        self.genotype = genotype
        self.params = params
        self.logics: AgentLogicProtocol = logics
        self.sugar = sugar
        self.breed_cooldown = 0
        self.metabolism = self.logics.metabolism(self)
        self.vision = self.logics.vision(self)
        self.age = 0
        self.num_children = 0

    def is_starved(self):
        """
        Helper function for self.maybe_die()
        """
        return self.sugar <= 0

    def move(self):
        """
        Function for agent to identify optimal move in 4 parts
        1 - identify all possible moves
        2 - determine which move maximizes sugar intake
        3 - find closest best option
        4 - move
        """

        # 1. identify all possible moves
        neighboring_cells = [
            cell
            for cell in self.cell.get_neighborhood(self.vision, include_center=True)
            if cell.is_empty
        ]

        # safety: if no moves available, stay put
        if not neighboring_cells:
            return

        # 2. determine which move maximizes sugar intake
        # welfare is just future sugar = current sugar + cell.sugar
        welfares = [self.sugar + cell.sugar for cell in neighboring_cells]

        # 3. Find closest best option
        max_welfare = max(welfares)

        candidate_indices = [
            i for i, w in enumerate(welfares) if math.isclose(w, max_welfare)
        ]
        candidates = [neighboring_cells[i] for i in candidate_indices]

        min_dist = min(get_distance(self.cell, cell) for cell in candidates)

        final_candidates = [
            cell
            for cell in candidates
            if math.isclose(get_distance(self.cell, cell), min_dist, rel_tol=1e-02)
        ]

        # 4. Move Agent
        self.cell = self.random.choice(final_candidates)

    def age_growth(self):
        self.age += 1
        if self.breed_cooldown > 0:
            self.breed_cooldown -= 1

    def eat(self):
        """
        Harvest sugar on current cell and metabolize
        """
        self.sugar += self.cell.sugar
        self.cell.sugar = 0
        self.sugar -= self.metabolism

    def maybe_die(self):
        """
        Remove agents who have consumed all their sugar
        """
        if self.is_starved() or self.age >= self.params.max_age:
            self.remove()

    def breed_with(self, other: SugarscapeAgent) -> SugarscapeAgent:
        """
        Breed with another agent to produce offspring.

        Assumes can_breed() and wants_to_breed_with() have been checked.
        """
        offspring_genome = self.logics.offspring_genome(self, other)

        cell_candidates = [
            cell
            for cell in self.cell.get_neighborhood(1, include_center=False)
            if cell.is_empty
        ]
        if not cell_candidates:
            # No available cell, breeding fails gracefully.
            return None  # type: ignore[return-value]

        cell = self.random.choice(cell_candidates)
        self_donation = self.logics.sugar_donation_to_offspring(self)
        other_donation = other.logics.sugar_donation_to_offspring(other)
        self.sugar -= self_donation
        other.sugar -= other_donation
        ini_sugar = self.params.initial_sugar + self_donation + other_donation

        offspring = SugarscapeAgent(
            model=self.model,
            cell=cell,
            genotype=offspring_genome,
            params=self.params,
            logics=self.logics,
            sugar=ini_sugar,
        )

        return offspring

    def attempt_breed(self) -> bool:
        """
        Attempt to breed with a neighboring agent within a given radius.
        Returns True if breeding was successful, False otherwise.
        """
        if not self.can_breed():
            return False

        neighbors = self.cell.get_neighborhood(
            self.params.reproduction_check_radius, include_center=False
        ).agents

        for neighbor in neighbors:
            if (
                isinstance(neighbor, SugarscapeAgent)
                and neighbor.can_breed()
                and self.wants_to_breed_with(neighbor)
                and neighbor.wants_to_breed_with(self)
            ):
                offspring = self.breed_with(neighbor)
                if offspring is not None:
                    self.breed_cooldown = self.params.reproduction_cooldown
                    neighbor.breed_cooldown = neighbor.params.reproduction_cooldown
                    self.num_children += 1
                    neighbor.num_children += 1
                    return True
        return False

    def can_breed(self) -> bool:
        if self.breed_cooldown > 0:
            return False
        return self.logics.can_breed(self)

    def wants_to_breed_with(self, other: SugarscapeAgent) -> bool:
        return self.logics.wants_to_breed_with(self, other)
