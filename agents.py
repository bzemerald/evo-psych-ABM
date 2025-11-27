import math

from mesa.discrete_space import CellAgent


# Helper function
def get_distance(cell_1, cell_2):
    """
    Calculate the Euclidean distance between two positions

    used in trade.move()
    """

    x1, y1 = cell_1.coordinate
    x2, y2 = cell_2.coordinate
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx**2 + dy**2)

class SugarscapeAgent(CellAgent):
    """
    SugarscapeAgent:
    - has sugar and a sugar metabolism
    - moves to harvest sugar
    """

    def __init__(self, model, cell, sugar=0, metabolism=0, vision=0):
        super().__init__(model)
        self.cell = cell
        self.sugar = sugar
        self.metabolism = metabolism
        self.vision = vision

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
        if self.is_starved():
            self.remove()