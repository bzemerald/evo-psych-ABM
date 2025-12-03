from __future__ import annotations
from typing_extensions import Self
from typing import Literal, Iterable, Mapping
from dataclasses import dataclass

import numpy as np

ChromType = Literal["autosomal", "X", "Y"]
Gender = Literal["male", "female", None]

rng = np.random.default_rng()


def set_seed(seed: int):
    global rng
    rng = np.random.default_rng(seed)


# ============================================================
# Karyotype
# ============================================================

@dataclass(frozen=True)
class Karyotype:
    name: str
    # for each chromosome type, how many alleles per locus are expected
    alleles_per_locus: Mapping[ChromType, int]

    def expected_count(self, chrom: ChromType) -> int:
        # default: 0 alleles if not specified
        return self.alleles_per_locus.get(chrom, 0)

    def validate(self, coll: Genotype) -> None:
        errors: list[str] = []

        for chrom in ("autosomal", "X", "Y"):
            expected = self.expected_count(chrom)
            for locus_name, genes in coll.by_chrom[chrom].items():
                n = len(genes)
                if n != expected:
                    errors.append(
                        f"{self.name}: chrom '{chrom}', locus '{locus_name}': "
                        f"got {n} alleles, expected {expected}."
                    )

        if errors:
            raise ValueError("Karyotype violation:\n" + "\n".join(errors))


# ============================================================
# Gene
# ============================================================

class Gene:
    mutation_rate = 0.05
    name = "Gene"
    num_alleles = 4
    chromosome: ChromType = "autosomal"
    loci_registry: dict[str, type["Gene"]] = {}

    def __init_subclass__(cls) -> None:
        cls.name = cls.__name__
        if cls is not Gene:
            Gene.loci_registry[cls.name] = cls
        # initialize allele registry based on current num_alleles
        cls._init_allele_registry()

    @classmethod
    def _init_allele_registry(cls) -> None:
        """(Re)initialize the allele registry for this Gene subclass."""
        cls.allele_registry: dict[int, "Gene"] = {}
        for a in range(1, cls.num_alleles + 1):
            cls.allele_registry[a] = cls(a)

    @classmethod
    def set_num_alleles(cls, num_alleles: int) -> None:
        """
        Dynamically change the number of alleles for this Gene subclass.

        This updates `cls.num_alleles` and rebuilds `allele_registry`.
        Existing allele instances from older configurations are discarded;
        new calls will use the updated registry.
        """
        if num_alleles <= 0:
            raise ValueError("num_alleles must be a positive integer")
        cls.num_alleles = int(num_alleles)
        cls._init_allele_registry()

    def __new__(cls, allele) -> Self:
        if allele <= 0 or allele > cls.num_alleles:
            raise ValueError(
                f"Allele number out of range:\n expected between 1 and {cls.num_alleles} but got {allele}"
            )
        if allele in cls.allele_registry:
            return cls.allele_registry[allele]
        obj = super().__new__(cls)
        obj.allele = allele
        cls.allele_registry[allele] = obj
        return obj

    def __init__(self, allele: int) -> None:
        pass

    @classmethod
    def effect_array(cls):
        return np.linspace(0, 1, cls.num_alleles)

    @property
    def effect_size(self) -> float:
        return type(self).effect_array()[self.allele - 1]

    def copy(self):
        """Return a new copy of the gene. Might mutate or might not."""
        return self.mutate() if rng.random() < self.mutation_rate else self.duplicate()

    def duplicate(self):
        """Return a new duplicated gene without mutation."""
        return type(self)(self.allele)

    def mutate(self):
        """Return a new mutated gene."""
        if self.num_alleles <= 1:
            return self.duplicate()
        new_allele = self.allele
        while new_allele == self.allele:
            new_allele = rng.integers(1, self.num_alleles + 1)
        return type(self)(new_allele)

    def __str__(self) -> str:
        return f"{self.name}: {self.allele}, effect: {self.effect_size}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.allele})"

    def phenotype_with(self, *others: Gene):
        """Default phenotype: average of effect sizes."""
        return (self.effect_size + sum(o.effect_size for o in others)) / (
            1 + len(others)
        )

    @classmethod
    def new_random(cls) -> Gene:
        return cls(rng.integers(1, cls.num_alleles + 1))


# ============================================================
# LociCollection / Genotype
# ============================================================

class LociCollection:
    """
    A collection of loci, each with 0 or more alleles.
    Lookup: by_name[locus] -> list[Gene]
            by_chrom[chrom][locus] -> list[Gene]
    """

    def __init__(self, names: Iterable[str], alleles: list[Gene]) -> None:
        names = list(names)
        for name in names:
            if name not in Gene.loci_registry:
                raise ValueError(f"{name} is not a registered Gene subclass")
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicated names in {names}")

        self.by_name: dict[str, list[Gene]] = {locus: [] for locus in names}
        self.by_chrom: dict[ChromType, dict[str, list[Gene]]] = {
            chrom: {
                locus: []
                for locus in names
                if Gene.loci_registry[locus].chromosome == chrom
            }
            for chrom in ["autosomal", "X", "Y"]
        }

        for a in alleles:
            if a.name not in names:
                raise ValueError(
                    f"Allele {a} was passed in but name {a.name} is not present in names."
                )
            self.by_name[a.name].append(a)
            self.by_chrom[a.chromosome][a.name].append(a)

    @property
    def is_gendered(self) -> bool:
        '''
        Returns True if the collection defines any loci on sex chromosomes (X or Y),
        regardless of whether alleles are currently present.
        '''
        has_x = bool(self.by_chrom["X"])
        has_y = bool(self.by_chrom["Y"])
        return has_x or has_y
    
    def __str__(self) -> str:
        result = "{"
        first = True
        for k, v in self.by_name.items():
            if first:
                first = False
            else:
                result += ", "
            result += str(k)
            result += ": "
            result += str([g.allele for g in v])
        result += "}"
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({list(self.by_name.keys())}, {sum(self.by_name.values(), start=[])})"


class Genotype(LociCollection):
    def __init__(self, names: Iterable[str], alleles: list[Gene], karyotype: Karyotype):
        self.karyotype = karyotype
        super().__init__(names, alleles)
        self.karyotype.validate(self)

    @property
    def sex(self) -> Gender | None:
        has_y = any(self.by_chrom["Y"].values())
        has_x = any(self.by_chrom["X"].values())
        if has_y:
            return "male"
        if has_x:
            return "female"
        return None

    def phenotype(self, *gene_names: str):
        genes: list[Gene] = []
        for gene_name in gene_names:
            genes.extend(self.by_name[gene_name])
        if not genes:
            return 0
        return genes[0].phenotype_with(*genes[1:])


# ============================================================
# EmptyLociCollection: random genotype generator
# ============================================================

class EmptyLociCollection(LociCollection):
    """Store loci to randomly generate genotypes. Useful to initialize a model."""

    def __init__(self, names: list[str]) -> None:
        super().__init__(names, [])

    def generate_random_genotype(self, *accepted_karyotypes: Karyotype) -> Genotype:
        """
        Randomly choose one of the accepted_karyotypes and generate
        a Genotype that satisfies its allele count constraints.
        """
        if not accepted_karyotypes:
            raise ValueError("At least one karyotype must be provided.")

        karyotype = rng.choice(accepted_karyotypes)

        names = list(self.by_name.keys())
        genes: list[Gene] = []

        for locus_name in names:
            gene_cls = Gene.loci_registry[locus_name]
            chrom: ChromType = gene_cls.chromosome

            n_alleles = karyotype.expected_count(chrom)

            for _ in range(n_alleles):
                genes.append(gene_cls.new_random())

        return Genotype(names, genes, karyotype)


# ============================================================
# Reproduction helpers
# ============================================================

def make_gamete(parent: Genotype, gamete_karyotype: Karyotype) -> Genotype:
    """
    Make a gamete from a parent genotype according to the desired gamete karyotype.

    For each locus, we pick the requested number of alleles from the parent's
    alleles at that locus (typically 1 from 2 for diploid parents).
    """
    names = list(parent.by_name.keys())
    genes: list[Gene] = []

    for locus_name in names:
        gene_cls = Gene.loci_registry[locus_name]
        chrom: ChromType = gene_cls.chromosome

        n_alleles = gamete_karyotype.expected_count(chrom)
        if n_alleles == 0:
            continue

        parent_alleles = parent.by_name[locus_name]
        if len(parent_alleles) == 0:
            raise ValueError(
                f"Parent has no alleles at locus '{locus_name}' on chrom '{chrom}', "
                f"but gamete karyotype '{gamete_karyotype.name}' expects {n_alleles}."
            )

        if n_alleles > len(parent_alleles):
            raise ValueError(
                f"Gamete karyotype '{gamete_karyotype.name}' expects {n_alleles} "
                f"alleles at locus '{locus_name}' on chrom '{chrom}', "
                f"but parent only has {len(parent_alleles)}."
            )

        chosen = rng.choice(parent_alleles, size=n_alleles, replace=False)
        for allele in np.atleast_1d(chosen):
            genes.append(allele.copy())

    return Genotype(names, genes, gamete_karyotype)


def combine_gametes(
    g1: Genotype, g2: Genotype, genome_karyotype: Karyotype
) -> Genotype:
    if set(g1.by_name.keys()) != set(g2.by_name.keys()):
        raise ValueError("Gene names mismatch between gametes")
    combined = sum(g1.by_name.values(), start=[]) + sum(g2.by_name.values(), start=[])
    return Genotype(list(g1.by_name.keys()), combined, genome_karyotype)


# ============================================================
# Karyotypes: genomes and gametes
# ============================================================

# Genomes
MALE_GENOME = Karyotype(
    name="male_genome",
    alleles_per_locus={
        "autosomal": 2,
        "X": 1,
        "Y": 1,
    },
)

FEMALE_GENOME = Karyotype(
    name="female_genome",
    alleles_per_locus={
        "autosomal": 2,
        "X": 2,
        "Y": 0,
    },
)

AGENDER_GENOME = Karyotype(
    name="agender_genome",
    alleles_per_locus={
        # Single allele per locus (haploid-style)
        "autosomal": 1,
        "X": 0,
        "Y": 0,
    },
)

# Gametes
FEMALE_GAMETE = Karyotype(
    name="female_gamete",
    alleles_per_locus={
        "autosomal": 1,
        "X": 1,
        "Y": 0,
    },
)

MALE_X_GAMETE = Karyotype(
    name="male_X_gamete",
    alleles_per_locus={
        "autosomal": 1,
        "X": 1,
        "Y": 0,
    },
)

MALE_Y_GAMETE = Karyotype(
    name="male_Y_gamete",
    alleles_per_locus={
        "autosomal": 1,
        "X": 0,
        "Y": 1,
    },
)

AGENDER_GAMETE = Karyotype(
    name="agender_gamete",
    alleles_per_locus={
        "autosomal": 1,
        "X": 0,
        "Y": 0,
    },
)
