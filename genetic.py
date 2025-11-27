from __future__ import annotations
from typing_extensions import Self
import numpy as np
from typing import Literal, Iterable

ChromType = Literal['autosomal', 'X', 'Y']
Gender = Literal['male', 'female', None]
rng = np.random.default_rng()

def set_seed(seed: int):
    global rng
    rng = np.random.default_rng(seed)

class Gene():
    mutation_rate = 0.05
    name = 'Gene'
    num_alleles = 4
    chromosome: ChromType = 'autosomal' 
    loci_registry: dict[str, Gene] = {}

    def __init_subclass__(cls) -> None:
        cls.name = cls.__name__
        if cls is not Gene: Gene.loci_registry[cls.name] = cls
        cls.allele_registry: dict[int, Gene] = {} # store instances in a dict as a class attribute to reuse
        for a in range(1, cls.num_alleles + 1):
            cls.allele_registry[a] = cls(a)

    def __new__(cls, allele) -> Self:
        if allele <=0 or allele > cls.num_alleles: raise ValueError(f"Allele number out of range:\n expected between 1 and {cls.num_alleles} but got {allele}")
        if allele in cls.allele_registry: return cls.allele_registry[allele]
        obj = super().__new__(cls)
        obj.allele = allele
        cls.allele_registry[allele] = obj
        return obj

    def __init__(self, allele: int) -> None:
        self.allele = allele

    @classmethod
    def effect_array(cls):
        return np.linspace(0, 1, cls.num_alleles)
    
    @property
    def effect_size(self) -> float:
        return type(self).effect_array()[self.allele - 1]
    
    def copy(self):
        '''Return a new copy of the gene. Might mutate or might not.'''
        return self.mutate() if rng.random() < self.mutation_rate else self.duplicate()
    
    def duplicate(self): 
        '''Return a new duplicated gene without mutation in new_genome.'''
        return type(self)(self.allele) # make it work in subclasses
        
    def mutate(self):
        '''Return a new mutated gene in new_genome.'''
        new_allele = self.allele
        while new_allele == self.allele:
            new_allele = rng.integers(1, self.num_alleles + 1)
        return type(self)(new_allele)
    
    def __str__(self) -> str:
        return f"{self.name}: {self.allele}, effect: {self.effect_size}"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.allele})"
    
    def phenotype_with(self, *others: Gene):
        '''The default way to calculate phenotype is to take the average'''
        # if not all([type(o) is type(self) for o in others]): raise TypeError(f"At least one or more other gene is not the type {type(self)}.")
        return (self.effect_size + sum([o.effect_size for o in others])) / (1 + len(others))
    
    @classmethod
    def new_random(cls) -> Gene:
        return cls(rng.integers(1, cls.num_alleles + 1))


class LociCollection():
    '''A collection of Loci, each with 0 or more alleles. Lookup the list of alleles with name.'''
    # The range of posssible numbers of alleles; override in subclasses
    auto_alleles = (0, float('inf'))
    x_alleles = (0, float('inf'))
    y_alleles = (0, float('inf'))
    
    def __init__(self, names: Iterable, alleles: list[Gene]) -> None:
        names = list(names)
        for name in names:
            if name not in Gene.loci_registry:
                raise ValueError(f"{name} is not a registered Gene subclass")
        if len(set(names)) != len(names): raise ValueError(f"Duplicated names in {names}")
        self.by_name: dict[str, list[Gene]] = {locus: [] for locus in names}
        self.by_chrom: dict[ChromType, dict[str, list[Gene]]] = {
            chrom: {
                locus: [] for locus in names if Gene.loci_registry[locus].chromosome == chrom
            } for chrom in ['autosomal', 'X', 'Y']
        }
        for a in alleles:
            if a.name not in names: raise ValueError(f"Allele {a} was passed in but name {a.name} is not present in names.")
            self.by_name[a.name].append(a)
            self.by_chrom[a.chromosome][a.name].append(a)

        self._check_allele_counts()

    def _check_allele_counts(self) -> None:
        """基类默认只做简单的 per-chrom min/max 检查，并给出详细错误信息。"""
        errors: list[str] = []
        def check_range(
            chrom: ChromType,
            limits: tuple[int, float]
        ) -> None:
            min_allowed, max_allowed = limits
            for locus_name, genes in self.by_chrom[chrom].items():
                n = len(genes)
                if not (min_allowed <= n <= max_allowed):
                    # 构造更清晰的提示：染色体类型 / locus 名 / 实际数量 / 允许范围
                    if max_allowed == float("inf"):
                        range_str = f"≥ {min_allowed}"
                    else:
                        range_str = f"between {min_allowed} and {max_allowed}"
                    errors.append(
                        f"Chromosome '{chrom}', locus '{locus_name}': "
                        f"got {n} alleles, expected {range_str}."
                    )

        check_range('autosomal', self.auto_alleles)
        check_range('X', self.x_alleles)
        check_range('Y', self.y_alleles)

        if errors:
            # 汇总所有有问题的 locus
            msg = "Incorrect number of alleles for one or more loci:\n" + "\n".join(errors)
            raise ValueError(msg)

    def __str__(self) -> str:
        result = "{"
        first = True
        for k, v in self.by_name.items():
            if first: first = False
            else: result += ", "
            result += str(k)
            result += ": "
            result += str([g.allele for g in v])
        result += "}"
        return result
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({list(self.by_name.keys())}, {sum(self.by_name.values(), start=[])})"

class EmptyLociCollection(LociCollection):
    '''Store loci to randomly generate genomes. Useful to initialize model.'''
    auto_alleles = (0,0)
    x_alleles = (0, 0)
    y_alleles = (0, 0)

    def __init__(self, names: list[str]) -> None:
        super().__init__(names, [])

    def generate_random_genome(self, sex: Gender|None = None) -> Genome:
        names = list(self.by_name.keys())
        genes = []
        if sex is None: sex = rng.choice(['male', 'female'])
        for n in names:
            g = Gene.loci_registry[n]
            chrom: ChromType = g.chromosome
            append_g = lambda: genes.append(g.new_random())
            if chrom == 'autosomal':
                append_g()
                append_g()
            elif chrom == 'X':
                append_g()
                if sex == 'female':
                    append_g()
            elif chrom == 'Y' and sex == 'male':
                append_g()
        return Genome(names, genes)

class Zygote(LociCollection):
    '''A collection of loci that can combine to form genomes.'''
    auto_alleles = (1, 1)
    x_alleles = (0, 1)
    y_alleles = (0, 1)

    def _check_allele_counts(self) -> None:
        super()._check_allele_counts()
        # 不能同时有X & Y
        if any(self.by_chrom['X'].values()) and any(self.by_chrom['Y'].values()): 
            raise ValueError("Zygote cannot have both X and Y genes.")
        
    def combine_with(self, other: Zygote) -> Genome:
        if set(other.by_name.keys()) != set(self.by_name.keys()): 
            raise ValueError(f"Gene names mismatch. self: {set(self.by_name.keys())}, other: {set(other.by_name.keys())}")
        combined = sum(self.by_name.values(), start=[]) + sum(other.by_name.values(), start=[])
        return Genome(list(self.by_name.keys()), combined)
    
class BlobGenome(LociCollection):
    '''
    Genome for asexual agents that reproduce through fission.
    There will only be one allele for autosomal chroms.
    '''
    auto_alleles = (1, 1)
    x_alleles = (0, 0)
    y_alleles = (0, 0)

    def reproduce(self) -> BlobGenome:
        names = self.by_name.keys()
        alleles = []
        for a in self.by_name.values():
            alleles.append(a.copy())
        return BlobGenome(names, alleles)
    
    def phenotype(self, *gene_names: str):
        genes = []
        for gene_name in gene_names:
            genes.extend(self.by_name[gene_name])
        if not genes: return 0
        else: return genes[0].phenotype_with(*genes[1:])

class Genome(LociCollection):
    '''A collection of gene pairs that can produce zygotes.'''
    auto_alleles = (2, 2)
    x_alleles = (1, 2)
    y_alleles = (0, 1)
    
    def _check_allele_counts(self) -> None:
        super()._check_allele_counts()
        if any(map(lambda g: len(g) == 2, self.by_chrom['X'].values())) and any(self.by_chrom['Y'].values()):
            raise ValueError("Genome cannot have 2 alleles on X chromosome loci and alleles on Y chromosome at the same time.")
        elif not any(self.by_chrom['Y'].values()) and any(map(lambda g: len(g) < 2, self.by_chrom['X'].values())):
            raise ValueError("Female must have 2 alleles on X chromosome loci. Got less than 2.")

    @property
    def sex(self) -> Gender|None:
        has_y = any([len(g) > 0 for g in self.by_chrom["Y"].values()])
        if has_y: return 'male'
        has_x = any([len(g) > 0 for g in self.by_chrom["X"].values()])
        return 'female' if has_x else None
    
    def create_zygote(self) -> Zygote:
        genes = []
        if self.sex == 'male':
            chrom = sum(self.by_chrom[rng.choice(['X', 'Y'])].values(), start=[]) # 雄性随机选 X 或 Y 一整套
            for g in chrom:
                genes.append(g.copy())
        gene_pairs = list(self.by_chrom['autosomal'].values()) + (
            list(self.by_chrom['X'].values()) if self.sex == 'female' else []
            ) # 雌性不选整套因为1. 不知道两条X哪个来自父哪个母 2. X交叉互换
        for gp in gene_pairs:
            old = rng.choice([gp[0], gp[1]])
            new = old.copy()
            genes.append(new)
        return Zygote(list(self.by_name.keys()), genes)
    
    def phenotype(self, *gene_names: str):
        genes = []
        for gene_name in gene_names:
            genes.extend(self.by_name[gene_name])
        if not genes: return 0
        else: return genes[0].phenotype_with(*genes[1:])