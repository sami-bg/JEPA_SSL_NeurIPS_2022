import enum
from omegaconf import OmegaConf



class ModelType(enum.Enum):
    VICReg = enum.auto()
    RSSM = enum.auto()
    SimCLR = enum.auto()
    VJEPA = enum.auto()
    HJEPA = enum.auto()

class DatasetType(enum.Enum):
    Single = enum.auto()
    Multiple = enum.auto()
