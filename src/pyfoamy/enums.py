"""Closed, validated vocabularies of OpenFOAM/pyfoamy type names."""

from enum import Enum


class CustomName(str):
    """A type name from a third-party library, which `coerce()` accepts
    in place of a member of a closed vocabulary.
    """


class OpenFoamEnum(str, Enum):
    """A `str` enum whose members render as their plain value.

    `__str__` is overridden because `Enum.__str__` otherwise wins over
    `str.__str__` (Python 3.9), and `to_grammar_value` calls `str(value)`.
    """

    def __str__(self) -> str:
        return self.value

    @classmethod
    def coerce(cls, value, kind=None):
        """Converts a plain string or member to a member of this enum.

        Args:
            kind: Noun for the error message; defaults to the class name.

        Returns:
            The member, or `value` itself if it is a `CustomName`.

        Raises:
            ValueError: If `value` matches no member.
        """
        if isinstance(value, CustomName):
            return value
        try:
            return cls(value)
        except ValueError:
            valid = sorted(member.value for member in cls)
            raise ValueError(
                f"unknown {kind or cls.__name__} {value!r}, supported "
                f"values are {valid}"
            ) from None


class SimulationType(OpenFoamEnum):
    """`TurbulenceSpec`'s `momentumTransport` `simulationType`."""

    LAMINAR = "laminar"
    RAS = "RAS"
    LES = "LES"


class TurbulenceModel(OpenFoamEnum):
    """A RAS or LES model name, validated per `SimulationType`."""

    K_EPSILON = "kEpsilon"
    REALIZABLE_KE = "realizableKE"
    RNG_K_EPSILON = "RNGkEpsilon"
    LAUNDER_SHARMA_KE = "LaunderSharmaKE"
    K_OMEGA = "kOmega"
    K_OMEGA_2006 = "kOmega2006"
    K_OMEGA_SST = "kOmegaSST"
    K_OMEGA_SST_LM = "kOmegaSSTLM"
    V2F = "v2f"
    SPALART_ALLMARAS = "SpalartAllmaras"
    SMAGORINSKY = "Smagorinsky"
    WALE = "WALE"
    K_EQN = "kEqn"
    DYNAMIC_K_EQN = "dynamicKEqn"


class ReactionModel(OpenFoamEnum):
    """`constant/reactionProperties`'s `reactionModel`."""

    NO_REACTION = "noReaction"
    LAMINAR = "laminar"
    PASR = "PaSR"
    EDC = "EDC"


class ChemistryType(OpenFoamEnum):
    """`constant/chemistryProperties`'s `type`."""

    STANDARD = "standard"


class EDCVersion(OpenFoamEnum):
    """`EDC`'s `version` coefficient (`ReactionSpec.coefficients`)."""

    V1981 = "v1981"
    V1996 = "v1996"
    V2005 = "v2005"
    V2016 = "v2016"


class SuModel(OpenFoamEnum):
    """`combustionProperties.laminarFlameSpeed`'s `model`."""

    LINEAR_EQUILIBRIUM = "linearEquilibrium"
    TRANSPORT = "transport"
    UNIFORM_CONSTANT = "uniformConstant"
    UNSTRAINED = "unstrained"


class XiModel(OpenFoamEnum):
    """`combustionProperties.flameWrinkling`'s `model`."""

    EQUILIBRIUM = "equilibrium"
    TRANSPORT = "transport"
    UNIFORM_CONSTANT = "uniformConstant"


class XiEqModel(OpenFoamEnum):
    """`flameWrinkling.eq`'s `model`."""

    CONSTANT = "constant"
    GULDER = "Gulder"
    INSTABILITY = "instability"


class XiGModel(OpenFoamEnum):
    """`flameWrinkling.generationRate`'s `model`."""

    INSTABILITY = "instability"
    KTS = "KTS"


class XiProfile(OpenFoamEnum):
    """`flameWrinkling.profile`'s `model`."""

    CUBIC = "cubic"
    LINEAR = "linear"
    UNIFORM = "uniform"


class MixtureType(OpenFoamEnum):
    """`XiFluidSetup`'s `mixture_type`: the supported (u, b) mixture pairs.

    Not an OpenFOAM name.
    """

    HOMOGENEOUS = "homogeneous"
    INHOMOGENEOUS = "inhomogeneous"
    INHOMOGENEOUS_EGR = "inhomogeneousEGR"


class TimeControlMode(OpenFoamEnum):
    """A function object's (or `controlDict`'s) `executeControl`/
    `writeControl` mode."""

    TIME_STEP = "timeStep"
    WRITE_TIME = "writeTime"
    ADJUSTABLE_RUN_TIME = "adjustableRunTime"
    RUN_TIME = "runTime"
    RUN_TIMES = "runTimes"
    CLOCK_TIME = "clockTime"
    CPU_TIME = "cpuTime"
    NONE = "none"


class VolFieldValueOperation(OpenFoamEnum):
    """A `volFieldValue` function object's `operation`."""

    NONE = "none"
    SUM = "sum"
    SUM_MAG = "sumMag"
    AVERAGE = "average"
    VOL_AVERAGE = "volAverage"
    VOL_INTEGRATE = "volIntegrate"
    MIN = "min"
    MAX = "max"
    MIN_MAG = "minMag"
    MAX_MAG = "maxMag"
    COV = "CoV"
    UI = "UI"


class SurfaceFieldValueOperation(OpenFoamEnum):
    """A `surfaceFieldValue` function object's `operation`.

    Differs from `volFieldValue`'s: it adds `orientedSum`/`area*`/
    `areaNormal*` and has no `vol*`/`cell*`.
    """

    NONE = "none"
    SUM = "sum"
    SUM_MAG = "sumMag"
    ORIENTED_SUM = "orientedSum"
    AVERAGE = "average"
    AREA_AVERAGE = "areaAverage"
    AREA_INTEGRATE = "areaIntegrate"
    MIN = "min"
    MAX = "max"
    MIN_MAG = "minMag"
    MAX_MAG = "maxMag"
    COV = "CoV"
    UI = "UI"
    AREA_NORMAL_AVERAGE = "areaNormalAverage"
    AREA_NORMAL_INTEGRATE = "areaNormalIntegrate"
