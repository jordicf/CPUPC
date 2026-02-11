"""Module containing constants for the floorset floorplans."""

from enum import IntEnum


class FsNetlist(IntEnum):
    """Constants for the netlist in the floorset format."""

    MODULES = 0
    MODULE2MODULE = 1
    PIN2MODULE = 2
    PIN_POS = 3


class FsConstraint(IntEnum):
    """Constants for the constraints in the floorset format."""

    AREA = 0
    HARD = 1
    FIXED = 2
    MIB = 3
    ADJ_CLUSTER = 4
    BOUNDARY = 5
