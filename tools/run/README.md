# `cpupc run`

Run the whole CPUPC flow in a single command.

Usage:
    cpupc run --flow FLOW [global-options(die,netlist,output)] [flow-options] 

Flows:
    qp              Quadratic placement (initial floorplanning)
    inirects        Initial rectangles (contract & expand)
    cvx_legalizer   Convex legalizer
    fine_tune       Non-convex legalizer

Execute `cpupc run --help` for help.
