"""Shared Pint UnitRegistry for qsp-hpc-tools.

All code that handles units must use this shared registry to ensure
quantities from different parts of the codebase can be compared and combined.
"""

import pint

# Create shared unit registry
ureg = pint.UnitRegistry()

# Define custom units for QSP modeling
ureg.define("cell = [cell_count]")
ureg.define("nanomolarity = nanomolar")
ureg.define("micromolarity = micromolar")
ureg.define("millimolarity = millimolar")
ureg.define("molarity = molar")
ureg.define("millimeter_mercury = mmHg")
