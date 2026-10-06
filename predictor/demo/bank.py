"""Synthetic demo course: Fluid Mechanics syllabus and question bank.

Everything here is invented for demonstration and testing. The papers generated from it
follow a known random process (see generator.py) so tests can check that the models
recover planted patterns.
"""

SYLLABUS_TEXT = """FLUID MECHANICS (CE 501)
Course Objectives:
1. To provide fundamental knowledge of fluid properties, fluid statics and fluid dynamics.
2. To apply conservation laws to flow measurement and pipe flow problems.

1. Properties of Fluids (4 hours)
1.1 Fluid and continuum: definition of fluid, continuum concept, density, specific weight, specific gravity
1.2 Viscosity: Newton's law of viscosity, dynamic and kinematic viscosity, Newtonian and non-Newtonian fluids
1.3 Surface tension and capillarity: capillary rise, pressure inside a droplet and a bubble
1.4 Compressibility and bulk modulus, vapour pressure, cavitation

2. Fluid Statics (10 hours)
2.1 Pressure and its measurement: Pascal's law, hydrostatic law, absolute and gauge pressure, piezometer, U-tube manometer, differential manometer
2.2 Hydrostatic forces on plane and curved surfaces, centre of pressure
2.3 Buoyancy and floatation: Archimedes' principle, metacentre, metacentric height, stability of floating and submerged bodies

3. Fluid Kinematics (6 hours)
3.1 Types of flow: steady and unsteady flow, uniform and non-uniform flow, laminar and turbulent flow, streamline, pathline and streakline
3.2 Continuity equation in one and three dimensions
3.3 Velocity potential and stream function, flow net, vortex motion

4. Fluid Dynamics (10 hours)
4.1 Euler's equation of motion and Bernoulli's equation: derivation, assumptions and applications
4.2 Flow measurement: venturimeter, orifice meter, pitot tube
4.3 Flow through orifices and mouthpieces, notches and weirs
4.4 Momentum equation and its applications: force on pipe bends, impact of jets

5. Flow Through Pipes (8 hours)
5.1 Laminar flow: Hagen-Poiseuille equation, flow between parallel plates
5.2 Darcy-Weisbach equation, friction factor, Moody diagram
5.3 Minor losses: sudden expansion, sudden contraction, bends and fittings
5.4 Pipes in series and parallel, equivalent pipe, hydraulic gradient line and total energy line

6. Dimensional Analysis and Similitude (5 hours)
6.1 Dimensional homogeneity, Rayleigh's method, Buckingham pi theorem
6.2 Dimensionless numbers: Reynolds, Froude, Euler, Weber and Mach numbers; model laws and similitude

7. Boundary Layer Theory (5 hours)
7.1 Boundary layer concept, boundary layer thickness, displacement thickness and momentum thickness
7.2 Boundary layer separation and its control, drag and lift on immersed bodies

Practical:
1. Determination of metacentric height of a floating body
2. Verification of Bernoulli's theorem
3. Calibration of a venturimeter

References:
1. Modi P.N. and Seth S.M., Hydraulics and Fluid Mechanics, Standard Book House
2. Bansal R.K., A Textbook of Fluid Mechanics and Hydraulic Machines, Laxmi Publications

Marks distribution
Chapter   Hours   Marks
1         4        8
2         10       16
3         6        10
4         10       18
5         8        14
6         5        8
7         5        6
"""

# Topic key -> (syllabus title prefix used to find the node, list of (type, text, marks)).
# A shared "family" suffix (#f1) marks paraphrases of the same underlying question.
BANK: dict[str, list[tuple[str, str, float, str]]] = {
    "fluid_continuum": [
        ("definition", "Define fluid. Explain the continuum concept in fluid mechanics.", 4, "f1"),
        ("definition", "What do you mean by the continuum assumption? Define density, specific weight and specific gravity.", 4, "f2"),
        ("numerical", "A liquid has a specific weight of 8.5 kN/m3. Calculate its density and specific gravity.", 4, "f3"),
        ("definition", "Define a fluid and state how it differs from a solid under shear stress.", 3, "f1b"),
    ],
    "viscosity": [
        ("conceptual_explanation", "Define viscosity. Explain Newton's law of viscosity with a neat sketch.", 6, "f1"),
        ("definition", "State Newton's law of viscosity and distinguish between dynamic and kinematic viscosity.", 5, "f2"),
        ("numerical", "A plate 0.025 mm distant from a fixed plate moves at 60 cm/s and requires a force of 2 N/m2 to maintain this speed. Determine the dynamic viscosity of the fluid between the plates.", 6, "f3"),
        ("compare_contrast", "Differentiate between Newtonian and non-Newtonian fluids with a rheological diagram.", 5, "f4"),
        ("numerical", "The space between two parallel plates 12 mm apart is filled with oil of viscosity 1.5 Pa.s. Calculate the shear stress when the upper plate moves at 2.5 m/s.", 6, "f5"),
    ],
    "surface_tension": [
        ("derivation", "Derive an expression for the capillary rise of a liquid in a small glass tube.", 6, "f1"),
        ("conceptual_explanation", "Explain surface tension and obtain the pressure inside a liquid droplet.", 5, "f2"),
        ("numerical", "Calculate the capillary rise of water in a glass tube of 3 mm diameter if the surface tension is 0.0735 N/m.", 5, "f3"),
    ],
    "compressibility": [
        ("definition", "Define bulk modulus and compressibility of a fluid.", 4, "f1"),
        ("conceptual_explanation", "What is vapour pressure? Explain how cavitation occurs in flowing liquids.", 5, "f2"),
        ("numerical", "Calculate the bulk modulus of a liquid whose volume decreases by 0.15 percent when the pressure increases from 70 bar to 130 bar.", 5, "f3"),
    ],
    "pressure_measurement": [
        ("derivation", "State Pascal's law and prove it for a fluid at rest.", 6, "f1"),
        ("conceptual_explanation", "Explain absolute pressure, gauge pressure and vacuum pressure with a neat diagram.", 5, "f2"),
        ("numerical", "A U-tube differential manometer containing mercury connects two pipes carrying water. The difference in mercury levels is 15 cm. Calculate the pressure difference between the pipes.", 8, "f3"),
        ("diagram", "With a neat sketch, explain the working of a U-tube manometer and a differential manometer.", 6, "f4"),
        ("derivation", "Derive the hydrostatic law for the variation of pressure in a static fluid.", 6, "f5"),
    ],
    "hydrostatic_forces": [
        ("derivation", "Derive an expression for the total pressure and the centre of pressure on a vertical plane surface submerged in a liquid.", 8, "f1"),
        ("numerical", "A rectangular gate 2 m wide and 3 m deep is placed vertically in water with its top edge 1.5 m below the free surface. Determine the total pressure force and the depth of the centre of pressure.", 8, "f2"),
        ("derivation", "Obtain expressions for the horizontal and vertical components of the hydrostatic force on a curved surface.", 7, "f3"),
        ("numerical", "A circular plate of 2.4 m diameter is immersed in water with its greatest and least depths below the free surface being 3 m and 1 m. Find the total pressure on one face and the position of the centre of pressure.", 8, "f4"),
    ],
    "buoyancy": [
        ("definition", "State Archimedes' principle. Define metacentre and metacentric height.", 4, "f1"),
        ("derivation", "Derive an expression for the metacentric height of a floating body by the analytical method.", 8, "f2"),
        ("conceptual_explanation", "Explain the conditions of stability of floating and submerged bodies with neat sketches.", 6, "f3"),
        ("numerical", "A wooden block 4 m long, 2 m wide and 1 m deep floats horizontally in water. If the specific gravity of wood is 0.7, find the metacentric height of the block.", 8, "f4"),
    ],
    "flow_types": [
        ("compare_contrast", "Differentiate between steady and unsteady flow, and uniform and non-uniform flow with examples.", 6, "f1"),
        ("definition", "Define streamline, pathline and streakline. When are they identical?", 5, "f2"),
        ("compare_contrast", "Distinguish between laminar and turbulent flow with suitable examples.", 4, "f3"),
    ],
    "continuity": [
        ("derivation", "Derive the continuity equation for a three-dimensional flow in Cartesian coordinates.", 8, "f1"),
        ("derivation", "Obtain the one-dimensional continuity equation for steady flow through a stream tube.", 5, "f2"),
        ("numerical", "Water flows through a pipe that tapers from 30 cm diameter to 15 cm diameter. If the velocity at the larger section is 2.5 m/s, calculate the discharge and the velocity at the smaller section.", 6, "f3"),
        ("numerical", "The velocity components in a flow field are u = 2x and v = -2y. Check whether the flow satisfies the continuity equation.", 5, "f4"),
    ],
    "potential_flow": [
        ("definition", "Define velocity potential and stream function. Show that the stream lines and equipotential lines are orthogonal.", 8, "f1"),
        ("conceptual_explanation", "What is a flow net? Explain its uses and limitations.", 5, "f2"),
        ("compare_contrast", "Differentiate between forced vortex and free vortex flow.", 5, "f3"),
        ("numerical", "The stream function for a two-dimensional flow is 3xy. Find the velocity at the point (2, 3) and the velocity potential function.", 6, "f4"),
    ],
    "bernoulli": [
        ("derivation", "Derive Bernoulli's equation from Euler's equation of motion along a streamline. State the assumptions made.", 8, "f1"),
        ("derivation", "Starting from Euler's equation of motion, obtain Bernoulli's equation and list its assumptions.", 8, "f1"),
        ("numerical", "Water flows through a pipe of 20 cm diameter at section 1 and 10 cm diameter at section 2, which is 5 m above section 1. The pressure at section 1 is 300 kPa and the discharge is 50 litres/s. Calculate the pressure at section 2 using Bernoulli's equation.", 10, "f2"),
        ("application", "Explain the practical applications of Bernoulli's equation with suitable examples.", 6, "f3"),
        ("numerical", "A pipe carrying water has a diameter of 25 cm at a point where the pressure is 150 kPa and the velocity is 3 m/s. At another point 4 m higher the diameter is 15 cm. Determine the pressure at the higher point.", 10, "f4"),
    ],
    "flow_measurement": [
        ("derivation", "Derive an expression for the discharge through a venturimeter.", 8, "f1"),
        ("diagram", "Explain the working principle of a pitot tube with a neat sketch.", 5, "f2"),
        ("numerical", "A horizontal venturimeter with inlet diameter 20 cm and throat diameter 10 cm is used to measure the flow of water. The pressure difference is 18 cm of mercury. Calculate the discharge if Cd is 0.98.", 8, "f3"),
        ("compare_contrast", "Compare a venturimeter and an orifice meter for flow measurement.", 5, "f4"),
    ],
    "orifices_notches": [
        ("derivation", "Derive an expression for the discharge over a rectangular notch.", 6, "f1"),
        ("derivation", "Obtain the expression for discharge over a triangular notch and state its advantages over a rectangular notch.", 7, "f2"),
        ("definition", "Define hydraulic coefficients of an orifice: coefficient of contraction, coefficient of velocity and coefficient of discharge.", 5, "f3"),
        ("numerical", "Find the discharge over a rectangular weir 2 m long when the head over the crest is 0.4 m. Take Cd as 0.62.", 6, "f4"),
    ],
    "momentum": [
        ("derivation", "State the impulse-momentum principle and derive an expression for the force exerted by a flowing fluid on a pipe bend.", 8, "f1"),
        ("numerical", "A jet of water 5 cm in diameter strikes a fixed flat plate normally with a velocity of 20 m/s. Determine the force exerted on the plate.", 6, "f2"),
        ("application", "Explain the applications of the momentum equation in engineering with examples.", 5, "f3"),
    ],
    "laminar_flow": [
        ("derivation", "Derive the Hagen-Poiseuille equation for laminar flow through a circular pipe.", 8, "f1"),
        ("derivation", "Obtain the velocity distribution for laminar flow between two fixed parallel plates.", 8, "f2"),
        ("numerical", "Oil of viscosity 0.1 Pa.s flows through a 10 cm diameter pipe at 2 litres/s. Calculate the pressure drop per metre length assuming laminar flow.", 6, "f3"),
    ],
    "darcy": [
        ("derivation", "Derive the Darcy-Weisbach equation for head loss due to friction in a pipe.", 8, "f1"),
        ("numerical", "Calculate the head loss due to friction in a pipe of 300 mm diameter and 450 m length carrying water at 2.5 m/s. Take the friction factor as 0.02.", 6, "f2"),
        ("conceptual_explanation", "Explain the Moody diagram and its use in pipe flow problems.", 5, "f3"),
        ("derivation", "Obtain an expression for the loss of head due to friction in pipes (Darcy-Weisbach formula).", 8, "f1"),
    ],
    "minor_losses": [
        ("derivation", "Derive an expression for the loss of head due to sudden expansion of a pipe.", 7, "f1"),
        ("conceptual_explanation", "Explain the various minor losses in pipe flow.", 5, "f2"),
        ("numerical", "A pipe of 15 cm diameter suddenly enlarges to 30 cm. If the discharge is 60 litres/s, calculate the head loss due to sudden enlargement.", 6, "f3"),
    ],
    "pipe_networks": [
        ("conceptual_explanation", "Explain the hydraulic gradient line and the total energy line with a neat sketch.", 5, "f1"),
        ("numerical", "Three pipes of lengths 800 m, 500 m and 400 m with diameters 500 mm, 400 mm and 300 mm are connected in series. Find the diameter of a single equivalent pipe of length 1700 m.", 8, "f2"),
        ("derivation", "Derive the condition for pipes in parallel and explain the concept of an equivalent pipe.", 6, "f3"),
    ],
    "dimensional_analysis": [
        ("derivation", "State the Buckingham pi theorem. Using it, show that the drag force on a sphere depends on the Reynolds number.", 8, "f1"),
        ("conceptual_explanation", "Explain dimensional homogeneity and Rayleigh's method of dimensional analysis.", 6, "f2"),
        ("numerical", "Using the Buckingham pi theorem, obtain an expression for the discharge over a weir in terms of head, gravity, viscosity and density.", 8, "f3"),
    ],
    "similitude": [
        ("definition", "Define Reynolds number, Froude number, Weber number and Mach number and state their significance.", 6, "f1"),
        ("conceptual_explanation", "Explain geometric, kinematic and dynamic similarity in model studies.", 5, "f2"),
        ("numerical", "A model of a dam is built to a scale of 1:25. If the flow over the model spillway is 0.2 m3/s, find the corresponding flow over the prototype using Froude's model law.", 6, "f3"),
    ],
    "boundary_layer": [
        ("definition", "Define boundary layer thickness, displacement thickness and momentum thickness.", 6, "f1"),
        ("conceptual_explanation", "Explain the development of a boundary layer over a flat plate with a neat sketch.", 6, "f2"),
        ("derivation", "Derive an expression for the displacement thickness of a boundary layer.", 6, "f3"),
    ],
    "separation_drag": [
        ("conceptual_explanation", "Explain boundary layer separation and the methods to control it.", 6, "f1"),
        ("compare_contrast", "Differentiate between drag and lift on an immersed body.", 4, "f2"),
        ("numerical", "A flat plate 1.5 m by 1.5 m moves at 50 km/h in still air of density 1.15 kg/m3. If the drag coefficient is 0.15, calculate the drag force.", 6, "f3"),
    ],
}

# Topics from an older syllabus that are no longer in the course (must never be predicted).
OUT_OF_SYLLABUS_BANK: dict[str, list[tuple[str, str, float, str]]] = {
    "turbines": [
        ("conceptual_explanation", "Explain the working of a Pelton wheel turbine with a neat sketch.", 8, "f1"),
        ("numerical", "A Francis turbine works under a head of 60 m. Calculate the specific speed if the power output is 500 kW at 400 rpm.", 8, "f2"),
        ("compare_contrast", "Differentiate between impulse turbines and reaction turbines such as the Kaplan turbine.", 6, "f3"),
    ],
    "pumps": [
        ("conceptual_explanation", "Explain priming of a centrifugal pump and define the net positive suction head (NPSH).", 6, "f1"),
        ("numerical", "A centrifugal pump impeller has an outer diameter of 40 cm and runs at 1450 rpm. Calculate the manometric efficiency if the manometric head is 30 m.", 8, "f2"),
    ],
}

# Ground-truth mapping from bank key to the syllabus topic number (for tests).
TOPIC_NUMBERS = {
    "fluid_continuum": "1.1", "viscosity": "1.2", "surface_tension": "1.3", "compressibility": "1.4",
    "pressure_measurement": "2.1", "hydrostatic_forces": "2.2", "buoyancy": "2.3",
    "flow_types": "3.1", "continuity": "3.2", "potential_flow": "3.3",
    "bernoulli": "4.1", "flow_measurement": "4.2", "orifices_notches": "4.3", "momentum": "4.4",
    "laminar_flow": "5.1", "darcy": "5.2", "minor_losses": "5.3", "pipe_networks": "5.4",
    "dimensional_analysis": "6.1", "similitude": "6.2",
    "boundary_layer": "7.1", "separation_drag": "7.2",
}

# Planted temporal behaviour for each topic.
#   core      : appears in most papers
#   regular   : appears about half the time
#   rare      : seldom appears
#   alt_even / alt_odd : alternates between papers
#   every3    : roughly every third paper
#   cooldown  : unlikely right after appearing, likely after a one-paper gap
#   emerging  : rare early, frequent in recent papers
#   fading    : frequent early, absent recently
PROFILES = {
    "fluid_continuum": "rare", "viscosity": "core", "surface_tension": "rare", "compressibility": "rare",
    "pressure_measurement": "core", "hydrostatic_forces": "cooldown", "buoyancy": "alt_even",
    "flow_types": "regular", "continuity": "core", "potential_flow": "rare",
    "bernoulli": "core", "flow_measurement": "alt_odd", "orifices_notches": "fading", "momentum": "regular",
    "laminar_flow": "cooldown", "darcy": "core", "minor_losses": "regular", "pipe_networks": "every3",
    "dimensional_analysis": "every3", "similitude": "regular",
    "boundary_layer": "emerging", "separation_drag": "emerging",
}
