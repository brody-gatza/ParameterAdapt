---
title: 'CompFlowLab: A Python code to develop and prototype new data-driven models for challenging compressible flow problems with shocks and chemical reactions'
tags:
  - Python
  - Fluid Flows
  - Reduced Order Modeling
  - Phyiscs
  - Navier Stockes
  - Compressible Flow
  - Reacting flows
  - Combustion
  - Numerical Method
authors:
  - name: Ali Mohaghegh
    orcid: 0009-0009-9321-5134
    affiliation: 1
  - name: Cheng Huang
    affiliation: 2
affiliations:
 - name: PhD Student, University of Kansas, United States
   index: 1
   ror: 00hx57361
 - name: Assistant Professor, University of Kansas, United States
   index: 2
date: 30 March 2026
bibliography: ref.bib
---

# Summary
`CompFlowLab` is an open-source Python environment that is capable of modeling different classes of compressible flow problems (including shocks, flames, and detonation waves) using a one-dimensional compressible Navier–Stokes solver with multi-species transport and chemical-reaction models. It is designed specifically for the data-driven modeling community as a lightweight, accessible prototyping platform to test, develop, and evaluatie new modeling methods on numerically and physically challenging compressible flow problems, especially those featuring shocks and chemical reactions. Specifically, `CompFlowLab` aims at (1) providing computationally efficient calculations on advection-dominanted problems that are well-recognized to be difficult for conventional data-driven modeling techniques [@cohen_optimal_2020,@bonnaillie-noel_efficient_2016], such as shocks, flames, and detonation waves, and more importantly (2) enabling rapid testing and prototyping of new data-driven models. The code serves three primary purposes: (1) generating high-fidelity full order model data for training of data-driven modeling, (2) providing a modular platform for implementing and testing novel data-driven algorithms (e.g., machine learning methods and reduced-order modeling techniques) on challenging physics, and (3) offering a standalone Computational Fluid Dynamics (CFD) solver with validated test cases that can also support numerical method development in the broader CFD community. By unifying these capabilities in a clean, extensible Python codebase, `CompFlowLab` lowers barriers to innovation at the intersection of model reduction and complex fluid dynamics.

# Statement of need
Researchers working on data-driven model development for fluid flows face a persistent challenge: evaluating new methods on problems with the relevant complexity and strong nonlinear phenomena, such as shocks, flames, and detonation waves, often requires connecting to large, legacy CFD codes that are commonly not open source. Even when the code is available, these simulations can be prohibitively expensive, requiring significant high performance computing resources and long runtimes. Conversely, building a custom high fidelity solver from scratch is time consuming and diverts focus from data-driven algorithm development. This fragmentation is particularly acute in the data-driven modeling community, where many promising ideas can remain underexplored on shock and reacting flows due to the lack of a suitable, shared testbed. `CompFlowLab` addresses this gap by providing a self contained Python environment where users can easily generate training data from validated full order simulations, implement new data-driven techniques with minimal effort, and benchmark against built-in challenging cases. The code is intentionally kept lightweight and modular, with the goal of enabling researchers to focus on algorithmic innovation rather than software development. At the same time, the included CFD solver and test suite; covering supersonic, hypersonic, and detonation regimes can also offer the broader CFD community as a simple platform to test and experiment new numerical methods, especially those for shock modeling. `CompFlowLab` thus fills a unique role as a community oriented tool that accelerates research at the intersection of data-driven modeling, CFD, and compressible flows.

# State of the field
The software landscape for developing new data-driven modeling techniques in fluid dynamics is broad, but existing tools typically fall into two categories that leave a critical gap between methodology prototyping and realistic applications.

On one end, many research codes and tutorials for data-driven techniques are centered on simple model equations (e.g., Burgers' equation, the heat equation, or the one-dimensional linear advection equation), that are not necessarily representing the governing dynamics in real applications. While these problems enable fast initial algorithm prototyping, they do not capture the key features of practical fluid dynamics of interest in realistic applications, including strong nonlinearities in highly compressible flows and multi-physics couplings in chemically reacting flows. As a result, methods successfully prototyped on these settings may not translate reliably to complex fluid flows in realistic settings.

On the other end, existing libraries and frameworks such as PyROM [@pyrom], RBniCS [@RozzaBallarinScandurraPichi2024], and packages for POD, DMD, and operator inference provide robust tools for constructing data-driven models. However, these tools are typically non-intrusive and require user to supply training data generated from external high fidelity solvers. Generating such training data can be resource-consuming since it requires configuring and validating a CFD solver, setting up problem-specific simulations (e.g., with shocks or flames), and performing computations. All of these required efforts in data generation present a barrier for researchers whose primary focus is on model/algorithm development rather than CFD implementation.

`CompFlowLab` is designed to bridge this gap. It solves the one-dimensional compressible Navier–Stokes equations with multi-species transport and chemical reaction models, which can accurately model challenging compressible flow problems with shocks and flames, and provide a more representative test platform for prototyping new data-driven techniques than those simplified model problems. Meanwhile, it provides a built-in, validated finite-volume CFD solver that enables users to generate training data within the same environment used for model development. Furthermore, because the full order model is infrastructured in a modular and accessible form, `CompFlowLab` also supports development of intrusive data-driven techniques (e.g., Galerkin projection) that require access to the underlying solver operators, and allows consistent evaluations and comparisons between intrusive and non-intrusive data-driven methods.

A build vs. contribute justification further distinguishes CompFlowLab from both standalone libraries/framework and production CFD codes. Incorporating a compressible reacting flow solver into an existing data-driven modeling framework would require substantial redevelopment within the framework's architecture. Conversely, extending production CFD codes (e.g., OpenFOAM or SU2) to support rapid data-driven model prototyping typically requires navigating large, complex codebase. `CompFlowLab` instead provides a purpose-built, lightweight Python-based environment where both the full order solver and the data-driven model components are clearly separated, modular, essy to access and modify. This allows researchers to focus on development of new data-driven modeling algorithmic innovation.

By combining challenging physics, built in data generation, and support for both intrusive and non-intrusive data-driven model development in a single lightweight environment, `CompFlowLab` fills a critical gap in the ecosystem of data-driven modeling community. It provides the community with a shared, reproducible, and economical testbed for developing and evaluating new methods on problems that closely mimic and reflects the relevant modeling challengines in realistic applications, while also serving as a valuable tool for CFD researchers seeking to test numerical methods on challenging compressible flow problems with shocks and chemical reactions.

# Software design
CompFlowLab is designed around three core principles: (1) providing a lightweight, easy-to-access platform for rapid prototyping of data-driven modeling techniques, (2) maintaining a modular and clear separation between the full-order model (FOM) and data-driven model components, and (3) leveraging established community tools and standards (e.g., NumPy for vectorized computations, Cantera for reacting flow thermodynamics).

`CompFlowLab` adopts a module-based structure in which separate solver modules are defined for each modeling approach, which include: (1) a dedicated FOM solver handles high-fidelity simulations of compressible fluid problems; and (2) a built-in module directly connecting with FOM for all necessary components required to construct different types of data-driven models. This structure provides a consistent interface that allows users to switch between FOM and data-driven model representations of the same physical problem with minimal alternation to the workflow. In addition, this design facilitates direct comparison of model performance and accuracy, while also providing a clear template for extending the code for prototyping new data-driven methods.

The codebase is organized into modular Python files (i.e., .py) that group functionality by physical and numerical methods. For example, equation-of-state, chemical reaction, flux schemes, time integration, and data-driven models are implemented in separate modules and can be imported as needed for different problems. This modular structure improves code readability and allows contributors to modify or extend individual components without affecting the other modules and the overall workflow.

Moreover, performance-critical operations, including flux computations, time integrations, and large matrix operations, are implemented using NumPy or SciPy vectorization to leverage efficient array operations while maintaining readability in Python. This approach strikes a balance between computational performance and code accessibility, ensuring that simulations run efficiently on standard hardware without introducing the complexity of compiled languages. `CompFlowLab` directly uses Cantera (a well-established community tool) to provide thermodynamic and chemical kinetic models for reacting flow simulations. This enables accurate representation of combustion process while aligning with widely used and acknoledged tools in the combustion research community. 

The overall modular architecture is designed to be extensibile. Researchers can adapt existing solver templates and modules to prototype new data-driven techniques by modifying the relevant components, such as exploring new projection strategies, hyper-reduction techniques, and novel manifold learning methods. This reduces development overhead and allows researchers to focus on methodological innovation rather than software infrastructure.

# Research impact statement
`CompFlowLab` has supported research in prototyping new data-driven modeling techniques for challenging compressible flow problems since its initial release. It serves as both a testbed/platform for developing and evaluating new data-driven modeling algorithms and a reliable tool to generate training data for compressible flow applications. This code has been used in the development and validating of novel reduced-order modeling techniques, including feature-guided sampling methods [@mohaghegh_feature-guided_2026] and self-adaptive ROM algorithms [@mohaghegh_self_2026], which have resulted in peer-reviewed publications. In addition, `CompFlowLab` has been adopted as a data generation tool for one dimensional rotating detonation engine (RDE) simulations and related reacting flow problems. In these context, it provides a lightweight alternative to large-scale CFD codes for producing high fidelity training datasets suitable for model development and evaluation.

By enabling computational studies on compressible flows with shocks, detonation, and combustion, `CompFlowLab` supports research in regimes that are less commonly represented in standard data-driven modeling benchmarks,. which makes it a useful platform for evaluating data-driven techniques under physically relevant and challenging conditions.

The codebase is actively maintained and continues to be developed, with documentation hosted at compflowlab.mintlify.app and source code released as open-source to support transparency and reuse. Ongoing development aims to expand functionality and encourage community contributions, with the goal of supporting reproducible research in data-driven modeling.

# AI usage disclosure
Generative AI tools were used in the debugging stages of this software and grammer check
of this manuscript.

# Acknowledgements
The authors acknowledge the supports from the Air Force Office of Scientific Research (AFOSR) through the Center of Excellence Grant FA9550-17-1-0195 (Technical Monitors: Fariba Fahroo, Justin Koo, and Ramakanth Munipalli) and the AFOSR under the grant FA9550- 23-1-0211 (Program managers: Drs. Chiping Li and Fariba Fahroo).

# References
