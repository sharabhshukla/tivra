# तिव्र (Tivra)

**तिव्र** (*Tivra*, derived from the Sanskrit word तीव्र, meaning "fast" or "swift") is a Python library designed for solving optimization problems at incredible speeds, using the **Primal-Dual Hybrid Gradient (PDHG)** algorithm. This solver integrates seamlessly with open-source optimization modeling frameworks like **Pyomo**, enabling users to leverage the power of rapid and efficient optimization in their Python workflows.

## About PDHG

The **Primal-Dual Hybrid Gradient (PDHG)** algorithm is a state-of-the-art optimization technique that efficiently solves saddle-point problems that often arise in convex optimization. Known for its simplicity and scalability, PDHG eliminates the need for expensive operations like matrix factorization, making it particularly suitable for solving large-scale problems quickly and with minimal resource overhead.

Why choose PDHG?

- **Fast Convergence**: PDHG provides high-speed convergence while maintaining numerical stability. It is one of the fastest optimization algorithms available for convex problems.
- **Efficiency at Large Scales**: The algorithm's computational simplicity and low memory requirements make it capable of handling large-scale models effectively.

## Leveraging Hardware Accelerators

One of the defining features of **Tivra** is its ability to utilize multiple hardware accelerators to significantly reduce solve times. Whether you are working on **CPUs**, **GPUs**, or **multi-core systems**, the solver takes full advantage of available computational resources, seamlessly distributing workloads to improve performance.

This feature allows **Tivra** to excel not only in optimizing single-thread tasks but also in parallelized computations, unlocking immense power for solving high-dimensional optimization problems quickly, across a variety of hardware environments.

---

## Features

- **Fast and Scalable**: Based on the PDHG algorithm, priority is given to performance at scale.
- **Seamless Integration with Pyomo**: Supports Python-based languages for defining optimization models fluidly.
- **Hardware Acceleration**: Includes support for CPUs, GPUs, and beyond.
- **Open Source**: Fully open and modifiable for a variety of use cases, adhering to community-driven development.

---

## Installation

Install the latest version of **Tivra** from PyPI:

```bash
pip install tivra
```

## Usage

### Integration with Pyomo

Here’s how you can solve a Pyomo model with **Tivra**:

1. Define your optimization model in Pyomo.
2. Use **Tivra** as the solver for the model.

Example:

```python
from pyomo.environ import *
from tivra import TivraSolver

# Define simple Pyomo model
model = ConcreteModel()
model.x = Var(domain=NonNegativeReals)
model.y = Var(domain=NonNegativeReals)
model.obj = Objective(expr=model.x + model.y, sense=minimize)
model.constraint = Constraint(expr=model.x + 2 * model.y >= 1)

# Use Tivra to solve the model
solver = TivraSolver()
results = solver.solve(model)

print(results)
```

### Standalone Use

You can also use **Tivra** independently of Pyomo by defining optimization problems directly in its native interface. Please refer to the documentation for advanced use cases.

## Hardware Acceleration

To run **Tivra** on hardware accelerators like GPUs or multi-core CPUs, configuration is automatic. However, you can fine-tune resource use (e.g., number of CPU threads, specific GPU device, etc.) through settings.

---

## Documentation

Comprehensive documentation for **Tivra** can be found at [Tivra Docs](https://tivra-docs.example.com). It includes:

- Installation Guide
- Tutorials
- API Reference
- Examples and Recipes

---

## Contribution

We warmly welcome contributions to **Tivra**. To contribute:

1. Fork the repository on GitHub.
2. Create a feature branch for your changes.
3. Create and submit a pull request explaining your addition.

For reporting bugs or requesting features, please open an issue on the GitHub tracker.

---

## License

**Tivra** is licensed under the **MIT License**. You are free to use, modify, and distribute it as per the terms of the license.

---

## Acknowledgements

We would like to thank all contributors and the Pyomo community for their collaboration and support in creating a fast and efficient solution for optimization problems.

---

Start solving faster today with **Tivra**—the swift and efficient solver for Python-based optimization. 🎉