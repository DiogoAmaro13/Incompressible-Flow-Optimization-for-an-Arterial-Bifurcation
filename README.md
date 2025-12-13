# Data Assimilation in Hemodynamics
A modular dolfinx-based implementation to solve the optimal control problem associated with blood flow within an arterial bifurcation. Assuming there exist parametrized velocity measures inside the artery, we want to find the control variable in order to retrieve the velocity and pressure fields in the whole domain. For more details, we refer to the problem [formulation](assets/formulation.pdf)

## Requirements

To run the code, you should create the environment that contains dolfinx 0.10, as it is the most stable version currently. To do so, run

```
conda env create -f env.yml
```

from the root directory and activate it using

```
conda activate env
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.