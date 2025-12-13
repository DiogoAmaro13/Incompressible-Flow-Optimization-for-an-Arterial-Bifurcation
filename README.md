# Data Assimilation in Hemodynamics
A modular dolfinx-based implementation to solve the optimal control problem associated with blood flow within an arterial bifurcation. Assuming there exists a parametrized velocity measurement inside the artery, we want to find the control variable in order to retrieve the velocity and pressure fields in the whole domain. For more details, we refer to the problem [formulation](notes/formulation.pdf).

## Project Structure
(under modifications)


## Requirements

To run the code, you should create the environment that contains dolfinx 0.10, as it is the most stable version currently. To do so, run

```
conda env create -f env.yml
```

from the root directory and activate it using

```
conda activate env
```

## Usage
From root, simply run
```
python main.py
```
## Custom Configuration
To change problem settings, such as artery dimensions, bifurcation orientation, mesh tags or fluid mechanics parameters, refer to [parameters file](parameters.py) and change it accordingly.

## Output Files
(under construction)
## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.