# Multi-aircraft Guidance and Separation Assurance

A Python implementation of the algorithm proposed in the paper "Scalable Multi-Agent Computational Guidance with Separation Assurance for Autonomous Urban Air Mobility Operations" by [Xuxi Yang](https://xuxiyang1993.github.io/) and [Peng Wei](https://web.seas.gwu.edu/pwei/).

Paper [link](https://arc.aiaa.org/doi/pdf/10.2514/1.G005000?casa_token=8hne9A6_jLsAAAAA:or3qJQKhTnqIcKcKaXgws9thg__BOG31IXL0OUDdCdx7CTdjN5PVmcwzx18-HJemO0qHxtoudvE)

Video [demo](https://www.youtube.com/watch?v=2cbRUig4G_I&t=)

## Install

This repository is only testesd under Python 3.6. To install the dependencies, run

```
pip install -r requirements.txt
```


## Optional arguments

`--save_path` the path where you want to save the output

`--seed` seed of the experiment

`--render` if render the env while running exp

`--debug` set to True if you want to debug the algorithm (the code will stop running and render the current state when there is conflict/LOS or NMAC, check this [line](https://github.com/xuxiyang1993/Multi_MCTS_Guidance_Separation_Assurance/blob/master/Simulators/MultiAircraftVertiHexSecGatePlusEnv.py#L231) for detail)

## Running the algorithm

Three case studies can be run in this repository.

For case study 1 in the paper, run

`python Agent_vertiHexSecGatePlus.py`

For case study 2, run

`python Agent_vertiHexSecGatePlusTwoStage.py`

For case study 3, run

`python Agent_vertiport.py`


## MCTS algorithm
The MCTS algorithm code is under the directory `MCTS/`

`common.py` defines the general MCTS node class and state class

`nodes*.py` defines the MCTS node class and state class specifically for Multi Agent Aircraft Guidance problem, e.g., given current aircraft state and current action, how to decide the next aircraft state

`search_multi.py` describes the search process of MCTS algorithm

## Simulator
The simulator code is under the directory of `simulators/`. The following described the main function in simulators.

`config*.py` defines the configurable parameters of the simulator. For example, airspace width/length, number of aircraft, scale (1 pixel = how many meters), conflict/NMAC distance, cruise/max speed of aircraft, heading angle change rate of aircraft, number simulations and search depth of MCTS algorithm, vertiport location, ...

* `__init__()` initialize the simulator by generating vertiports, sectors, loading configuration parameters, and generating aircraft.

* `reset()` will reset the number of conflicts/NMACs to 0 and reset the aircraft dictionary. Note here all the aircraft objects are stored in the `AircraftDict` class, where you can add/remove aircraft from it, and get aircraft object by id.

* `_get_ob()` will return the current state, which is n by 8 matrix, where n is the number of aircraft. Each aircraft has (x, y, vx, vy, speed, heading, gx, gy) state information.

* `_get_normalized_ob()` will normalize the state to be in range [0, 1], which will be useful if you want to feed state into a neural network.

* `step()` will return next state, reward, terminal, info given current state and current action. Each aircraft will fly according to the given action. We have a clock at each vertiport to decide whether to generate new flight request/aircraft.

* `_terminal_reward()` will return the reward function for current state. This function will check if there is any conflict/NMAC between any two aircraft and update conflict/NMAC number. It will also remove aircraft that reaches goal position and aircraft pair that has NMAC.

* `render()` will visualize all of the current aircraft and vertiport.

## Citing this work
If you find this codebase useful for your research work, we encourage you to cite our paper using the following BibTex citation:

```
@article{yang2020scalable,
  title={Scalable Multi-Agent Computational Guidance with Separation Assurance for Autonomous Urban Air Mobility},
  author={Yang, Xuxi and Wei, Peng},
  journal={Journal of Guidance, Control, and Dynamics},
  volume = {43},
  number = {8},
  pages = {1473-1486},
  year={2020},
  publisher={American Institute of Aeronautics and Astronautics},
  doi = {10.2514/1.G005000},
}
```

### Note

If you have any comments, questions, or suggestions, feel free to let me know!

Email: xuxiyang1993@gmail.com
