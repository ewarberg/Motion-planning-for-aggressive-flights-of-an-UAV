# Motion Planning for Aggressive Flights of an UAV

Erik Warberg (warberg@kth.se) and Alexander Medén (ameden@kth.se) 

Supervisor: Xiao Tan

This source code in Python is implementing a method for motion planning of quadrocopters. 
Together with the motion planner there are files for simulation and plotting in MATLAB.

The used method is described in the following paper: A. Medén and E. Warberg, "Motion Planning for Aggressive Flights of an Unmanned Aerial Vehicle", 2021.

The paper is now available online [through DiVa](https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1634288&dswid=9470).

## Results

Below is a simulated trajectory showing the motion planners capability in a complex environement:
<img src="https://github.com/ewarberg/Motion-planning-for-aggressive-flights-of-an-UAV/blob/main/complex_environment_result.gif" width="400" height="400">

This result is using a obstacle avoidance motion primitive generator capable of aggressive maneuvers. 
<img src="https://gits-15.sys.kth.se/warberg/KEX2021/blob/master/obstacle_avoidance_CBF.gif" width="400" height="400">

## Start Guide

Download the files in the latest branch and open `traj_plot.m` in MATLAB.
Be aware that this might take a long time depending on the defined obstacle environment in `motion_planning.py`.
It's adviced to lower K to be below 1000 for simpler obstacle environments.

The file `quadrocoptertrajectory.py` is used a s a sub-module in the algorithm and the original GitHub repository is found [here][quad_git].
The MATLAB files for simulation and visualization are from [this project][2020_git].

[quad_git]: https://github.com/markwmuller/RapidQuadrocopterTrajectories
[2020_git]: https://gits-15.sys.kth.se/palfelt/KEX
