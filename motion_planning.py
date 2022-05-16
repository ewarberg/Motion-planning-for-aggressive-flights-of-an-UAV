"""Implementation of a motion planning algorithm for an UAV"""

import numpy as np
import random
import quadrocoptertrajectory as quadtraj
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt     # Comment out if you are going to simulate in Matlab
from scipy.spatial import ConvexHull
import mpl_toolkits.mplot3d as a3
from cvxopt import matrix, solvers
from scipy.integrate import odeint
import polytope as pc
from operator import itemgetter
from collections import deque


"""def get_cbf_data(cbf, goal, uav, obstacles):
    dt = cbf.dt
    node = goal
    goal_pos = node.data[0]
    pos = []
    vel = []
    acc = []
    rot = []
    time = []
    thr = []
    rates = []
    t = 0
    num_samples = len(node.traj)
    sample_points = np.linspace(0, num_samples-1, num_samples)
    for j in sample_points[0::5]:
        j = round(j)
        p = node.traj[j][0][0]
        v = node.traj[j][0][1]
        a = node.traj[j][0][2]
        r = node.traj[j][1]
        w = node.traj[j][2]
        f = node.traj[j][3]
        pos.append(p)
        vel.append(v)
        acc.append(a)
        rot.append(r)
        time.append(t + 5 * dt)
        rates.append(w)
        thr.append(f)
        t += 5 * dt
    node_state = [node.data[0][0], node.data[0][1], node.data[0][2], node.data[1][0], node.data[1][1], node.data[1][2],
                           node.data[2][0], node.data[2][1], node.data[2][2], node.traj[-1][3], node.traj[-1][2][0],
                           node.traj[-1][2][1], node.traj[-1][2][2]]
    obs_extreme = []
    obs_pos = []
    for obstacle in obstacles:
        obs_extreme.append(pc.extreme(obstacle.poly))
        obs_pos.append(obstacle.pos)
    obs_extreme = np.array(obs_extreme)
    obs_pos = np.array(obs_pos)
    position = np.array(pos)
    velocity = np.array(vel)
    acceleration = np.array(acc)
    R_matrices = np.array(rot)
    node_points = np.array(node_state)
    timevec = np.array(time)
    thrust = np.array(thr)
    body_rates = np.array(rates)
    return [2, position, velocity, acceleration, R_matrices, node_points, timevec, obs_extreme, goal_pos,
            obs_pos, uav.width, uav.height, thrust, body_rates]"""


def get_graph_size(node, i=0):
    """
    Get the number of vertices in the high-level trajectory

    :param node:    A vertex
    :param i:       Number of vertices in the high-level trajectory
    :return:        Return number of vertices
    """
    if node.parent is not None:
        i = get_graph_size(node.parent, i)
        i += 1
        return i
    else:
        return 1


def get_rrt_data(rrt, uav, obstacles):
    """
    Gets the data required for the simulations in Matlab

    :param rrt:         RRT object
    :param uav:         UAV object
    :param obstacles:   List of obstacle objects
    :return:            Motion planning data
    """
    dt = rrt.dt
    node = rrt.G.goal
    goal_pos = node.data[0]
    trajsize = get_graph_size(node)
    sampling = rrt.cbf.dt  # Corresponds to the rate of the obstacle avoidance
    pos_que = deque([])
    vel_que = deque([])
    acc_que = deque([])
    rot_que = deque([])
    time_que = deque([])
    node_data = deque([])
    rates_que = deque([])
    thrust_que = deque([])
    t = 0
    for i in range(1, trajsize):
        pos = deque([])
        vel = deque([])
        acc = deque([])
        rot = deque([])
        time = deque([])
        thr = deque([])
        rates = deque([])
        if node.isCBF:
            num_samples = len(node.traj)
            sample_points = np.linspace(0, num_samples-5, num_samples-4)
            for j in sample_points[0::5]:
                j = round(j)
                p = node.traj[j][0][0]
                v = node.traj[j][0][1]
                a = node.traj[j][0][2]
                r = node.traj[j][1]
                w = node.traj[j][2]
                f = node.traj[j][3]
                pos.append(p)
                vel.append(v)
                acc.append(a)
                rot.append(r)
                time.append(t+5*sampling)
                rates.append(w)
                thr.append(f)
                t += 5*sampling
            t += node.duration
            node_state = [node.data[0][0], node.data[0][1], node.data[0][2], node.data[1][0], node.data[1][1],
                          node.data[1][2],
                          node.data[2][0], node.data[2][1], node.data[2][2], node.traj[-1][3], node.traj[-1][2][0],
                          node.traj[-1][2][1], node.traj[-1][2][2]]
        else:
            time_points = np.linspace(0, node.duration-dt, round(node.duration/dt-1))
            for t_i in time_points:
                p = node.traj.get_position(t_i)
                v = node.traj.get_velocity(t_i)
                a = node.traj.get_acceleration(t_i)
                r = rrt.rotation.normalvec_to_R(node.traj.get_normal_vector(t_i))
                w = node.traj.get_body_rates(t_i)
                f = node.traj.get_thrust(t_i)
                pos.append(p)
                vel.append(v)
                acc.append(a)
                rot.append(r)
                time.append(t+t_i)
                rates.append(w)
                thr.append(f)
            t += node.duration
            node_state = np.array([node.data[0][0], node.data[0][1], node.data[0][2],
                                   node.data[1][0], node.data[1][1], node.data[1][2],
                                   node.data[2][0], node.data[2][1], node.data[2][2],
                                   node.traj.get_thrust(node.duration),
                                   node.traj.get_body_rates(node.duration)[0],
                                   node.traj.get_body_rates(node.duration)[1],
                                   node.traj.get_body_rates(node.duration)[2]])
        if node is rrt.G.goal:
            pos.append(node.traj.get_position(node.duration))
            vel.append(node.traj.get_velocity(node.duration))
            acc.append(node.traj.get_acceleration(node.duration))
            rot.append(rrt.rotation.normalvec_to_R(node.traj.get_normal_vector(node.duration)))
            time.append(node.duration)
            rates.append(node.traj.get_body_rates(node.duration))
            thr.append(node.traj.get_thrust(node.duration))
        pos += pos_que
        pos_que = pos
        vel += vel_que
        vel_que = vel
        acc += acc_que
        acc_que = acc
        rot += rot_que
        rot_que = rot
        time_que += time
        rates += rates_que
        rates_que = rates
        thr += thrust_que
        thrust_que = thr
        node_data.appendleft(node_state)
        node = node.parent

    obs_extreme = []
    obs_pos = []
    for obstacle in obstacles:
        obs_extreme.append(pc.extreme(obstacle.poly))
        obs_pos.append(obstacle.pos)
    obs_extreme = np.array(obs_extreme)
    obs_pos = np.array(obs_pos)
    position = np.array(list(pos_que))
    velocity = np.array(list(vel_que))
    acceleration = np.array(list(acc_que))
    R_matrices = np.array(list(rot_que))
    node_points = np.array(list(node_data))
    timevec = np.array(list(time_que))
    thrust = np.array(list(thrust_que))
    body_rates = np.array(list(rates_que))

    return [trajsize, position, velocity, acceleration, R_matrices, node_points, timevec, obs_extreme, goal_pos,
            obs_pos, uav.width, uav.height, thrust, body_rates]


class Node:
    def __init__(self, data, rot, traj=None, duration=0, isCBF=False, parent=None):
        """
        :param data:        data = [position, velocity, acceleration]
        :param rot:         Rotational matrix
        :param traj:        Trajectory object
        :param duration:    Trajectory duration
        :param isCBF:       True if the trajectory has been generated by the obstacle avoidance, false otherwise
        :param cbf_cost:    Cost of the trajectory generated by the obstacle avoidance
        :param parent:      Parent vertex
        """
        self.traj = traj
        self.data = data
        self.rot = rot
        self.parent = parent
        self.duration = duration
        self.isCBF = isCBF


class Tree:
    def __init__(self, root, goal):
        """
        :param root:    Root vertex containing the initial state of the UAV.
        :param goal:    Goal vertex containing the goal state of the UAV.
        """
        self.root = root
        self.goal = goal
        self.V = [root, goal]
        self.tree = None

    def add_vertex(self, q_new):
        """
        :param q_new:   New vertex to add to the tree.
        """
        self.V.append(q_new)

    def add_edge(self, q_parent, q_child):
        """
        :param q_parent:    Parent vertex
        :param q_child:     Child vertex
        """
        q_child.parent = q_parent

    def find_near(self, new_point, d_q):
        """
        :param new_point:   New position returned by new_conf
        :param d_q:         Incremental distance
        :return:            List of the vertices positioned within a radius d_q around the new point
        """
        near = []
        for node in self.V:
            dist = (node.data[0][0] - new_point[0]) ** 2 + (node.data[0][1] - new_point[1]) ** 2 + \
                         (node.data[0][2] - new_point[2]) ** 2
            if dist <= d_q ** 2 and dist != 0.0:
                near.append(node)
        return near

    def find_nearest(self, new_point):
        """
        :param new_point:   New position returned by rand_conf
        :return:            The vertex nearest to the new point
        """
        nearest = self.root
        shortest_dist = (nearest.data[0][0] - new_point[0]) ** 2 + (nearest.data[0][1] - new_point[1]) ** 2 + (
                    nearest.data[0][2] - new_point[2]) ** 2
        for node in self.V:
            dist = (node.data[0][0] - new_point[0]) ** 2 + (node.data[0][1] - new_point[1]) ** 2 + (
                        node.data[0][2] - new_point[2]) ** 2
            if dist < shortest_dist and dist != 0.0:
                shortest_dist = dist
                nearest = node
        return nearest


class TrajGeneration:

    def generate_traj(self, p0, v0, a0, pf, vf, af, T):
        """
        :param p0:  Initial position of the trajectory
        :param v0:  Initial velocity of the trajectory
        :param a0:  Initial acceleration of the trajectory
        :param pf:  Goal position of the trajectory
        :param vf:  Goal velocity of the trajectory
        :param af:  Goal acceleration of the trajectory
        :param T:   Duration of the trajectory
        :return:    Trajectory object
        """
        traj = quadtraj.RapidTrajectory(p0, v0, a0, [0, 0, -9.81])
        traj.set_goal_position(pf)
        traj.set_goal_velocity(vf)
        traj.set_goal_acceleration(af)
        traj.generate(T)
        return traj

    def get_traj_duration(self, p0, pf, d_q):
        """
        :param p0:          Initial position of the trajectory
        :param pf:          Goal position of the trajectory
        :param d_q:         Incremental distance
        :return:            Duration of the trajectory
        """
        direction = p0 - pf
        dist = abs(np.linalg.norm(direction))
        t_unit = d_q/2
        dist_ratio = dist/d_q
        T = dist_ratio*t_unit
        return T

    def get_goal_acceleration(self, rot):
        """
        :param rot:     Attitude of the UAV
        :return:        Acceleration array in the desired direction
        """
        e3 = np.array([0, 0, 1])
        k_s = 0.01
        acceleration = np.dot(rot, e3)*k_s
        return acceleration

    def get_traj(self, child, parent, d_q):
        """
        :param child:       Child vertex
        :param parent:      Parent vertex
        :param d_q:         Incremental distance
        :param G:           RRT graph
        :return:            List of trajectory data, [trajectory object, duration, goal velocity, goal acceleration]
        """
        p0 = parent.data[0]
        v0 = parent.data[1]
        a0 = parent.data[2]
        if type(child) is not Node:
            pf = child[0]
            af = self.get_goal_acceleration(child[1])
            vf = [None, None, None]
        else:
            pf = child.data[0]
            af = child.data[2]
            vf = child.data[1]
        T = self.get_traj_duration(p0, pf, d_q)
        traj = self.generate_traj(p0, v0, a0, pf, vf, af, T)
        vf = traj.get_velocity(T)
        return [traj, T, vf, af]


class Rotation:

    def create_rot(self, pitch, roll):
        """
        Creates rotational matrix from Euler angles.

        :param pitch:   Pitch angle
        :param roll:    Roll angle
        :return:        Rotational matrix
        """
        R_matrix = np.array([[np.cos(pitch), np.sin(pitch) * np.sin(roll), np.sin(pitch) * np.cos(roll)],
                             [0, np.cos(roll), - np.sin(roll)],
                             [-np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(roll)]])
        return R_matrix

    def normalvec_to_R(self, normalvec):
        """This function was aqcuired from the file H_level_k_opt.py in the github repository
                https://gits-15.sys.kth.se/palfelt/KEX by Oscar Palfelt and Fredrik Skjernov"""

        """
        Finds the rotational matrix which rotates n0 onto normalvec
        
        :param normalvec:   Normal vector of the UAV surface.
        :return:            Rotational matrix corresponding to the normal vector.
        """

        n0 = np.array([0, 0, 1])
        ds = np.arccos(np.dot(normalvec, n0))
        ax = np.array([[0, -n0[2], n0[1]], [n0[2], 0, -n0[0]], [-n0[1], n0[0], 0]])
        axe3 = np.dot(ax, normalvec)
        if np.linalg.norm(axe3) == 0:
            return np.eye(3)

        axe3 = axe3 / np.linalg.norm(axe3)
        axe3x = np.array([[0, -axe3[2], axe3[1]], [axe3[2], 0, -axe3[0]], [-axe3[1], axe3[0], 0]])
        R = np.eye(3) + np.sin(ds) * axe3x + (1 - np.cos(ds)) * np.dot(axe3x, axe3x)

        return R


class RRT:

    def __init__(self, root_pos, goal_pos, obstacles, uav, K, interval, d_q, dt, cbf=None, use_cbf=False):
        """
        :param root_pos:    Initial position of the UAV.
        :param goal_pos:    Goal position of the UAV.
        :param obstacles:   List of obstacle objects.
        :param uav:         UAV object
        :param K:           Number of times a random node will be generated.
        :param interval:    Interval of the area in which random nodes will be generated.
        :param d_q:         Incremental distance
        :param dt:          Time step for collision checking.
        :param cbf:         CBF object
        :param use_cbf:     True if CBF will be used, False otherwise.
        """
        self.rotation = Rotation()
        self.G = Tree(self.root_vertex(root_pos), self.goal_vertex(goal_pos))
        self.d_q = d_q
        self.uav = uav
        self.obstacles = obstacles
        self.dt = dt
        self.path_found = False
        self.K = K
        self.interval = interval
        self.traj_gen = TrajGeneration()
        self.check_collision = CollisionCheck()
        self.use_cbf = use_cbf
        self.cbf = cbf

    def rand_conf(self):
        """
        :return:    Random position and random attitude.
        """
        x_rand = random.uniform(self.interval[0][0]*10, self.interval[0][1]*10)/10
        y_rand = random.uniform(self.interval[1][0]*10, self.interval[1][1]*10)/10
        z_rand = random.uniform(self.interval[2][0]*10, self.interval[2][1]*10)/10
        rand_pos = np.array([x_rand, y_rand, z_rand])
        roll_rand = random.uniform(-130, 130)/100    # theta
        pitch_rand = random.uniform(-130, 130)/100   # psi
        # Rotational matrix based on intrinsic rotation and Euler/Tait-Bryan angles.
        rand_rot = self.rotation.create_rot(pitch_rand, roll_rand)
        return [rand_pos, rand_rot]

    def root_vertex(self, pos):
        """
        :param pos:    Position of the root vertex.
        :return:       Root vertex
        """
        vel = np.array([0, 0, 0])
        acc = np.array([0, 0, 0])
        data = [pos, vel, acc]
        roll = 0
        pitch = 0
        rot = self.rotation.create_rot(pitch, roll)
        vertex = Node(data, rot)
        return vertex

    def goal_vertex(self, pos):
        """
        :param pos:    Position of the goal vertex.
        :return:       Goal vertex
        """
        vel = np.array([0, 0, 0])
        acc = np.array([0, 0, 0])
        data = [pos, vel, acc]
        roll = 0
        pitch = 0
        rot = self.rotation.create_rot(pitch, roll)
        vertex = Node(data, rot)
        return vertex

    def new_conf(self, nearest, q_rand):
        """
        :param nearest:    The nearest vertex to the randomly generated position.
        :param q_rand:     List containing a random position and a random attitude.
        :return:           List contining the position of the new vertex, which is within
                           the incremental distance from the nearest vertex in the direction
                           from the nearest vertex to the random position, and the attitude of the new vertex.
        """
        direction = q_rand[0] - nearest.data[0]
        dist = abs(np.linalg.norm(direction))
        if round(dist, 2) > self.d_q:
            unit_vec = direction/dist
            d_q_vec = self.d_q*unit_vec
            q_rand[0] = nearest.data[0] + d_q_vec
        if self.check_collision.point_collision(q_rand[0], self.obstacles):
            return None
        else:
            self.uav.update(q_rand[0], q_rand[1])
            if self.check_collision.uav_collision(self.uav, self.obstacles):
                return None
        q_new = [q_rand[0], q_rand[1]]
        return q_new

    def steer(self, q_new, parent):
        """
        :param q_new:       List contating a position and an attitude.
        :param parent:      Parent vertex
        :return:            If there is no collision or the obstacle avoidance finds a feasible trajectory,
                            the function returns a new vertex that is connected to the parent vertex by a
                            motion primitive, otherwise the function returns None.
        """
        traj_data = self.traj_gen.get_traj(q_new, parent, self.d_q)
        T = traj_data[1]
        traj = traj_data[0]
        vf = traj_data[2]
        af = traj_data[3]
        numSegments = round(T/self.dt)
        colliding, obs_index = self.check_collision.traj_collision(traj, T, numSegments, self.obstacles, self.uav)
        if not colliding:
            data = [q_new[0], vf, af]
            new_node = Node(data, q_new[1], traj, T)
            new_node.isCBF = False
            self.G.add_vertex(new_node)
            self.G.add_edge(parent, new_node)
            return new_node
        elif self.use_cbf and obs_index < 10 and not self.path_found:
            cbf_nodes = self.doCBF(q_new, parent)
            if cbf_nodes is not None:
                parent_node = parent
                for node in cbf_nodes:
                    self.G.add_vertex(node)
                    self.G.add_edge(parent_node, node)
                    parent_node = node
                return cbf_nodes[-1]
        return None

    def doCBF(self, q_new, parent):
        """
        :param q_new:       List containing a position and an attitude
        :param parent:      The parent vertex of q_new
        :return:            The vertices that is generated by dividing the trajectory generated by
                            the obstacle avoidance into shorter segments
        """
        isCBF, samples, duration = self.cbf.run(parent, q_new)
        if isCBF:
            """Divides the samples generated by the obstacle avoidance into 
            several motion primitives with associated vertices"""
            if len(samples) > 10:
                length = round(len(samples)/30)+1
                node_indices = np.linspace(0, len(samples), length)
                index_pre = 0
                nodes = []
                if length > 2:
                    for index in node_indices[1:-1]:
                        index = round(index)
                        samples_seg = samples[index_pre:index]
                        T_i = (len(samples_seg)-1)*self.cbf.dt
                        node = Node(samples[index][0], samples[index][1], samples_seg, T_i, True)
                        nodes.append(node)
                        index_pre = index
                last_seg = samples[index_pre:]
                T = (len(last_seg)-1)*self.cbf.dt
                last_node = Node(samples[-1][0], samples[-1][1], last_seg, T, True)
                nodes.append(last_node)
                return nodes
        return None

    def get_path_cost(self, q_start, q_goal):
        """
        :param q_start:     Starting vertex
        :param q_goal:      Goal vertex
        :return:            Cost in jerk of the trajectory connecting the starting vertex and the goal vertex
        """
        q = q_goal
        cost = 0
        if q_start != q_goal:
            while q is not q_start:
                if q.isCBF:
                    cost = cost + self.cbf.get_cbf_cost(q)
                else:
                    cost = cost + q.traj.get_cost()
                q = q.parent
        return cost

    def connect2best_parent(self, q_near, q_new):
        """
        :param q_near:      List of vertices positioned within the incremental distance from q_new
        :param q_new:       List containing position and attitude
        :return:            List containing a new vertex connected to the best parent by a motion primitive
                            and a cost from the best parent vertex to the root vertex
        """
        parent_list = []
        for node in q_near:
            if node is not self.G.goal:
                traj_data = self.traj_gen.get_traj(q_new, node, self.d_q)
                node_cost = self.get_path_cost(self.G.root, node)
                cost2root = node_cost + traj_data[0].get_cost()
                parent_list.append([node, traj_data, cost2root])

        sorted_list = sorted(parent_list, key=itemgetter(2))
        for parent in sorted_list:
            best_parent = parent[0]
            best_traj = parent[1]
            min_cost = parent[2]
            numSamples = round(best_traj[1]/self.dt)

            colliding, obs_index = self.check_collision.traj_collision(best_traj[0], best_traj[1], numSamples, self.obstacles, self.uav)
            if not colliding:
                q_near.remove(best_parent)
                data = [q_new[0], best_traj[2], best_traj[3]]
                new_node = Node(data, q_new[1], best_traj[0], best_traj[1])
                self.G.add_edge(best_parent, new_node)
                self.G.add_vertex(new_node)
                return [new_node, min_cost]
            elif self.use_cbf and obs_index < 10 and not self.path_found:
                cbf_nodes = self.doCBF(q_new, best_parent)
                if cbf_nodes is not None:
                    q_near.remove(best_parent)
                    parent_node = best_parent
                    for node in cbf_nodes:
                        self.G.add_vertex(node)
                        self.G.add_edge(parent_node, node)
                        parent_node = node
                    return [cbf_nodes[-1], self.get_path_cost(self.G.root, cbf_nodes[-1])]
        return None

    def rewire(self, new_node, cost2root, q_near):
        """
        Rewires the trajectory between the vertices to minimize the total cost

        :param new_node:        New vertex
        :param cost2root:       Cost from the parent of the new vertices to the root vertex
        :param q_near:          Vertices near to the new vertex
        """
        for node in q_near:
            traj_data = self.traj_gen.get_traj(node, new_node, self.d_q)
            traj = traj_data[0]
            T = traj_data[1]
            vf = traj_data[2]
            if node is self.G.goal and not self.path_found:
                node_cost = 0
            else:
                node_cost = self.get_path_cost(self.G.root, node)
            rewire_cost = cost2root + traj.get_cost()
            if rewire_cost < node_cost or (node is self.G.goal and not self.path_found):
                numSamples = round(T/self.dt)
                colliding, obs_index = self.check_collision.traj_collision(traj, T, numSamples, self.obstacles, self.uav)
                if not colliding:
                    if node is self.G.goal:
                        self.path_found = True
                    node.traj = traj
                    node.isCBF = False
                    node.duration = T
                    node.data[1] = vf
                    self.G.add_edge(new_node, node)

    def nearest_vertex(self, q_rand):
        """
        :param q_rand:  List contatining a random position and a random attitude.
        :return:        The vertex that is nearest to the randomly generated position.
        """
        if len(self.G.V) == 2:
            nearest = self.G.root
        else:
            nearest = self.G.find_nearest(q_rand[0])
        return nearest

    def near_vertices(self, q_new):
        """
        :param q_new:   List containing a position generated by new_conf and an attitude.
        :return:        List of vertices positioned within the incremental distance from q_new.
        """
        if len(self.G.V) == 2:
            near = [self.G.root]
        else:
            near = self.G.find_near(q_new[0], self.d_q)
        return near

    def connect2goal(self, new_node):
        """
        Connects the new vertex to the goal vertex if there is no collision in the trajectory connecting the two vertices.

        :param new_node:    New vertex
        """
        if new_node.isCBF:
            q = new_node
            while q.isCBF:
                T = self.traj_gen.get_traj_duration(q.data[0], self.G.goal.data[0], self.d_q)
                traj = self.traj_gen.generate_traj(q.data[0], q.data[1], q.data[2],
                                                   self.G.goal.data[0],
                                                   self.G.goal.data[1], self.G.goal.data[2], T)

                numSamples = round(T / self.dt)
                colliding, obs_index = self.check_collision.traj_collision(traj, T, numSamples, self.obstacles,
                                                                           self.uav)
                if not colliding:
                    self.G.goal.traj = traj
                    self.G.goal.duration = T
                    self.G.add_edge(q, self.G.goal)
                    self.path_found = True
                    self.G.goal.isCBF = False
                    self.use_cbf = False
                    break
                q = q.parent
        else:
            T = self.traj_gen.get_traj_duration(new_node.data[0], self.G.goal.data[0], self.d_q)
            traj = self.traj_gen.generate_traj(new_node.data[0], new_node.data[1], new_node.data[2], self.G.goal.data[0],
                                               self.G.goal.data[1], self.G.goal.data[2], T)
            numSamples = round(T/self.dt)
            colliding, obs_index = self.check_collision.traj_collision(traj, T, numSamples, self.obstacles, self.uav)
            if not colliding:
                self.G.goal.traj = traj
                self.G.goal.duration = T
                self.G.add_edge(new_node, self.G.goal)
                self.path_found = True
                self.G.goal.isCBF = False
                self.use_cbf = False

    def run(self):
        """The main function of RRT* that iterates through the chosen number of generated vertices"""
        rewire_count = 0
        for k in range(1, self.K):
            print(k)
            q_rand = self.rand_conf()
            q_nearest = self.nearest_vertex(q_rand)
            q_new = self.new_conf(q_nearest, q_rand)
            if q_new is not None:
                q_near = self.near_vertices(q_new)
                if not self.path_found:
                    newIsClosest = False
                    if len(q_near) > 1:
                        newIsClosest = True
                        new2goal = np.linalg.norm(q_new[0]-self.G.goal.data[0])
                        for q in q_near:
                            if new2goal > np.linalg.norm(q.data[0]-self.G.goal.data[0]):
                                newIsClosest = False
                    isRoot = len(q_near) == 1 and self.G.goal not in q_near
                    if newIsClosest or isRoot:
                        new_node = self.steer(q_new, q_nearest)
                        if new_node is not None:
                            dist2goal = abs(np.linalg.norm(new_node.data[0] - self.G.goal.data[0]))
                            close2goal = round(dist2goal, 2) <= self.d_q and round(dist2goal, 2) != 0.0
                            if close2goal or new_node.isCBF:
                                self.connect2goal(new_node)
                        continue
                best_parent = self.connect2best_parent(q_near, q_new)
                if best_parent is not None:
                    new_node = best_parent[0]
                    cost2root = best_parent[1]
                    if len(q_near) > 0:
                        rewire_count += 1
                        self.rewire(new_node, cost2root, q_near)
                        if self.path_found:
                            print("Total trajectory cost at rewire count",
                                  rewire_count, ":", self.get_path_cost(self.G.root, self.G.goal))
        if not self.path_found:
            print("Could not find a feasible trajectory")


class CBF:

    def __init__(self, obstacles, uav, a_1, a_2, dt):
        """
        :param obstacles:   List of obstacle objects
        :param uav:         UAV object
        :param a_1:         constant for 1st order CBF
        :param a_2:         constant for 2nd order CBF
        :param dt:          Time step for iteration
        """
        self.J = np.array([[0.082, 0, 0], [0, 0.0845, 0], [0, 0, 0.1377]])  # Inertia matrix
        self.J_inv = np.linalg.inv(self.J)  # Inverted intertia matrix
        self.m = 1  # Mass of the UAV

        self.k_v = 8
        self.k_R = 15
        self.k_omega = 2.6
        self.k_x = 7

        self.a_1 = a_1
        self.a_2 = a_2

        self.dt = dt
        self.start = None
        self.goal_pos = []
        self.goal_rot = []
        self.obstacles = obstacles
        self.uav = uav
        self.traj_gen = TrajGeneration()    # Object for accessing trajectory generation methods
        self.check_collision = CollisionCheck()     # Object for accessing collision checking methods
        self.rotation = Rotation()      # Object for accessing rotation methods

    def run(self, start, goal_data):
        """
        Main function of the obstacle avoidance algorithm and tries to follow a generated trajectory and avoid obstacles
        :param start:       Starting node of the desired trajectory
        :param goal_data:   List containing trajectory, duration of the trajectory and position and
                            rotational matrix of the goal of the trajectory
        :return:            Data of the obstacle avoidance generated trajectory
        """
        self.start = start
        self.goal_pos = goal_data[0]
        self.goal_rot = goal_data[1]
        if self.start.parent is not None:
            if self.start.isCBF:
                pre_state = self.start.traj[-2]
                state = self.start.traj[-1]
            else:
                T = self.start.duration
                pre_state = self.get_state(self.start.traj, T-self.dt)
                state = [self.start.data, self.start.rot, self.start.traj.get_body_rates(self.start.duration),
                         self.start.traj.get_thrust(self.start.duration)*self.m]
        else:
            pre_state = [self.start.data, self.start.rot, np.array([0, 0, 0]), 0]
            state = pre_state

        samples = []
        a = 0
        t = 0
        cbf_time = 0
        try:
            """Sampling through the motion primitive and using obstacle avoidance for every time instance"""
            while np.linalg.norm(state[0][0]-self.goal_pos) > 0.01:
                cbf_time += self.dt
                if a > 1200:
                    raise Exception
                a = a + 1
                t += self.dt

                u_des = self.get_u_des(state, pre_state)
                pre_state = state
                h, h_grad, h_dot, h_dot_grad = self.compute_h(state)
                f = self.f_func(state)
                g = self.g_func(state[1])
                Lf_psi = self.get_Lf_psi(f, h_grad, h_dot_grad)
                Lg_psi = self.get_Lg_psi(g, h_grad, h_dot_grad)
                beta = self.get_beta(h_dot, h)
                safe = self.check_safety(Lf_psi, Lg_psi, beta, u_des)   # Checking if the UAV is safe

                """If the UAV is not safe we try to generate a safe control signal by solving QP, 
                            otherwise we continue using u desired"""
                if not safe:
                    u_star = self.solve_QP(u_des, Lf_psi, Lg_psi, beta)
                else:
                    u_star = u_des
                state = self.update_state(state, u_star)
                samples.append(state)
            print("CBF worked!")
            isCBF = True
            duration = cbf_time
            return isCBF, samples, duration
        except:
            print("CBF failed!")
            isCBF = False
            duration = 0
            return isCBF, samples, duration

    def calc_R_c(self, state, pre_state):
        """
        Calculates the control attitude for the position based flight mode controller

        :param state:       The current state of the UAV
        :param pre_state:   The state of the UAV at the previous sample
        :return:            The control attitude and its time derivatives
        """
        a_pre = pre_state[0][2]
        a = state[0][2]
        v = state[0][1]
        x = state[0][0]
        x_d = self.goal_pos
        x_d_dot = np.array([0, 0, 0])
        x_d_ddot = np.array([0, 0, 0])
        e_v = v - x_d_dot
        e_v_dot = a
        e_v_ddot = (a - a_pre) / self.dt
        e_x = x - x_d
        e_x_dot = v
        e_x_ddot = a
        e3 = -np.array([0, 0, 1])
        g = 9.81

        A = np.array(-self.k_x * e_x - self.k_v * e_v - self.m * g * e3 + self.m * x_d_ddot)
        A_dot = np.array(-self.k_x * e_x_dot - self.k_v * e_v_dot)
        A_ddot = np.array(-self.k_x * e_x_ddot - self.k_v * e_v_ddot)
        b_3c = A / np.linalg.norm(A)
        b_3c_dot = -(-A_dot / np.linalg.norm(A) + np.dot(np.dot(A, A_dot) / np.linalg.norm(A) ** 3, A))
        b_3c_ddot = -(-A_ddot / np.linalg.norm(A) + np.dot(np.dot(2 * A, A_dot) / np.linalg.norm(A) ** 3, A_dot) + \
                      np.dot((np.linalg.norm(A_dot) ** 2 + np.dot(A, A_ddot)) / np.linalg.norm(A) ** 3, A) - \
                      3 * np.dot(np.dot(np.dot(A, A_dot), np.dot(A, A_dot)) / np.linalg.norm(A) ** 5, A))
        b_1d = np.array([self.goal_rot[0][0], self.goal_rot[1][0], self.goal_rot[2][0]])

        C = np.cross(b_3c, b_1d)
        C_dot = np.cross(b_3c_dot, b_1d)
        C_ddot = np.cross(b_3c_ddot, b_1d)
        b_2c = C / np.linalg.norm(C)
        b_2c_dot = -(-C_dot / np.linalg.norm(C) + np.dot(np.dot(C, C_dot) / np.linalg.norm(C) ** 3, C))
        b_2c_ddot = -(-C_ddot / np.linalg.norm(C) + np.dot(np.dot(2 * C, C_dot) / np.linalg.norm(C) ** 3, C_dot) + \
                      np.dot((np.linalg.norm(C_dot) ** 2 + np.dot(C, C_ddot)) / np.linalg.norm(C) ** 3, C) - \
                      3 * np.dot(np.dot(np.dot(C, C_dot), np.dot(C, C_dot)) / np.linalg.norm(C) ** 5, C))

        b_1c = np.cross(b_2c, b_3c)
        b_1c_dot = np.cross(b_2c_dot, b_3c) + np.cross(b_2c, b_3c_dot)
        b_1c_ddot = np.cross(b_2c_ddot, b_3c) + np.cross(2 * b_2c_dot, b_3c_dot) + np.cross(b_2c, b_3c_ddot)
        R_c = np.array([b_1c, b_2c, b_3c])
        R_c_dot = np.array([b_1c_dot, b_2c_dot, b_3c_dot])
        R_c_ddot = np.array([b_1c_ddot, b_2c_ddot, b_3c_ddot])

        return R_c.transpose(), R_c_dot.transpose(), R_c_ddot.transpose()

    def get_u_des(self, state, pre_state):
        """
        Calculates the desired control signal with a position based flight mode controller

        :param state:       The current state of the UAV
        :param pre_state:   The state of the UAV at the previous sample
        :return:            The desired control signal
        """
        v = state[0][1]
        x = state[0][0]
        x_d = self.goal_pos
        x_d_dot = np.array([0, 0, 0])
        x_d_ddot = np.array([0, 0, 0])
        e_v = v - x_d_dot
        e_x = x - x_d
        e3 = -np.array([0, 0, 1])
        g = 9.81

        R_c, R_c_dot, R_c_ddot = self.calc_R_c(state, pre_state)

        R = state[1]
        omega = state[2]

        omega_c_hat = np.dot(R_c.transpose(), R_c_dot)
        omega_c = np.array([-omega_c_hat[1][2], omega_c_hat[0][2], -omega_c_hat[0][1]])

        omega_c_dot_hat = np.dot(R_c.transpose(), R_c_ddot)-np.dot(omega_c_hat, omega_c_hat)
        omega_c_dot = np.array([-omega_c_dot_hat[1][2], omega_c_dot_hat[0][2], -omega_c_dot_hat[0][1]])

        e_omega = omega-np.dot(R.transpose(), np.dot(R_c, omega_c))
        e_R_hat = (1 / 2) * (np.dot(R_c.transpose(), R) - np.dot(R.transpose(), R_c))
        e_R = np.array([-e_R_hat[1][2], e_R_hat[0][2], -e_R_hat[0][1]])
        omega_hat = np.array([[0, -omega[2], omega[1]], [omega[2], 0, -omega[0]], [-omega[1], omega[0], 0]])

        f = np.dot(self.k_x*e_x+self.k_v*e_v+self.m*g*e3-self.m*x_d_ddot, np.dot(R, e3))

        if f < 0:
            f = 0
        elif f > 40:
            f = 40

        M = -self.k_R*e_R-self.k_omega*e_omega+np.cross(omega, np.dot(self.J, omega))-\
            np.dot(self.J, (np.dot(omega_hat, np.dot(R.transpose(), np.dot(R_c, omega_c)))-
                            np.dot(R.transpose(), np.dot(R_c, omega_c_dot))))
        max_index = np.argmax(M)
        min_index = np.argmin(M)
        if np.linalg.norm(x - self.start.data[0]) < 0.1:
            limit = 5
        else:
            limit = 40
        if M[max_index] > limit or M[min_index] < -limit:
            if abs(M[min_index]) > M[max_index]:
                M = limit / abs(M[min_index]) * M
            else:
                M = limit / M[max_index] * M
        u_des = np.array([f, M[0], M[1], M[2]])
        return u_des

    def get_cbf_cost(self, node):
        """
        :param samples:     Samples of the trajectory created by the obstacle avoidance
        :return:            The cost of the obstacle avoidance generated trajectory
        """
        jerk_squared = []
        samples = node.traj
        start = node.parent
        if start.parent is not None:
            if start.isCBF:
                pre_state = start.traj[-2]
            else:
                pre_state = self.get_state(start.traj, start.duration-self.dt)
        else:
            pre_state = samples[0]
        for state in samples:
            omega_hat = np.array([[0, -state[2][2], state[2][1]], [state[2][2], 0, -state[2][0]],
                              [-state[2][1], state[2][0], 0]])
            jerk_squared.append(np.linalg.norm(np.dot(state[1], np.dot(omega_hat, np.array([0, 0, state[-1]])) +
                                                            np.dot(state[1], np.array(
                                                                [0, 0, (state[-1] - pre_state[-1]) / self.dt])))) ** 2)
            pre_state = state
        cost = sum(jerk_squared)*self.dt / node.duration
        return cost

    def get_traj_samples(self, traj, T, num_samples):
        """
        :param traj:            Trajectory object.
        :param T:               Duration of the trajectory.
        :param num_samples:     Number of samples of the trajectory.
        :return:                List of samples of the state of the trajectory.
        """
        time_points = np.linspace(0, T, num_samples)
        traj_samples = []
        for t in time_points:
            sample = self.get_state(traj, t)
            traj_samples.append(sample)
        return traj_samples

    def get_state(self, traj, t):
        """
        Gets the state data from the RapidTrajectory object traj
        :param traj:            Trajectory object
        :param t:               Time instance of the trajectory that is sampled
        :return:                return state data as a list including [pos, vel, acc], attitude, body rates and thrust
        """
        rates = traj.get_body_rates(t)
        thrust = self.m * traj.get_thrust(t)
        rot = self.rotation.normalvec_to_R(traj.get_normal_vector(t))
        pos = traj.get_position(t)
        vel = traj.get_velocity(t)
        acc = traj.get_acceleration(t)
        state = [[pos, vel, acc], rot, rates, thrust]
        return state

    def state_dot_func(self, s, t, J, J_inv, u):
        """
        Calculates the rate of change of the current state
        :param s: 		The state of the UAV at the current sample
        :param t: 		Time
        :param J: 		Inertia matrix
        :param J_inv: 	Inverted inertia matrix
        :param u: 		The control signal solved with QP
        :return: 		The time derivative of the new state
        """
        omega = np.array([s[6], s[7], s[8]])
        cross_mult = np.cross(-omega, J.dot(omega))
        omega_dot = J_inv.dot(cross_mult)
        r_1_1 = s[9]
        r_1_2 = s[10]
        r_1_3 = s[11]
        r_2_1 = s[12]
        r_2_2 = s[13]
        r_2_3 = s[14]
        r_3_1 = s[15]
        r_3_2 = s[16]
        r_3_3 = s[17]
        p = omega[0]
        q = omega[1]
        r = omega[2]
        f = np.array([s[3],s[4],s[5], 0, 0, -9.81, omega_dot[0], omega_dot[1], omega_dot[2],
                  r_1_2*r-r_1_3*q, r_1_3*p-r_1_1*r, r_1_1*q-r_1_2*p, r_2_2*r-r_2_3*q, r_2_3*p-r_2_1*r,
                  r_2_1*q-r_2_2*p, r_3_2*r-r_3_3*q, r_3_3*p-r_3_1*r, r_3_1*q-r_3_2*p])
        g_dot_u = np.array([0,0,0,u[0]*s[11]/self.m,u[0]*s[14]/self.m,u[0]*s[17]/self.m,J_inv[0][0]*u[1]+J_inv[0][1]*u[2]+
                            J_inv[0][2]*u[3], J_inv[1][0]*u[1]+J_inv[1][1]*u[2]+J_inv[1][2]*u[3],
                            J_inv[2][0]*u[1]+J_inv[2][1]*u[2]+J_inv[2][2]*u[3],0,0,0,0,0,0,0,0,0])
        state_dot = f+g_dot_u
        return state_dot

    def update_state(self, state, u):
        """
        Integrates the state’s rate of change to get the next state after dt seconds
        :param state:  Current state of the UAV
        :param u:	   Control signal
        :return:	   The next state of the UAV (by integrating the array returned by state_dot_func)
        """
        y0 = [state[0][0][0],state[0][0][1],state[0][0][2],state[0][1][0],state[0][1][1],state[0][1][2], state[2][0],
              state[2][1],state[2][2],state[1][0][0],state[1][0][1],state[1][0][2],state[1][1][0],state[1][1][1],
              state[1][1][2],state[1][2][0],state[1][2][1],state[1][2][2]]
        sol = odeint(self.state_dot_func, y0, np.array([0,self.dt]), args=(self.J, self.J_inv, u))
        data = sol[1]
        new_state = [[np.array([data[0], data[1], data[2]]), np.array([data[3], data[4], data[5]]),
                      np.array([u[0]*data[11]/self.m, u[0]*data[14]/self.m, u[0]*data[17]/self.m-9.81])],
                     np.array([[data[9], data[10], data[11]], [data[12], data[13], data[14]],
                               [data[15], data[16], data[17]]]), np.array([data[6], data[7], data[8]]), u[0]]
        return new_state

    def f_func(self, state):
        """
        :param state: 	The latest state of the UAV
        :return:		18x1 array f(x) = [v, gravity, J^-1, J^-1(-omega x J*omega), R[omega]_x]
        """
        vel = state[0][1]
        X_rest = state[1:]
        p = X_rest[1][0]
        q = X_rest[1][1]
        r = X_rest[1][2]
        r_1_1 = X_rest[0][0][0]
        r_1_2 = X_rest[0][0][1]
        r_1_3 = X_rest[0][0][2]
        r_2_1 = X_rest[0][1][0]
        r_2_2 = X_rest[0][1][1]
        r_2_3 = X_rest[0][1][2]
        r_3_1 = X_rest[0][2][0]
        r_3_2 = X_rest[0][2][1]
        r_3_3 = X_rest[0][2][2]
        omega = np.array([p, q, r])
        omega = np.transpose(omega)
        cross_mult = np.cross(-omega, self.J.dot(omega))
        omega_dot = self.J_inv.dot(cross_mult)
        f = np.array([vel[0], vel[1], vel[2], 0, 0, -9.81, omega_dot[0], omega_dot[1], omega_dot[2],
                           r_1_2*r-r_1_3*q, r_1_3*p-r_1_1*r, r_1_1*q-r_1_2*p, r_2_2*r-r_2_3*q, r_2_3*p-r_2_1*r,
                           r_2_1*q-r_2_2*p, r_3_2*r-r_3_3*q, r_3_3*p-r_3_1*r, r_3_1*q-r_3_2*p])
        return f

    def g_func(self, rot):
        """
        :param rot:		Rotational matrix from current state
        :return:		18x4 array matrix g(x)= [0 3x4, R/m e_3, J^-1, 0 9x4]
        """
        g = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                      [rot[0][2]/self.m, 0, 0, 0], [rot[1][2]/self.m, 0, 0, 0], [rot[2][2]/self.m, 0, 0, 0],
                      [0, self.J_inv[0][0], self.J_inv[0][1], self.J_inv[0][2]],
                      [0, self.J_inv[1][0], self.J_inv[1][1], self.J_inv[1][2]],
                      [0, self.J_inv[2][0], self.J_inv[2][1], self.J_inv[2][2]],
                      [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                      [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        return g

    def compute_h(self, state):
        """
        Updates the positions of the 16 “sphere centre points” on the UAV. For every obstacle,
        compute_h calls h_func for every “sphere centre point” on the UAV

        :param state: Current state of the UAV
        :return: h(x), the time derivative of h(x), the gradient of h(x) and the gradient of the time derivative of h(x)
                 for all the "sphere centre points" of the UAV
        """
        position = state[0][0]
        rot = state[1]
        self.uav.update(position, rot)
        h_P = []
        h_dot_P = []
        h_grad_P = []
        h_dot_grad_P = []
        for obs in self.obstacles:
            for P in self.uav.points:
                h_P_i, h_grad_P_i, h_dot_P_i, h_dot_grad_P_i = self.h_func(obs, state, P)
                h_P.append(h_P_i)
                h_grad_P.append(h_grad_P_i)
                h_dot_P.append(h_dot_P_i)
                h_dot_grad_P.append(h_dot_grad_P_i)

        return h_P, h_grad_P, h_dot_P, h_dot_grad_P

    def h_func(self, obs, state, P):
        """
        Calculates h(x) and its time derivative and also both their gradients
        :param obs:		One obstacle
        :param P:		One of the “sphere centre points” for the current state of the UAV
        :param state:	Current state
        :return:		h, h_grad, h_dot, h_dot_grad for one of the “sphere centre points” of the UAV
        """
        [min_coord, max_coord] = obs.poly.bbox
        x_interval = [min_coord[0][0], max_coord[0][0]]
        y_interval = [min_coord[1][0], max_coord[1][0]]
        z_interval = [min_coord[2][0], max_coord[2][0]]

        """Calculates the closest point on all the half planes of the obstacle polytope to the current sphere centre point"""
        Q_list = []
        for normal, d in zip(obs.poly.A, obs.poly.b):
            k = -(normal[0] * P[0] + normal[1] * P[1] + normal[2] * P[2] - d) / (
                        normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
            Q = P + k * normal
            Q_edge = []
            if x_interval[0] <= Q[0] <= x_interval[1] and y_interval[0] <= Q[1] <= y_interval[1] \
                    and z_interval[0] <= Q[2] <= z_interval[1]:
                Q_list.append(Q)
            else:
                if x_interval[0] <= P[0] <= x_interval[1]:
                    Q_edge.append(P[0])
                elif x_interval[0] > P[0]:
                    Q_edge.append(x_interval[0])
                else:
                    Q_edge.append(x_interval[1])
                if y_interval[0] <= P[1] <= y_interval[1]:
                    Q_edge.append(P[1])
                elif y_interval[0] > P[1]:
                    Q_edge.append(y_interval[0])
                else:
                    Q_edge.append(y_interval[1])
                if z_interval[0] <= P[2] <= z_interval[1]:
                    Q_edge.append(P[2])
                elif z_interval[0] > P[2]:
                    Q_edge.append(z_interval[0])
                else:
                    Q_edge.append(z_interval[1])
                Q_list.append(Q_edge)

        h_list = []
        for Q_i in Q_list:
            """"h(x) function"""
            h_list.append(((P[0] - Q_i[0]) ** 2 + (P[1] - Q_i[1]) ** 2 + (P[2] - Q_i[2]) ** 2) ** 0.5 - self.uav.sphere_rad)
        index = np.argmin(h_list)
        if x_interval[0] <= P[0] <= x_interval[1] and y_interval[0] <= P[1] <= y_interval[1] \
                and z_interval[0] <= P[2] <= z_interval[1]:
            h = -h_list[index]
        else:
            h = h_list[index]

        Q = Q_list[index]

        x = self.uav.pos[0]
        y = self.uav.pos[1]
        z = self.uav.pos[2]
        x_dot = state[0][1][0]
        y_dot = state[0][1][1]
        z_dot = state[0][1][2]
        R = state[1]

        """Calculates the position of a point on the sphere surface in the direction 
        from the sphere centre point to the closest obstacle point"""
        dir = (P-Q)/abs(np.linalg.norm(P-Q))
        R_inv = np.linalg.inv(R)
        P0 = np.dot(R_inv, P + self.uav.sphere_rad * dir - self.uav.pos)

        R_P_x = R[0][0]*P0[0]+R[0][1]*P0[1]+R[0][2]*P0[2]
        R_P_y = R[1][0]*P0[0]+R[1][1]*P0[1]+R[1][2]*P0[2]
        R_P_z = R[2][0]*P0[0]+R[2][1]*P0[1]+R[2][2]*P0[2]
        omega = state[2]
        R_dot = np.array([[R[0][1]*omega[2]-R[0][2]*omega[1], R[0][2]*omega[0]-R[0][0]*omega[2], R[0][0]*omega[1]-R[0][1]*omega[0]],
                          [R[1][1]*omega[2]-R[1][2]*omega[1], R[1][2]*omega[0]-R[1][0]*omega[2], R[1][0]*omega[1]-R[1][1]*omega[0]],
                          [R[2][1]*omega[2]-R[2][2]*omega[1], R[2][2]*omega[0]-R[2][0]*omega[2], R[2][0]*omega[1]-R[2][1]*omega[0]]])
        R_P_x_dot = R_dot[0][0]*P0[0]+R_dot[0][1]*P0[1]+R_dot[0][2]*P0[2]
        R_P_y_dot = R_dot[1][0]*P0[0]+R_dot[1][1]*P0[1]+R_dot[1][2]*P0[2]
        R_P_z_dot = R_dot[2][0]*P0[0]+R_dot[2][1]*P0[1]+R_dot[2][2]*P0[2]

        p_q_x = x-Q[0]+R_P_x
        p_q_y = y-Q[1]+R_P_y
        p_q_z = z-Q[2]+R_P_z

        norm_sqrt = (p_q_x ** 2 + p_q_y ** 2 + p_q_z ** 2) ** 0.5

        norm_3_2 = (p_q_x ** 2 + p_q_y ** 2 + p_q_z ** 2) ** 1.5

        drot_x = (x_dot-P0[2]*(omega[0]*R[0][1]-omega[1]*R[0][0])+P0[1]*
                  (omega[0]*R[0][2]-omega[2]*R[0][0])-P0[0]*(omega[1]*R[0][2]-omega[2]*R[0][1]))

        drot_y = (y_dot-P0[2]*(omega[0]*R[1][1]-omega[1]*R[1][0])+P0[1]*
                  (omega[0]*R[1][2]-omega[2]*R[1][0])-P0[0]*(omega[1]*R[1][2]-omega[2]*R[1][1]))

        drot_z = (z_dot-P0[2]*(omega[0]*R[2][1]-omega[1]*R[2][0])+P0[1]*
                  (omega[0]*R[2][2]-omega[2]*R[2][0])-P0[0]*(omega[1]*R[2][2]-omega[2]*R[2][1]))

        sum_mult = (drot_x * p_q_x + drot_y * p_q_y + drot_z * p_q_z)

        """Time derivative of h(x)"""
        h_dot = ((R_P_x_dot+x_dot)*p_q_x+(R_P_y_dot+y_dot)*p_q_y+(R_P_z_dot+z_dot)*p_q_z)/norm_sqrt

        """Gradient of h(x)"""
        h_grad = np.array([p_q_x/norm_sqrt, p_q_y/norm_sqrt, p_q_z/norm_sqrt, 0, 0, 0, 0, 0, 0,
                               (P0[0]*p_q_x)/norm_sqrt, (P0[1]*p_q_x)/norm_sqrt, (P0[2]*p_q_x)/norm_sqrt,
                               (P0[0]*p_q_y)/norm_sqrt, (P0[1]*p_q_y)/norm_sqrt, (P0[2]*p_q_y)/norm_sqrt,
                               (P0[0]*p_q_z)/norm_sqrt, (P0[1]*p_q_z)/norm_sqrt, (P0[2]*p_q_z)/norm_sqrt])

        """Gradient of the time derivative of h(x)"""
        h_dot_grad = np.array([drot_x/norm_sqrt - (sum_mult*p_q_x)/norm_3_2,
                                   drot_y/norm_sqrt - (sum_mult*p_q_y)/norm_3_2,
                                   drot_z/norm_sqrt - (sum_mult*p_q_z)/norm_3_2, p_q_x/norm_sqrt,
                                   p_q_y/norm_sqrt, p_q_z/norm_sqrt,
                                   ((P0[1]*R[0][2]-P0[2]*R[0][1])*p_q_x+(P0[1]*R[1][2]-P0[2]*R[1][1])*p_q_y+
                                    (P0[1]*R[2][2]-P0[2]*R[2][1])*p_q_z)/norm_sqrt,
                                   -((P0[0]*R[0][2]-P0[2]*R[0][0])*p_q_x+(P0[0]*R[1][2]-P0[2]*R[1][0])*p_q_y+
                                     (P0[0]*R[2][2]-P0[2]*R[2][0])*p_q_z)/norm_sqrt,
                                   ((P0[0]*R[0][1]-P0[1]*R[0][0])*p_q_x+(P0[0]*R[1][1]-P0[1]*R[1][0])*p_q_y+
                                    (P0[0]*R[2][1]-P0[1]*R[2][0])*p_q_z)/norm_sqrt,
                                   (P0[0]*drot_x + (P0[2]*omega[1] - P0[1]*omega[2])*p_q_x)/norm_sqrt -
                                   (P0[0]*sum_mult*p_q_x)/norm_3_2,
                                   (P0[1]*drot_x - (P0[2]*omega[0] - P0[0]*omega[2])*p_q_x)/norm_sqrt -
                                   (P0[1]*sum_mult*p_q_x)/norm_3_2,
                                   (P0[2]*drot_x + (P0[1]*omega[0] - P0[0]*omega[1])*p_q_x)/norm_sqrt -
                                   (P0[2]*sum_mult*p_q_x)/norm_3_2,
                                   (P0[0]*drot_y + (P0[2]*omega[1] - P0[1]*omega[2])*p_q_y)/norm_sqrt -
                                   (P0[0]*sum_mult*p_q_y)/norm_3_2,
                                   (P0[1]*drot_y - (P0[2]*omega[0] - P0[0]*omega[2])*p_q_y)/norm_sqrt -
                                   (P0[1]*sum_mult*p_q_y)/norm_3_2,
                                   (P0[2]*drot_y + (P0[1]*omega[0] - P0[0]*omega[1])*p_q_y)/norm_sqrt -
                                   (P0[2]*sum_mult*p_q_y)/norm_3_2,
                                   (P0[0]*drot_z + (P0[2]*omega[1] - P0[1]*omega[2])*p_q_z)/norm_sqrt -
                                   (P0[0]*sum_mult*p_q_z)/norm_3_2,
                                   (P0[1]*drot_z - (P0[2]*omega[0] - P0[0]*omega[2])*p_q_z)/norm_sqrt -
                                   (P0[1]*sum_mult*p_q_z)/norm_3_2,
                                   (P0[2]*drot_z + (P0[1]*omega[0] - P0[0]*omega[1])*p_q_z)/norm_sqrt -
                                   (P0[2]*sum_mult*p_q_z)/norm_3_2])
        if x_interval[0] <= P[0] <= x_interval[1] and y_interval[0] <= P[1] <= y_interval[1] \
                and z_interval[0] <= P[2] <= z_interval[1]:
            h_dot = -h_dot
            h_grad = -h_grad
            h_dot_grad = -h_dot_grad

        return h, h_grad, h_dot, h_dot_grad

    def get_Lf_psi(self, f, h_grad, h_dot_grad):
        """
        :param f:			f(x) from f_func
        :param h_grad:		The gradient of h(x) as an array from compute_h
        :param h_dot_grad:	The gradient of the time derivative of h(x) -||-
        :return:			Array containing Lf_psi
        """
        Lf_psi = []
        for h_grad_i, h_dot_grad_i in zip(h_grad, h_dot_grad):
            Lf_psi.append(np.dot(h_dot_grad_i+self.a_1*h_grad_i, f.transpose()))
        return np.array(Lf_psi)

    def get_Lg_psi(self, g, h_grad, h_dot_grad):
        """
        :param g:			g from f_func
        :param h_grad:		The gradient of h(x) as an array from compute_h
        :param h_dot_grad:	The gradient of the time derivative of h(x) -||-
        :return:			Array containing Lg_psi
        """
        Lg_psi = []
        for h_grad_i, h_dot_grad_i in zip(h_grad, h_dot_grad):
            Lg_psi.append(np.dot(h_dot_grad_i+self.a_1*h_grad_i, g))
        return np.array(Lg_psi)

    def get_beta(self, h_dot, h):
        """
        beta = a_2*psi_1

        :param h_dot:	The time derivative of h(x) as an array from compute_h
        :param h:		h(x) -||-
        :return:		Array containing beta
        """
        beta = []
        for h_i, h_dot_i in zip(h, h_dot):
            beta.append(self.a_2*(h_dot_i+self.a_1*h_i))
        return np.array(beta).transpose()

    def check_safety(self, Lf_psi, Lg_psi, beta, u):
        """
        :param Lf_psi:	Array of Lf_psi from get_Lf_psi
        :param Lg_psi:	Array of Lg_psi from get_Lg_psi
        :param beta:	Array of beta(psi(x))from get_Lf_psi
        :param u:		u desired
        :return:		Returns True if Lf psi_1(x) + Lg psi_1(x)u + a_2*psi_1(x) => 0 and False otherwise
        """
        is_safe = Lf_psi + Lg_psi.dot(u) + beta
        return all(i >= 0 for i in is_safe)

    def solve_QP(self, u_des, Lf_psi, Lg_psi, beta):
        """
        Tries to find the optimal control signal by minimizing ||u-u_des|| while considering the C
        BF constraints and the actuation constraints

        :param u_des:	Desired u based on generated trajectory
        :param Lf_psi:	Array of Lf psi_1(x) from get_Lf_psi
        :param Lg_psi:	Array of Lg psi_1(x) from get_Lg_psi
        :param beta:	Array of a_2*psi_1(x) from get_Lf_psi
        :return: 	    New control signal solved with QP
        """
        P = matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], tc='d')

        q = -matrix(u_des)
        G = -np.array([Lg_psi[0][0], Lg_psi[0][1], Lg_psi[0][2], Lg_psi[0][3]])
        for constraint in Lg_psi[1:]:
            new_const = np.array([-constraint[0], -constraint[1], -constraint[2], -constraint[3]])
            G = np.vstack((G, new_const))

        u_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                             [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])

        u_limits = np.array([40., 40., 40., 40., 0, 40., 40., 40.])   # Actuation constraints

        G = np.vstack((G, u_matrix))
        h = []
        i = 0
        for constraint in Lf_psi:
            new_const = (constraint+beta[i])
            h.append(new_const)
            i = i+1

        for limit in u_limits: h.append(limit)
        h = matrix(np.transpose(h), tc='d')

        G = matrix(G, tc='d')
        sol = solvers.qp(P, q, G, h)
        u_n = np.array(sol['x']).flatten()

        return u_n


class Obstacle:
    def __init__(self, pos, dim):
        """
        :param pos:     Position of the center of mass of an obstacle
        :param dim:     Dimenions of an obstacle
        """
        self.pos = pos
        self.dim = dim

        self.poly = self.create_poly()
        self.poly.bbox = self.poly.bounding_box
        self.poly.vertices = pc.extreme(self.poly)
        self.rad = self.calc_rad()

    def create_poly(self):
        """
        :return:    Polytope of an obstacle
        """
        x = self.pos[0]
        y = self.pos[1]
        z = self.pos[2]
        obs_x = self.dim[0]
        obs_y = self.dim[1]
        obs_z = self.dim[2]

        A = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [-1, -0, -0],
                      [-0, -1, -0],
                      [-0, -0, -1]])

        b = np.array([obs_x / 2 + x, obs_y / 2 + y, obs_z / 2 + z, obs_x / 2 - x, obs_y / 2 - y,
                      obs_z / 2 - z])
        return pc.Polytope(A, b)

    def calc_rad(self):
        """
        :return:    The radius of the sphere bounding the obstacle cuboid
        """
        obs_rad = 0
        for vertex in self.poly.vertices:
            rad = np.linalg.norm(vertex - self.pos)
            if rad > obs_rad:
                obs_rad = rad
        return obs_rad


class UAV:
    def __init__(self, pos, width, height):
        """
        :param pos:     Position of the center of mass of the UAV
        :param width:   Width of the UAV cuboid
        :param height:  Height of the UAV cuboid
        """
        self.width = width
        self.height = height
        self.pos = pos
        self.poly_init = self.create_poly_init()
        self.poly_init.vertices = pc.extreme(self.poly_init)
        self.poly_init.bbox = self.poly_init.bounding_box
        self.rad = self.calc_rad()
        self.sphere_rad = self.width / 6
        self.init_points = self.create_init_points()

        self.poly = self.poly_init
        self.points = self.init_points

    def create_poly_init(self):
        """
        :return:    The UAV polytope in the B frame
        """
        A = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [-1, -0, -0],
                      [-0, -1, -0],
                      [-0, -0, -1]])

        b = np.array(
            [self.width / 2, self.width / 2, self.height / 2, self.width / 2, self.width / 2, self.height / 2])
        return pc.Polytope(A, b)

    def calc_rad(self):
        """
        :return:    The radius of the sphere bounding the UAV cuboid
        """
        uav_rad = 0
        for vertex in self.poly_init.vertices:  # calculate UAV radius, i.e. furthest extreme point from c.o.m.
            rad = np.linalg.norm(vertex - self.pos)
            if rad > uav_rad:
                uav_rad = rad
        return uav_rad

    def create_init_points(self):
        """
        :return:    Returns the sphere center points of the UAV in the B frame
        """
        [min_coord, max_coord] = self.poly_init.bbox

        x_interval = [min_coord[0][0], max_coord[0][0]]
        y_interval = [min_coord[1][0], max_coord[1][0]]

        x_points = np.linspace(x_interval[0] + self.sphere_rad - self.width / 6,
                               x_interval[1] - self.sphere_rad + self.width / 6, 4)
        y_points = np.linspace(y_interval[0] + self.sphere_rad - self.width / 6,
                               y_interval[1] - self.sphere_rad + self.width / 6, 4)
        z = 0
        points = []
        for x in x_points:
            for y in y_points:
                points.append([x, y, z])
        return np.array(points)

    def update(self, new_pos, new_rot):
        """
        :param new_pos:     New position of the center of mass of the UAV
        :param new_rot:     New attitude of the UAV
        """
        self.pos = new_pos
        vertex_rot = np.dot(new_rot, self.poly_init.vertices.transpose()).transpose() + new_pos
        self.points = np.dot(new_rot, self.init_points.transpose()).transpose() + new_pos
        self.poly = pc.qhull(vertex_rot)


class CollisionCheck:

    def __init__(self):
        self.rotation = Rotation()

    def point_collision(self, point, obstacles):
        """
        :param point:       A three dimensional coordinate
        :param obstacles:   List containing obstacle objects
        :return:            Returns True if the coordinate coincides with an obstacle
        """
        for obstacle in obstacles:
            if obstacle.poly.__contains__(point):
                return True
        return False

    def traj_collision(self, traj, T, numSamples, obstacles, uav):
        """
        :param traj:         RapidTrajectory object
        :param T:            Duration of the trajectory
        :param numSamples:   Number of samples to check for collision
        :param obstacles:    List containing obstacle objects
        :param uav:          UAV object
        :return:             Returns True if there is a collision with the index of the concerned obstacle,
                             otherwise returns False
        """
        time_points = np.linspace(0, T, numSamples)
        for t in time_points:
            obs_index = 0
            position = traj.get_position(t)
            normalvec = traj.get_normal_vector(t)
            uav.update(position, self.rotation.normalvec_to_R(normalvec))
            for obs in obstacles:
                if np.linalg.norm(position - np.array(obs.pos)) <= obs.rad + uav.rad:
                    if self.uav_collision(uav, [obs]):
                        return True, obs_index
                obs_index += 1
        return False, obs_index

    def uav_collision(self, uav, obstacles):
        """
        :param uav:         UAV object
        :param obstacles:   List containing obstacle objects
        :return:            Returns True if there is a collision, False otherwise
        """
        for obs in obstacles:
            if np.linalg.norm(uav.pos - obs.pos) <= (obs.rad + uav.rad):
                if uav.poly.intersect(obs.poly):
                    return True
        return False


def plotTraj(q, obstacles, rrt, uav, min_bound, max_bound):
    """
    :param q:           A vertex in the trajectory
    :param obstacles:   List containing obstacle objects
    :param rrt:         RRT object
    :param uav:         UAV object
    :param min_bound:   Minimum boundary of the box bounding the trajectory
    :param max_bound:   Maximum boundary of the box bounding the trajectory
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    num_obs = len(obstacles)-6
    for obs, color in zip(obstacles[0:num_obs], ['b', 'r', 'g', 'y', 'c', 'b', 'r', 'g', 'y', 'c', 'b', 'r']):
        verts = obs.poly.vertices
        hull = ConvexHull(verts)
        for s in hull.simplices:
            sq = [[verts[s[0], 0], verts[s[0], 1], verts[s[0], 2]],
                  [verts[s[1], 0], verts[s[1], 1], verts[s[1], 2]],
                  [verts[s[2], 0], verts[s[2], 1], verts[s[2], 2]]]

            f = a3.art3d.Poly3DCollection([sq])
            f.set_color(color)
            f.set_edgecolor('k')
            f.set_alpha(1)
            ax.add_collection3d(f)

    rotation = Rotation()
    while q is not rrt.G.root:
        if q.isCBF:
            num_samples = len(q.traj)
            time_samples = np.linspace(0, num_samples - 1, num_samples)
            time_points = time_samples[0::25]
        else:
            T = q.duration
            time_points = np.linspace(0, T, round(T*5))
        for t in time_points:
            if q.isCBF:
                pos = q.traj[round(t)][0][0]
                rot = q.traj[round(t)][1]
            else:
                pos = q.traj.get_position(t)
                rot = rotation.normalvec_to_R(q.traj.get_normal_vector(t))
            uav.update(pos, rot)
            verts = np.array(pc.extreme(uav.poly))
            hull = ConvexHull(verts)
            for s in hull.simplices:
                sq = [[verts[s[0], 0], verts[s[0], 1], verts[s[0], 2]],
                      [verts[s[1], 0], verts[s[1], 1], verts[s[1], 2]],
                      [verts[s[2], 0], verts[s[2], 1], verts[s[2], 2]]]
                f = a3.art3d.Poly3DCollection([sq])
                f.set_color('g')
                f.set_edgecolor('k')
                f.set_alpha(1)
                ax.add_collection3d(f)
        q = q.parent

    for i in ["x", "y", "z"]:
        eval("ax.set_{:s}label('{:s}')".format(i, i))

    ax.set_xlim(min_bound, max_bound)
    ax.set_ylim(min_bound, max_bound)
    ax.set_zlim(min_bound, max_bound)

    plt.show()



def main():
    uav_width = 0.5
    uav_height = 0.1

    root_pos = np.array([1.25, 1.25, 1])

    goal_pos = np.array([3.75, 3.75, 4])

    uav = UAV([0, 0, 0], uav_width, uav_height)

    max_bound = 5
    min_bound = 0
    interval = [[min_bound, max_bound], [min_bound, max_bound], [min_bound, max_bound]]

    """Scenario 1: Cubic obstacles in a checkerbaord pattern"""

    obs_pos = [[1.25, 1.25, 2.5], [1.25, 3.75, 2.5], [3.75, 1.25, 2.5], [2.5, 2.5, 2.5], [3.75, 3.75, 2.5],
               [min_bound, (max_bound + min_bound) / 2, (max_bound + min_bound) / 2],
               [(max_bound + min_bound) / 2, min_bound, (max_bound + min_bound) / 2],
               [max_bound, (max_bound + min_bound) / 2, (max_bound + min_bound) / 2],
               [(max_bound + min_bound) / 2, max_bound, (max_bound + min_bound) / 2],
               [(max_bound + min_bound) / 2, (max_bound + min_bound) / 2, max_bound],
               [(max_bound + min_bound) / 2, (max_bound + min_bound) / 2, min_bound]]
    obs_dims = [[1.25, 1.25, 1.25], [1.25, 1.25, 1.25], [1.25, 1.25, 1.25], [1.25, 1.25, 1.25], [1.25, 1.25, 1.25],
                [0.01, max_bound - min_bound, max_bound - min_bound],
                [max_bound - min_bound, 0.01, max_bound - min_bound],
                [0.01, max_bound - min_bound, max_bound - min_bound],
                [max_bound - min_bound, 0.01, max_bound - min_bound],
                [max_bound - min_bound, max_bound - min_bound, 0.01],
                [max_bound - min_bound, max_bound - min_bound, 0.01]]

    """Scenario 2: A wall with a narrow gap in the middle of the room"""

    """obs_pos = [[2.5, 3.875, 2.5], [2.5, 1.125, 2.5], [min_bound, (max_bound + min_bound) / 2, (max_bound + min_bound) / 2],
               [(max_bound + min_bound) / 2, min_bound, (max_bound + min_bound) / 2],
               [max_bound, (max_bound + min_bound) / 2, (max_bound + min_bound) / 2],
               [(max_bound + min_bound) / 2, max_bound, (max_bound + min_bound) / 2],
               [(max_bound + min_bound) / 2, (max_bound + min_bound) / 2, max_bound],
               [(max_bound + min_bound) / 2, (max_bound + min_bound) / 2, min_bound]]
    obs_dims = [[0.01, 2.25, 5], [0.01, 2.25, 5], [0.01, max_bound-min_bound, max_bound-min_bound], 
                [max_bound-min_bound, 0.01, max_bound-min_bound], [0.01, max_bound-min_bound, max_bound-min_bound], 
                [max_bound-min_bound, 0.01, max_bound-min_bound], [max_bound-min_bound, max_bound-min_bound, 0.01], 
                [max_bound-min_bound, max_bound-min_bound, 0.01]]"""

    """Scenario 3: Complex environment with several narrow gaps and two barriers of stacked thin obstacles"""

    """obs_pos = [[2.5, 0.4375, 2.5], [2.5, 1.8125, 2.5], [2.5, 3.1875, 2.5], [2.5, 4.5625, 2.5], [1.3, 2.5, 2.5],
               [1.3, 2.5, 4.5], [1.3, 2.5, 0.5], [3.85, 2.5, 2.5], [3.85, 2.5, 4.5], [3.85, 2.5, 0.5],
               [min_bound, (max_bound + min_bound) / 2, (max_bound + min_bound) / 2],
               [(max_bound + min_bound) / 2, min_bound, (max_bound + min_bound) / 2],
               [max_bound, (max_bound + min_bound) / 2, (max_bound + min_bound) / 2],
               [(max_bound + min_bound) / 2, max_bound, (max_bound + min_bound) / 2],
               [(max_bound + min_bound) / 2, (max_bound + min_bound) / 2, max_bound],
               [(max_bound + min_bound) / 2, (max_bound + min_bound) / 2, min_bound]]

    obs_dims = [[0.05, 0.875, 5], [0.05, 0.875, 5], [0.05, 0.875, 5], [0.05, 0.875, 5], [0.1, 5, 1], [0.1, 5, 1],
                [0.1, 5, 1], [0.1, 5, 1], [0.1, 5, 1], [0.1, 5, 1], [0.01, max_bound-min_bound, max_bound-min_bound],
                [max_bound-min_bound, 0.01, max_bound-min_bound], [0.01, max_bound-min_bound, max_bound-min_bound],
                [max_bound-min_bound, 0.01, max_bound-min_bound], [max_bound-min_bound, max_bound-min_bound, 0.01],
                [max_bound-min_bound, max_bound-min_bound, 0.01]]"""

    """Three rooms obstacles"""

    """obs_pos = [[0.5, 2, 2.5], [2, 2, 2.5], [2.5, 2.75, 2.5], [2.5, 4.5, 2.5], [3.75, 1.85, 2.5],
               [min_bound, (max_bound + min_bound) / 2, (max_bound + min_bound) / 2],
               [(max_bound + min_bound) / 2, min_bound, (max_bound + min_bound) / 2],
               [max_bound, (max_bound + min_bound) / 2, (max_bound + min_bound) / 2],
               [(max_bound + min_bound) / 2, max_bound, (max_bound + min_bound) / 2],
               [(max_bound + min_bound) / 2, (max_bound + min_bound) / 2, max_bound],
               [(max_bound + min_bound) / 2, (max_bound + min_bound) / 2, min_bound]]
    obs_dims = [[1, 0.05, 5], [1, 0.05, 5], [0.05, 1.5, 5], [0.05, 1, 5], [2.5, 1.5, 5], [0.01, max_bound-min_bound, 
                max_bound-min_bound], [max_bound-min_bound, 0.01, max_bound-min_bound], [0.01, max_bound-min_bound, 
                max_bound-min_bound], [max_bound-min_bound, 0.01, max_bound-min_bound], 
                [max_bound-min_bound, max_bound-min_bound, 0.01], [max_bound-min_bound, max_bound-min_bound, 0.01]]"""

    obstacles = []
    for pos, dim in zip(obs_pos, obs_dims):
        obstacles.append(Obstacle(pos, dim))

    use_cbf = False  # True if you want to use RRT* with obstacle avoidance, False if you only want to use RRT*
    a_1 = 20   # Alpha 1 constant for CBF
    a_2 = 70   # Alpha 2 constant for CBF
    dt_cbf = 4e-3   # Obstacle avoidance sampling time

    cbf = CBF(obstacles, uav, a_1, a_2, dt_cbf)

    K = 200 #3000    # Number of randomly generated vertices
    d_q = 3 #1.5   # Incremental distance
    dt_rrt = 25e-3  # Time interval in collision checking

    # random.seed(22)  # Used to save an RRT* run
    rrt = RRT(root_pos, goal_pos, obstacles, uav, K, interval, d_q, dt_rrt, cbf, use_cbf)
    rrt.run()

    q = rrt.G.goal
    plotTraj(q, obstacles, rrt, uav, min_bound, max_bound)

    return get_rrt_data(rrt, uav, obstacles)


if __name__ == "__main__":
    main()

