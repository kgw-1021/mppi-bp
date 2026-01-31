# MPPI-BP (MPPI Belief Propagation)

## Algorithm Overview

This project implements a **MPPI-BP** framework for multi-agent navigation. Instead of relying on a centralized solver, each agent maintains its own local **Factor Graph** to optimize its trajectory in real-time.

The core optimization engine utilizes a sampling-based inference method (similar to **MPPI** - Model Predictive Path Integral), allowing the system to handle non-convex constraints—such as dynamic collision avoidance—without requiring gradient differentiability.

## Result

<div style="text-align: center; margin: 20px 0;">
  <img src="/src\simulation_result_center_obs.gif" alt="Center Obs" width="80%" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
  <br>
  <em style="color: gray; font-size: 0.9em;">Agents plan their paths by avoiding major obstacles and negotiating with other agents.</em>
</div>

---

<div style="text-align: center; margin: 20px 0;">
  <img src="/src\simulation_result_multi_obs.gif" alt="Multi Obs" width="80%" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
  <br>
  <em style="color: gray; font-size: 0.9em;">Agents plan their paths by avoiding a number of obstacles and negotiating with other agents.</em>
</div>

## Method

### 1. Problem Formulation: Factor Graph

The trajectory of an agent $i$ over a horizon $T$ is defined as a sequence of state variables:

$$
X^i = \{ \mathbf{x}_0^i, \mathbf{x}_1^i, \dots, \mathbf{x}_T^i \}
$$

where the state vector is $\mathbf{x}_t^i = [p_x, p_y, v_x, v_y]^\top$. The optimal trajectory is found by maximizing the joint probability distribution of the Factor Graph:

$$
P(X) \propto \prod_{t=0}^{T} \underbrace{f_{dyn}(\mathbf{x}_t, \mathbf{x}_{t+1})}_{\text{Kinematics}} \cdot \underbrace{f_{goal}(\mathbf{x}_T)}_{\text{Target}} \cdot \underbrace{f_{obs}(\mathbf{x}_t)}_{\text{Static Obs}} \cdot \prod_{j \in \mathcal{N}_i} \underbrace{f_{coll}(\mathbf{x}_t^i, \mathbf{x}_t^j)}_{\text{Dynamic Collision}}
$$

### 2. Optimization: EKI with MPPI-based Messages

Unlike standard Belief Propagation, we employ an iterative **Particle Transport** method. The optimization loop consists of **Factor Updates (Message Generation)** and **Variable Updates (Ensemble Kalman Inversion)**.

#### Step 1: Prior Sampling (Prediction)
Each variable node maintains an ensemble of particles representing the current belief:

$$
\mathbf{x}_t^{(k)} \sim \mathcal{N}(\mu_t, \Sigma_t), \quad k=1 \dots K
$$

#### Step 2: Factor Update via MPPI (Message Generation)
Factors evaluate the particles to generate a "target belief" (message) that satisfies their specific constraints. This is done using MPPI-style importance sampling:

1.  **Cost Evaluation:** Calculate cost $\mathcal{J}^{(k)}$ for each particle (e.g., distance to obstacle).
2.  **Weighting:** Compute importance weights using the softmax function:

$$
w^{(k)} \propto \exp \left( -\frac{1}{\lambda} \mathcal{J}^{(k)} \right)
$$

3.  **Message Construction:** Compute the target mean and covariance preferred by this factor:

    $$
    \mu_{msg} = \sum_{k} w^{(k)} \mathbf{x}_t^{(k)}, \quad \Sigma_{msg} = \sum_{k} w^{(k)} (\mathbf{x}_t^{(k)} - \mu_{msg})(\mathbf{x}_t^{(k)} - \mu_{msg})^\top
    $$

#### Step 3: Variable Update via EKI (Particle Transport)
The variable node aggregates messages from multiple factors and updates its particles. Instead of simply replacing the distribution, we use **Ensemble Kalman Inversion** to *transport* particles toward the optimal posterior:

$$
\mathbf{x}_t^{(k)} \leftarrow \mathbf{x}_t^{(k)} + \alpha \cdot \mathbf{K} \cdot (\mu_{msg} - \mathbf{x}_t^{(k)}) + \text{Noise}
$$

Here, $\mathbf{K}$ acts as a Kalman Gain that balances the current belief uncertainty with the factor's target certainty. This allows the trajectory to smoothly converge to a solution that satisfies all constraints.

### 3. Factors Implementation Details

#### A. Goal Factor (`GoalSampleFNode`)
Penalizes the Euclidean distance between the final state and the target position to ensure convergence.

$$
\mathcal{J}_{goal}(\mathbf{x}_T) = \| \mathbf{p}_T - \mathbf{p}_{goal} \|^2 + \gamma \| \mathbf{v}_T \|^2
$$

#### B. Obstacle Factor (`ObstacleSampleFNode`)
Uses the Signed Distance Field (SDF) from the map. It imposes a high penalty for collisions and a decaying potential field for safe proximity.

$$
\mathcal{J}_{obs}(\mathbf{x}_t) = \begin{cases} 
\infty & \text{if } \text{SDF}(\mathbf{p}_t) < r_{safe} \quad \text{(Collision)} \\ 
\alpha \cdot \exp\left( -\beta (\text{SDF}(\mathbf{p}_t) - r_{safe}) \right) & \text{otherwise}
\end{cases}
$$

#### C. Neighbor Collision Factor (`DistSampleFNode`)
Prevents collisions with other agents using Continuous Collision Detection (Point-to-Segment distance $d$). It combines a hard constraint for safety and a soft repulsion for smooth avoidance.

$$
\mathcal{J}_{coll}(\mathbf{x}_t) = \begin{cases} 
\infty & \text{if } d < d_{safe} \quad \text{(Hard Constraint)} \\ 
20 \cdot \exp\left( -(d - d_{safe}) \right) & \text{if } d_{safe} \le d < 2d_{safe} \quad \text{(Soft Repulsion)} \\
0 & \text{otherwise}
\end{cases}
$$

#### D. Kinematics Factor (`KinematicsFNode`)
Enforces physical feasibility by minimizing deviations from the dynamic model (e.g., Constant Velocity). It effectively penalizes unrealistic accelerations.

$$
\mathcal{J}_{dyn}(\mathbf{x}_t, \mathbf{x}_{t+1}) = \| \mathbf{v}_{t+1} - \mathbf{v}_t \|^2_{\Sigma_{acc}^{-1}} \quad \text{subject to } \mathbf{p}_{t+1} = \mathbf{p}_t + \mathbf{v}_{t+1} \Delta t
$$

### 4. Distributed Architecture

The system operates in a fully distributed manner:

1.  **Topology Update:** Agents detect neighbors within a communication radius ($R_{comm}$) and dynamically attach/detach **Collision Factors**.
2.  **Information Sharing:** Agents broadcast only the **statistical belief (Mean, Covariance)** of their future trajectory. No raw sensor data or control inputs are shared.
3.  **Local Optimization:** Each agent runs the EKI loop locally, treating neighbor beliefs as read-only external constraints.
4.  **Receding Horizon:** The system executes the first step of the optimized trajectory and replans in the next cycle (MPC).
