# Powerflow

The powerflow problem is about the calculation of voltage magnitudes and angles for all network nodes.
The solution is obtained from a subset of voltages and power injections.

## Power System Model
Power systems are modeled as a network of nodes (buses) and branches (lines and transformers).
Power sources (generators) and sinks (loads) can be connected to the nodes.
Each node in the network is fully described by the following four electrical quantities:

* \\(\vert V_k \vert\\): voltage magnitude
* \\(\theta_k\\): voltage phase angle
* \\(P_k\\): active power
* \\(Q_k\\): reactive power

There are three types of network nodes: VD, PV and PQ.
Depending on the node type, two of the four electrical quantities are specified.

| Node Type	| Known		                      | Unknown                         |
|        ---|                              ---|                              ---|
| \\(VD\\)  | \\(\vert V_k \vert, \theta_k\\) | \\(P_k, Q_k\\)                  |
| \\(PV\\)  | \\(P_k, \vert V_k \vert\\)      | \\(Q_k, \theta_k\\)             |
| \\(PQ\\)  | \\(P_k, Q_k\\)                  | \\(\vert V_k \vert, \theta_k\\) |


## Newton Raphson

The goal is to bring a nonlinear mismatch function \\(f\\) to zero.
The value of the mismatch function depends on a solution vector \\(x\\):

\\[ f(x) = 0 \\]

As \\(f(x)\\) is nonlinear, the equation system is solved iteratively using Newton-Raphson:

\\[ x_{i+1} = x_i + \Delta x_i = x_i - \textbf{J}_f(x_i)^{-1} f(x_i) \\]

where \\(\Delta x\\) is the correction of the solution vector and \\(\textbf{J}_f\\) is the Jacobian matrix.

Instead of computing \\(\Delta x_i = - \textbf{J}_f(x_i)^{-1} f(x_i)\\), the linear equation set 

\\[ - \textbf{J}_f(x_i) \Delta x_i = f(x_i) \\]

is solved for \\(\Delta x_i\\).

Iterations are stopped when the mismatch is sufficiently small:

\\[ f(x_i) < \epsilon \\]


## Powerflow Solution

The solution vector \\(x\\) represents the voltage \\(V\\) either in polar coordinates

\\[ \left [ \begin{array}{c} \delta \\ \vert V \vert \end{array} \right ] \\]

or rectangular coordinates

\\[ \left [ \begin{array}{c} V_{real} \\ V_{imag} \end{array} \right ] \\]

The mismatch function \\(f\\) represents the power mismatch

\\[ \Delta S = \left [ \begin{array}{c} \Delta P \\ \Delta Q \end{array} \right ] \\]


or the current mismatch

\\[ \Delta I = \left [ \begin{array}{c} \Delta I_{real} \\ \Delta I_{imag} \end{array} \right ] \\]


This results in four different formulations of the powerflow problem:

* power mismatch function and polar coordinates
* power mismatch function and rectangular coordinates
* current mismatch function and polar coordinates
* current mismatch function and rectangular coordinates

To solve the problem using Newton-Raphson, we need to formulate \\(\textbf{J}_f\\) and \\(f\\) for each powerflow problem formulation.


### Powerflow with Power Mismatch Function and Polar Coordinates

The injected power at a node \\(k\\) is given by

\\[ S_k = V_k I_k^* \\]

The current injection into any node \\(k\\) is

\\[ I_k = \sum_{j=1}^N Y_{kj} V_j \\]

Substitution yields

\\[
\begin{align*}
S_k &= V_k \left ( \sum_{j=1}^N Y_{kj} V_j \right )^* \\\\
    &= V_k \sum_{j=1}^N Y_{kj}^* V_j^*
\end{align*}
\\]

\\(G_{kj}\\) and \\(B_{kj}\\) are defined as the real and imaginary part of the admittance matrix element \\(Y_{kj}\\), so that \\(Y_{kj} = G_{kj} + jB_{kj}\\).
This results in

\\[
\begin{align*}
S_k &= V_k \sum_{j=1}^N Y_{kj}^* V_j^* \\\\
    &= \vert V_k \vert \angle \theta_k \sum_{j=1}^N (G_{kj} + jB_{kj})^* ( \vert V_j \vert \angle \theta_j)^* \\\\
    &= \vert V_k \vert \angle \theta_k \sum_{j=1}^N (G_{kj} - jB_{kj}) ( \vert V_j \vert \angle - \theta_j) \\\\
    &= \sum_{j=1}^N \left \vert V_k \vert \vert V_j \vert \angle (\theta_k - \theta_j) \right (G_{kj} - jB_{kj}) \\\\
    &= \sum_{j=1}^N \vert V_k \vert \vert V_j \vert \left ( cos(\theta_k - \theta_j) + jsin(\theta_k - \theta_j) \right ) (G_{kj} - jB_{kj})
\end{align*}
\\]

If we perform the algebraic multiplication of the two terms inside the parentheses, and collect real and imaginary parts, and recall that \\(S_k = P_k + jQ_k\\), we can split this into two equations: one for the real part, and one for the imaginary part.

\\[
\theta_{kj} = \theta_k - \theta_j \\\\
P_k = \sum_{j=1}^N \vert V_k \vert \vert V_j \vert \left ( G_{kj}cos(\theta_{kj}) + B_{kj} sin(\theta_{kj}) \right ) \\\\
Q_k = \sum_{j=1}^N \vert V_k \vert \vert V_j \vert \left ( G_{kj}sin(\theta_{kj}) - B_{kj} cos(\theta_{kj}) \right )
\\]

These are called the power flow equations.

We consider a power system network having \\(N\\) buses. We assume one VD bus, \\(N_{PV}-1\\) PV buses and \\(N-N_{PV}\\) PQ buses.
We assume that the VD bus is numbered bus \\(1\\), the PV buses are numbered \\(2,...,N_{PV}\\), and the PQ buses are numbered \\(N_{PV}+1,...,N\\).
We define the vector of unknowns as the composite vector of unknown angles \\(\theta\\) and voltage magnitudes \\(\vert V \vert\\):

\\[
x = \left[ \begin{array}{c} \theta \\\\ \vert V \vert \\\\ \end{array} \right ]
  = \left[ \begin{array}{c} \theta_2 \\\\ \theta_{3} \\\\ \vdots \\\\ \theta_N \\\\ \vert V_{N_{PV+1}} \vert \\\\ \vert V_{N_{PV+2}} \vert \\\\ \vdots \\\\ \vert V_N \vert \end{array} \right]
\\]

The right-hand sides of the powerflow equations for \\(P_k\\) and \\(Q_k\\) depend on the elements of the unknown vector \\(x\\).

Expressing this dependency more explicitly, we rewrite these equations as

\\[
\begin{align*}
P_k^{spec} = P_k^{calc} (x) \Rightarrow  P_k^{calc} (x) - P_k^{spec} &= 0 \quad \quad k = 2,...,N \\\\
Q_k^{spec} = Q_k^{calc} (x) \Rightarrow  Q_k^{calc} (x) - Q_k^{spec} &= 0 \quad \quad k = N_{PV}+1,...,N
\end{align*}
\\]

We define the mismatch \\({f} (x)\\) as

\\[
\begin{align*}
f(x) = \left [ \begin{array}{c} f_1(x) \\\\ \vdots \\\\ f_{N-1}(x) \\\\ ------ \\\\ f_N(x) \\\\ \vdots \\\\ f_{2N-N_{PV} -1}(x) \end{array} \right ]
     = \left [ \begin{array}{c} P_2(x) - P_2 \\\\ \vdots \\\\ P_N(x) - P_N \\\\ --------- \\\\ Q_{N_{PV}+1}(x) - Q_{N_{PV}+1} \\\\ \vdots \\\\ Q_N(x) - Q_N \end{array} \right]
     = \left [ \begin{array}{c} \Delta P_2 \\\\ \vdots \\\\ \Delta P_N \\\\ ------ \\\\ \Delta Q_{N_{PV}+1} \\\\ \vdots \\\\ \Delta Q_N \end{array} \right ]
     = 0
\end{align*}
\\]

That is a system of nonlinear equations.
The nonlinearity stems from the fact that \\(P_k\\) and \\(Q_k\\) have terms containing products of unknowns and also terms containing trigonometric functions of unknowns.


The Jacobian matrix is obtained by taking all first-order partial derivates of the power mismatch function with respect to the voltage angles \\(\theta_k\\) and magnitudes \\(\vert V_k \vert\\):

\\[
\theta_{jk} = \theta_j - \theta_k \\\\
\begin{align*}
J_{jk}^{P \theta} &= \frac{\partial P_j (x ) } {\partial \theta_k} = \vert V_j \vert \vert V_k \vert \left ( G_{jk} sin(\theta_{jk}) - 																B_{jk} cos(\theta_{jk} ) \right ) \\\\
J_{jj}^{P \theta} &= \frac{\partial P_j(x)}{\partial \theta_j} = -Q_j (x ) - B_{jj} \vert V_j \vert ^{2} \\\\
J_{jk}^{Q \theta} &= \frac{\partial Q_j(x)}{\partial \theta_k} = - \vert V_j \vert \vert V_k \vert \left ( G_{jk} cos(\theta_{jk}) + 																B_{jk} sin(\theta_{jk}) \right ) \\\\
    J_{jj}^{Q \theta} &= \frac{\partial Q_j(x)}{\partial \theta_k} = P_j (x ) - G_{jj} \vert V_j \vert ^{2} \\\\
    J_{jk}^{PV} &= \frac{\partial P_j (x ) } {\partial \vert V_k \vert } = \vert V_j \vert \left ( G_{jk} cos(\theta_{jk}) + 																B_{jk} sin(\theta_{jk}) \right ) \\\\
    J_{jj}^{PV} &= \frac{\partial P_j(x)}{\partial \vert V_j \vert } = \frac{P_j (x )}{\vert V_j \vert} + G_{jj} \vert V_j \vert \\\\
    J_{jk}^{QV} &= \frac{\partial Q_j (x ) } {\partial \vert V_k \vert } = \vert V_j \vert \left ( G_{jk} sin(\theta_{jk}) - B_{jk} cos(\theta_{jk}) \right ) \\\\
    J_{jj}^{QV} &= \frac{\partial Q_j(x)}{\partial \vert V_j \vert } = \frac{Q_j (x )}{\vert V_j \vert} - B_{jj} \vert V_j \vert \\\\
\end{align*}
\\]

The linear system of equations that is solved in every Newton iteration can be written in matrix form as follows

\\[
\begin{align*}
-J(x) \left [ \begin{array}{c} \Delta \theta \\\\ \Delta \vert V \vert \end{array} \right ] &= \left [ \begin{array}{c} \Delta P \\\\ \Delta Q \end{array} \right ] \\\\
\Rightarrow J(x) \left [ \begin{array}{c} \Delta \theta \\\\ \Delta \vert V \vert \end{array} \right ] &= \left [ \begin{array}{c} -\Delta P \\\\ -\Delta Q \end{array} \right ]
\end{align*}
\\]

\\[
\begin{align*}
\left [ \begin{array}{cccccc} 
    \frac{\partial \Delta P_2 }{\partial \theta_2} & \cdots & \frac{\partial \Delta P_2 }{\partial \theta_N} &
    \frac{\partial \Delta P_2 }{\partial \vert V_{N_{G+1}} \vert} & \cdots & \frac{\partial \Delta P_2 }{\partial \vert V_N \vert} \\\\
    \vdots & \ddots & \vdots & \vdots & \ddots & \vdots	\\\\
    \frac{\partial \Delta P_N }{\partial \theta_2} & \cdots & \frac{\partial \Delta P_N}{\partial \theta_N} &
    \frac{\partial \Delta P_N}{\partial \vert V_{N_{G+1}} \vert } & \cdots & \frac{\partial \Delta P_N}{\partial \vert V_N \vert} \\\\
    \frac{\partial \Delta Q_{N_{G+1}} }{\partial \theta_2} & \cdots & \frac{\partial \Delta Q_{N_{G+1}} }{\partial \theta_N} &
    \frac{\partial \Delta Q_{N_{G+1}} }{\partial \vert V_{N_{G+1}} \vert } & \cdots & \frac{\partial \Delta Q_{N_{G+1}} }{\partial \vert V_N \vert}	\\\\
    \vdots & \ddots & \vdots & \vdots & \ddots & \vdots	\\\\
    \frac{\partial \Delta Q_N}{\partial \theta_2} & \cdots & \frac{\partial \Delta Q_N}{\partial \theta_N} &
    \frac{\partial \Delta Q_N}{\partial \vert V_{N_{G+1}} \vert } & \cdots & \frac{\partial \Delta Q_N}{\partial \vert V_N \vert}
\end{array} \right ]
\left [ \begin{array}{c} \Delta \theta_2 \\\\ \vdots \\\\ \Delta \theta_N \\\\ \Delta \vert V_{N_{G+1}} \vert \\\\ \vdots \\\\ \Delta \vert V_N \vert \end{array} \right ]
= \left [ \begin{array}{c} -\Delta P_2 \\\\ \vdots \\\\ -\Delta P_N \\\\ -\Delta Q_{N_{G+1}} \\\\ \vdots \\\\ -\Delta Q_N \end{array} \right ]
\end{align*}
\\]

### Solution Steps

1. Set the iteration counter to \\(i=1\\). Use the initial solution \\(V_{i} = 1 \angle 0^{\circ}\\)
2. Compute the mismatch vector \\(f({x_i})\\) using the power flow equations
3. Check the stopping criterion
	* If \\(\vert \Delta P_{i} \vert < \epsilon_{P}\\) for all type PQ and PV buses and
	* If \\(\vert \Delta Q_{i} \vert < \epsilon_{Q}\\) for all type PQ
	* Then go to step 6
	* Else, go to step 4
4. Evaluate the Jacobian matrix \\(\textbf{J}_f(x_i)\\) and compute \\(\Delta x_i\\).
5. Compute the new solution vector \\(x_{i+1}\\) and return to step 3.
6. Stop.