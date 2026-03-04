Kelly Criterion — Optimal Growth and Monte Carlo Simulation

This project studies the Kelly criterion, a classical result in probability, information theory, and quantitative finance that determines the optimal fraction of capital to allocate in order to maximize long-run logarithmic wealth growth.

The project combines:

• theoretical derivation of optimal bet sizing
• growth-optimal portfolio interpretation
• admissibility constraints and fractional Kelly strategies
• Monte Carlo validation of wealth dynamics

⸻

Optimal Growth Objective

Consider repeated investments with return R. Wealth evolves as

W_{t+1} = W_t (1 + fR)

The Kelly strategy maximizes expected logarithmic growth

g(f) = \mathbb{E}[\log(1 + fR)]

Maximizing this objective yields a strategy that is asymptotically optimal for repeated independent bets.

⸻

Classical Kelly Fraction

For a binary gamble

R =
\begin{cases}
+b & \text{with probability } p \\
-a & \text{with probability } 1-p
\end{cases}

the optimal allocation is

f^* = \frac{pb - (1-p)a}{ab}

This fraction balances expected edge against downside risk, producing the maximal exponential wealth growth rate.

⸻

Admissibility Condition

The Kelly strategy requires

1 + fR > 0

otherwise wealth may reach zero and the logarithmic objective becomes undefined.

For the binary model this implies

f < \frac{1}{a}

⸻

Fractional Kelly

In practice, probabilities and return distributions must be estimated from data.
To reduce estimation risk, practitioners often use fractional Kelly

f_\lambda = \lambda f^*, \quad 0 < \lambda < 1

Common choices:

• Half Kelly
• Quarter Kelly

These strategies reduce drawdowns while preserving most of the growth rate.

⸻

Multi-Asset Kelly Portfolio

For return vector R and portfolio weights f

W_{t+1} = W_t (1 + f^\top R)

Using a small-return approximation

\mathbb{E}[\log(1 + f^\top R)]
\approx
f^\top \mu - \frac{1}{2} f^\top \Sigma f

Maximization yields

f^* = \Sigma^{-1}\mu

This solution is identical to the mean-variance optimal portfolio scaled for log-utility growth, linking the Kelly criterion to Markowitz portfolio theory.

⸻

Monte Carlo Simulation

We simulate repeated wealth trajectories under different bet sizes.

Example parameters
p = 0.55
b = 1
a = 1
T = 10,000 rounds
M = 5,000 simulations

Example outcomes

Strategy	Mean Log Growth	Large Drawdown Frequency
Full Kelly	0.0103	65%
Half Kelly	0.0081	35%
Quarter Kelly	0.0046	18%

Observation:

• Full Kelly maximizes growth
• Fractional Kelly significantly reduces drawdown risk

Repository Structure

kelly-criterion-simulation/

├── report.pdf        # Full mathematical derivation
├── report.tex        # LaTeX source
├── simulation.py     # Monte Carlo experiment
└── README.md

Simulation

The simulation script

• generates repeated wealth paths
• compares Full, Half, and Quarter Kelly strategies
• estimates empirical growth rates
• measures drawdown frequencies

Run with
python simulation.py

References

Kelly — A New Interpretation of Information Rate
Thorp — Optimal Gambling Systems
Cover & Thomas — Elements of Information Theory

⸻

Author

Aryan Khan
Drexel University
