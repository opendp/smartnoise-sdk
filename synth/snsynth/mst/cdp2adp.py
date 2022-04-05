"""
   Copyright 2020 (https://github.com/IBM/discrete-gaussian-differential-privacy)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

# Code for computing approximate differential privacy guarantees
# for discrete Gaussian and, more generally, concentrated DP
# See https://arxiv.org/abs/2004.00010
# - Thomas Steinke dgauss@thomas-steinke.net 2020

import math

# *********************************************************************
# Now we move on to concentrated DP

# compute delta such that
# rho-CDP implies (eps,delta)-DP
# Note that adding cts or discrete N(0,sigma2) to sens-1 gives rho=1/(2*sigma2)


# start with standard P[privloss>eps] bound via markov
def cdp_delta_standard(rho, eps):
    assert rho >= 0
    assert eps >= 0
    if rho == 0:
        return 0  # degenerate case
    # https://arxiv.org/pdf/1605.02065.pdf#page=15
    return math.exp(-((eps - rho)**2) / (4 * rho))


# Our new bound:
# https://arxiv.org/pdf/2004.00010v3.pdf#page=13
def cdp_delta(rho, eps):
    assert rho >= 0
    assert eps >= 0
    if rho == 0:
        return 0  # degenerate case

    # search for best alpha
    # Note that any alpha in (1,infty) yields a valid upper bound on delta
    # Thus if this search is slightly "incorrect" it will only result in larger delta (still valid)
    # This code has two "hacks".
    # First the binary search is run for a pre-specificed length.
    # 1000 iterations should be sufficient to converge to a good solution.
    # Second we set a minimum value of alpha to avoid numerical stability issues.
    # Note that the optimal alpha is at least (1+eps/rho)/2. Thus we only hit this constraint
    # when eps<=rho or close to it. This is not an interesting parameter regime, as you will
    # inherently get large delta in this regime.

    amin = 1.01  # don't let alpha be too small, due to numerical stability
    amax = (eps + 1) / (2 * rho) + 2
    for i in range(1000):  # should be enough iterations
        alpha = (amin + amax) / 2
        derivative = (2 * alpha - 1) * rho - eps + math.log1p(-1.0 / alpha)
        if derivative < 0:
            amin = alpha
        else:
            amax = alpha

    # now calculate delta
    delta = math.exp((alpha - 1) * (alpha * rho - eps) + alpha * math.log1p(-1 / alpha)) / (alpha - 1.0)
    return min(delta, 1.0)  # delta<=1 always


# Above we compute delta given rho and eps, now we compute eps instead
# That is we wish to compute the smallest eps such that rho-CDP implies (eps,delta)-DP
def cdp_eps(rho, delta):
    assert rho >= 0
    assert delta > 0
    if delta >= 1 or rho == 0:
        return 0.0  # if delta>=1 or rho=0 then anything goes
    epsmin = 0.0  # maintain cdp_delta(rho,eps)>=delta
    epsmax = rho + 2 * math.sqrt(rho * math.log(1 / delta))  # maintain cdp_delta(rho,eps)<=delta

    # to compute epsmax we use the standard bound
    for i in range(1000):
        eps = (epsmin + epsmax) / 2
        if cdp_delta(rho, eps) <= delta:
            epsmax = eps
        else:
            epsmin = eps
    return epsmax


# Now we compute rho
# Given (eps,delta) find the smallest rho such that rho-CDP implies (eps,delta)-DP
def cdp_rho(eps, delta):
    assert eps >= 0
    assert delta > 0
    if delta >= 1:
        return 0.0  # if delta>=1 anything goes
    rhomin = 0.0  # maintain cdp_delta(rho,eps)<=delta
    rhomax = eps + 1  # maintain cdp_delta(rhomax,eps)>delta
    for i in range(1000):
        rho = (rhomin + rhomax) / 2
        if cdp_delta(rho, eps) <= delta:
            rhomin = rho
        else:
            rhomax = rho
    return rhomin
