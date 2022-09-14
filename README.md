# List of files and descriptions

UVFsolver - a solver written to employ universal variable formulation to solve for position/velocity information for an orbit at a given point in time

arraytest - used to test manipulations of numpy arrays (not physics code)

diffsolver - the original code written to solve the differential equation for the position/velocity of a particle in Cartesian coordinates given a radiation pressure function that is a function of time, which uses scipy's odeint function for an exact numerical solution

pstraj/pstrajold - uses code from diffsolver to track one or a series of orbits and generate phase space information/trajectory information (see below)

testparticlecode - old code that used second order central differencing to solve equations of motion for a particle orbit

plotfromdata - allows for plotting of graphs from generated data files

extraplotting - written to plot mu and beta

juliadiffeqtesting - used to familiarize myself with solving differential equations in Julia

pstrajjulia - an attempt to see if Julia could handle the trajectory solving process from pstraj

# Radiation pressure labels and related info
no rp - mu = 0

p7 rp - mu = .7

s2 rp - mu = sin^2(2pi/11yrs * t)

p5s2 rp - mu = .5 + sin^2(2pi/11yrs * t)

p5s2adj rp - mu = .5 + sin^2(pi/11yrs * t)

cosexp rp - mu = .75 + .243 * cos(2pi/11yrs * t - pi) * e^(cos(2pi/11yrs * t - pi))

To calculate the radiation pressure at a specific point in time, using time in seconds:

s2 rp: mu = sin^2(2*pi*t/(3.47e8))

p5s2 rp: mu = .5 + sin^2(2*pi*t/(3.47e8))

p5s2adj rp: mu = .5 + sin^2(pi*t/(3.47e8))

cosexp rp: mu = .75 + .243 * cos(2*pi/(3.47e8) * t - pi) * e^(cos(2*pi/(3.47e8) * t - pi))

I chose to work with times around 6e9 seconds because that was the characteristic time scale for orbits to reach the area of interest around the sun when starting at t = 0 at the plane x = 1000 au. Since the sine term oscillates regularly and there isn’t any damping included in the model yet, any other time would work just as well - the choice of times around this regime was arbitrary. I plan to shift this time to be closer to t=0 (specifically with a minimum of mu at t=0) at some point.



# Tracking process

Initial conditions are sampled from a distribution. The initial condition on x (positive x being the upstream direction) is large, in the case of my testing 1000 au. Since we are assuming, at least for the moment, a 2D plane of orbit, the only other spatial position that varies is y. There is a span of initial conditions on y, which are sampled with a series of initial conditions on the velocity component v_x (v_y and v_z are assumed to be zero to adhere with the center of the Maxwellian distribution) centered around v_x=-26 km/s to produce a series of orbits with varying initial conditions. All of the orbits start from t=0 and go to some t greater than the time it would take to pass by the sun, which in my case is generally around t=7e9 seconds.

These trajectories, once fully solved using the exact differential equation solver odeint (from the scipy.interpolate package), are then probed to determine if there are any points in the orbit in which the orbit comes within a characteristic distance from the point of interest from which we want to trace particles backward (I’ve generally been using a radius from the point of interest of .01 au).

# List generation

Once these points in the orbits are identified, the phase space coordinates (position and velocity components) are then stored, along with the time point at which the particle passes this point, which is important for orbits calculated under the assumption of time dependent radiation pressure. The velocity components (specifically v_x and v_y) are then printed in a list, along with the time of passing, to give an idea of what the average velocity components of orbits passing the point of interest at a specific point in time happen to be.

# Backtracing

At the point of me updating this, this is the most relevant capability of the code.

A series of initial conditions are given for the velocity components, as well as a time array that starts at the time at which the trajectories reach the target point and goes backward in time through a sufficient period to capture all relevant trajectories. The trajectories are all inputted into the odeint function to solve for the trajectory data.

When we want to examine the transformation of the Maxwellian at the point of interest, we can trace the orbits back, select a region that defines the Maxwellian out of the velocity components in the x=100 au plane, and record the velocity conditions at the point of interest that correspond to velocities that are within the Maxwellian at x=100 au, as well as the time of travel for these points and the corresponding value of the distribution function.

For these trajectories, we also factor in losses that correspond to the photoionization of the particles, with a photoionization rate defined within the code.

# Working with the code

I have the code set up to run in three modes: 1 allows you to generate the list using forward tracing, 2 maps an individual orbit and graphs it, and 3 performs the backtracing and plotting of the Maxwellian. You still need to provide initial conditions yourself and designate which radiation pressure you want to use (I’ve written functions for all of the radiation pressures above), but the code does the rest for you. Mode is designated through the variable “mode” at the top and only takes 1, 2, and 3 (it will accept other things, but they won’t actually do anything). Be sure, if you are on mode 3, to write the path correctly if you want to save to a file, or else it will throw an error.


If you're reading this and aren't sure what's going on at some point in the code, feel free to email me at lucas.r.dyke.gr@dartmouth.edu with questions.
