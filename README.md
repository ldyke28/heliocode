# Radiation pressure labels
no rp - mu = 0

p7 rp - mu = .7

s2 rp - mu = sin^2(2pi/11yrs * t)

p5s2 rp - mu = .5 + sin^2(2pi/11yrs * t)


# Tracking process

Initial conditions are sampled from a distribution. The initial condition on x (positive x being the upstream direction) is large, in the case of my testing 1000 au. Since we are assuming, at least for the moment, a 2D plane of orbit, the only other spatial position that varies is y. There is a span of initial conditions on y, which are sampled with a series of initial conditions on the velocity component v_x (v_y and v_z are assumed to be zero to adhere with the center of the Maxwellian distribution) centered around v_x=-26 km/s to produce a series of orbits with varying initial conditions. All of the orbits start from t=0 and go to some t greater than the time it would take to pass by the sun, which in my case is generally around t=7e9 seconds.

These trajectories, once fully solved using the exact differential equation solver odeint (from the scipy.interpolate package), are then probed to determine if there are any points in the orbit in which the orbit comes within a characteristic distance from the point of interest from which we want to trace particles backward (I’ve generally been using a radius from the point of interest of .01 au).

# List generation

Once these points in the orbits are identified, the phase space coordinates (position and velocity components) are then stored, along with the time point at which the particle passes this point, which is important for orbits calculated under the assumption of time dependent radiation pressure. The velocity components (specifically v_x and v_y) are then printed in a list, along with the time of passing, to give an idea of what the average velocity components of orbits passing the point of interest at a specific point in time happen to be.

# Backtracing

Once we know what average velocities are occurring corresponding to given orbits passing at certain points in time, we can set the starting time to be one of the times within the list and center the initial conditions for the velocity components around the average for each component of the velocity from that point in time. For example: 

![alt text](https://github.com/ldyke28/heliocode/blob/d0d1aca802099abc3c1eb7b318f2655d79a67b65/extra/examplelist.png)
(taken from p5s2 rp list)

Here, we see these orbits are passing the point of interest at around t=5.53575e9 seconds. We can use v_x ~= -34.5 km/s and v_y ~= -1.1 km/s as the center of our velocity components and generate initial conditions within an equal range around both of these points to trace back. These orbits are traced backward until it is certain they have passed through the plane defined by x = 100 au, and the components of velocity are recorded when they pass through this plane.

When we want to examine the transformation of the Maxwellian at the point of interest, we can trace the orbits back, select a region that defines the Maxwellian out of the velocity components in the x=100 au plane, and record the velocity conditions at the point of interest that correspond to velocities that are within the Maxwellian at x=100 au, as well as the time of travel for these points and the corresponding value of the distribution function.

# Working with the code

I have the code set up to run in three modes: 1 allows you to generate the list using forward tracing, 2 maps an individual orbit and graphs it, and 3 performs the backtracing and plotting of the Maxwellian. You still need to provide initial conditions yourself and designate which radiation pressure you want to use (I’ve written functions for mu = 0, .7, sin^2(2pi/11yrs * t), and .5 + sin^2(2pi/11yrs * t) so far), but the code does the rest for you. Mode is designated through the variable “mode” at the top and only takes 1, 2, and 3 (it will accept other things, but they won’t actually do anything).
