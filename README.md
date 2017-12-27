Model Predictive Controller (MPC) Project
==========

Summary
-------

The goal of this project is to write a C++ program that uses waypoints to drive a car around a track in the Udacity CarND simulator. The outputs from the Model Predictive Controller (MPC) are the steering angle and accelerator values. To make the task more realistic, there is an artificial latency added of 100 ms that we have to handle. This version allows the car to mostly max out it's speed at 100mph, only breaking when neccessary to complete a turn.

The Model
---------

Note: These coordinates are all in world-space, not car-local space as we'll see later.

To be able to predict a near-optimal path for the car to follow, we first get some way-points from the simulator:

* ptsx, ptsy - coordinates in of the next few way-points.

We also get some details about the car in world space.

* px, py - the car's coordinates.
* psi - the car's current heading (in radians).
* v - the car's velocity (in miles per hour).
* steering_angle - the car's steering angle (in radians).
* throttle - the car's current throttle (between +1 with full acceleration and -1 if full-breaking)

### Converting to Car-local coordinates
Since we need to get the steering angle and plot points in car-local coordinates, we convert everything. First we convert the waypoints using the `convertPt()` function.

```C++
std::vector<double> convertPt(double global_x, double global_y, double px, double py, double psi) {
  // Convert world space coordinates into car space coordinates.
  double vehicle_x = (global_x - px) * cos(psi) + (global_y - py) * sin(psi);
  double vehicle_y = -(global_x - px) * sin(psi) + (global_y - py) * cos(psi);
  return {vehicle_x, vehicle_y};
}
```

Converting the car model coordinates is simple because the position is always `(0,0)` and straight ahead is `0` deg and the velocity is the same.

### Fitting a line

Next, we use the waypoints (now in car-local coordinates) to fit a line. Using the way-point line's coefficients, we can then calculate our current cross-track error and heading error (epsi).

```C++
// Get the CTE as the y offset at 0.
// Note: the CTE in car space is relative to the car, so we don't need to
// subtract py from the value returned.
double cte = polyeval(coeffs, 0);
// Similarly, in car space, psi is equal to zero, so it doesnt' need to
// be added to epsi.
double epsi = -atan(coeffs[1]);
```

That completes all the values that we need for the state of the car, but with the addition of latency, we actually need to predict the location where the car will be after the latency has been accounted for.

### Latency

If we didn't account for the 100ms in latency, the model would be predicting values for the actuators too late and would end up going off the road, especially at high speeds. To account for this we implemented the predict() function that takes the current state and the delay from the latency to calculate where the car would be at that much time in the future.

```C++
Eigen::VectorXd predict(Eigen::VectorXd state, double delay,
                        double current_steer, double current_throttle, Eigen::VectorXd coeffs, double Lf) {

  // This discussion was helpful in figuring out how to implement the latency.
  // https://discussions.udacity.com/t/how-to-incorporate-latency-into-the-model/257391/63?u=fcarey

  // current state
  double x0 = state[0];
  double y0 = state[1];
  double psi0 = state[2];
  double v0 = state[3];
  // state after delay
  double x = x0 + v0 * cos(psi0) * delay;
  double y = y0 + v0 * sin(psi0) * delay;
  double psi = psi0 - v0 * current_steer * delay / Lf;
  double v = v0; // Assume a constant velocity considering the short delay.
  double epsi = -atan(coeffs[1]) + psi;
  double cte = polyeval(coeffs, 0) - y0 + v0 * sin(epsi) * delay;

  Eigen::VectorXd return_state(6);
  return_state << x, y, psi, v, cte, epsi;

  return return_state;
}
```

Solving for a near-optimal path.
----------

Now that we have everything we need, we can solve for a route that optimizes for the constraints we set.

### Costs

We can prioritize different ways to optimize the same goal by setting the cost function. For each cost we add, we set a multiplier depending on how strongly we want the cost to effect the overall cost. These values were iterated on and tweaked so that the car moves smoothly around the track, but still tried to go as fast as possible.

```C++
// Initial cost function
fg[0] = 0;

// The part of the cost based on the reference state.
for (unsigned int t = 0; t < N; t++) {
        // Cross track error.
  fg[0] += 500 * CppAD::pow(vars[cte_start + t], 2);
        // Direction (psi) error.
  fg[0] += 500 * CppAD::pow(vars[epsi_start + t], 2);
        // Reference velocity error.
  fg[0] += 2. * CppAD::pow(vars[v_start + t] - ref_v, 2);
}

// Minimize the use of actuators.
for (unsigned int t = 0; t < N - 1; t++) {
        // The steering angle.
  fg[0] += 2000 * CppAD::pow(vars[delta_start + t], 2);
        // The change in acceleration.
  fg[0] += 5 * CppAD::pow(vars[a_start + t], 2);
}

// Minimize the value gap between sequential uses of the actuators.
    // This should smooth things out a bit.
for (unsigned int t = 0; t < N - 2; t++) {
        // The difference between the next steering angle and this one.
  fg[0] += 2000 * CppAD::pow(vars[delta_start + t + 1] - vars[delta_start + t], 2);
        // The difference between the next acceleration and this one.
  fg[0] += 10 * CppAD::pow(vars[a_start + t + 1] - vars[a_start + t], 2);
}
```

### Time horizon

We also needed to decide on how long of a time horizon we would consider when optimizing. The time horizon is made up of two components `horizon = N * dt` where:

N - the number of discrete time steps.
dt - the amount of time between time steps.

2 seconds seemed a reasonable horizon. At first, I tried 20 and 0.1 respectively, but it was a little slow. 10 and 0.1 caused the car to crash eventually, but 10 and 0.2 worked well. Out of interest I tried N of 100 and dt of 0.02, but the car immediately turned off the road. 20 and 0.1 worked ok at first, but became unstable with the steering oscillating back and forth and would also crash. The optimizer only has a limited time to try an find a path, so these values seemed to work best.

### Sending actuations back to the simulator

Finally, we get back first actuator settings from near-optimal path: steering angle (turning the steering wheel) and the accelerator/brake.

The steering angle that comes back is in radians, so we divide by 25 degrees (in radians) to map the steering angle to between -1 and +1 that the simulator expects.

### Plotting lines.

The x,y coordinates of the near-optimal path also get returned from `MPC::Solve()` and we us that to plot the green line in front of the car.

The line we fitted from the way-points is also plotted ahead of the car as a yellow line to help visualize where the center of the track is.