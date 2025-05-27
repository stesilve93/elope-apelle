# elope

In a nutshell
Estimate velocities during lunar descents from event streams. Additionally, a rangemeter measurement is provided as well as data from the Inertial Measurment Unit (IMU).

Background
In 2021, the DAVIS240 sensor became the first neuromorphic device to be launched into space and remains operational aboard the International Space Station (ISS), having been delivered as part of the Falcon Neuro Project. This milestone reflects a broader and growing interest in neuromorphic sensing and computing within the aerospace community, which has gained significant momentum over the past decade [1]. Despite this growing enthusiasm, the systematic evaluation and quantification of the benefits offered by event-based cameras in space applications are still in their early stages.

Several promising applications have been proposed for event based cameras, including star tracking [6], particle detection and tracking [3], spacecraft pose estimation [7], and autonomous landing guidance [3–4]. These proposals leverage the unique advantages of event-based vision sensors, particularly their high dynamic range, low latency, and excellent temporal resolution—characteristics that make them especially suited for the demanding and dynamic conditions encountered in space.

Event-streams from the Moon
The ELOPE challenge is envisaged as the first of a series of challenges moving towards more and more realistic and challenging event streams. With reference to an earlier version of our dataset production pipeline, tailored at purely synthetic streams [2], the dataset we curated for ELOPE simulates optimal landings of a spacecraft on the challenging South Pole region of the Moon. There is a growing interest in landing near the Moon’s South Pole because of its incredible potential for future exploration and sustained human presence. Unlike most of the lunar surface, some areas near the South Pole are in near-constant sunlight, which is ideal for solar power generation. Even more importantly, permanently shadowed regions in nearby craters are believed to contain water ice — a critical resource for life support, fuel production, and more. Access to this ice could make long-term lunar missions far more feasible, turning the South Pole into a strategic hub for both science and future space infrastructure.


Example of a PANGU generated image.
The landings provided in the ELOPE dataset correspond to different light conditions and landing sites close to the Malapert crater. The dataset was created by using digital elevation models of this area, which remain undisclosed and are thus not available for solving the challenge. The only sources of information during the decent are the event streams, telemetry from a simulated IMU and the readings of a rangemeter. The exact descent profiles were found applying optimal control theory to a deterministically modelled 6DOF lunar landing module with the objective to minimize propellant consumption. Be careful though as some trajectories might include corrective maneuvers, reflecting the possibility of future landers to select the safest landing spot autonomously.

The event streams were simulated using the Planet and Asteroid Natural Scene Generation Utility (PANGU) developed by the University of Dundee’s Space Technology Centre  and the realistic dynamic vision sensor event camera data synthesis from frame-based video v2e tool, developed by the Sensors Group in ETH Zurich.

 

Landing Geometry
The lunar lander in this challenge is equipped with a low-resolution event-based camera mounted on the lunar module. The 3D points of the rugged lunar surface map to a 2D image plane according to the camera geometry. The ego-motion of the lander results in an uninterrupted stream of events in this imaging plane.

Since the event stream alone does not provide information about scale or absolute distances, a rangemeter is used to measure the distance corresponding to the central pixel of the imaging plane.

This figure shows you the reference frames and quantities involved to describe the generic event-based landing of the spacecraft. The camera frame as well as the inertial frame used are visualized. In green, the rangemeter measurement is given.

Check out the data page for a precise description of the provided information.

