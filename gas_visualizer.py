
"""
Matplotlib-based visualizer for a two-dimensional ideal-gas simulation.

The visualizer handles the graphical display and animation. The user
supplies the particle positions and a callback function that advances
the simulation by one time step.

Example
-------
    import numpy as np
    from gas_visualizer import GasVisualizer

    # Initial positions and velocities for three particles.
    r = np.array([
        [-10.0, 0.0],
        [0.0, 10.0],
        [10.0, 0.0],
    ])
    v = np.array([
        [10.0, 5.0],
        [-5.0, 10.0],
        [0.0, -10.0],
    ])
    dt = 0.05

    def update_simulation():
        # Advance particle positions by one time step.
        r[:] += v * dt

        # Add wall and particle collision handling here.

    visualizer = GasVisualizer(box_size=100, radius=2)
    visualizer.run(r, update_simulation)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle


class GasVisualizer:
    """Animate particles moving inside a two-dimensional square box.

    The box is centered at the origin, with x and y coordinates ranging
    from -box_size / 2 to +box_size / 2.

    This class is responsible only for visualization. The user supplies
    a callback that updates the particle positions and implements the
    desired simulation physics.

    Parameters
    ----------
    box_size : float, default=100.0
        Length of each side of the square simulation box.

    radius : float, default=2.0
        Displayed radius of each particle, in the same coordinate units
        as the particle positions.

    steps_per_frame : int, default=1
        Number of simulation updates performed before displaying each
        animation frame. Larger values advance the simulation faster
        relative to the displayed frame rate.

    interval : int, default=10
        Delay between animation frames, in milliseconds.

    particle_color : str, default="blue"
        Matplotlib color specification used for particle faces and
        outlines. Examples include "blue", "red", and "#336699".

    show_axes : bool, default=True
        If True, display the coordinate axes and labels. If False,
        hide the axes and labels.

    num_frames : int, default=None
            If specified, the number of shown frames before simulation 
            terminates. 

    Raises
    ------
    ValueError
        If the box size, radius, steps per frame, or interval is invalid.
    """

    def __init__(
        self,
        box_size=100.0,
        radius=2.0,
        steps_per_frame=2,
        interval=10,
        particle_color="blue",
        show_axes=False,
        num_frames=None,
    ):
        if box_size <= 0:
            raise ValueError("box_size must be positive.")
        if radius <= 0 or radius >= box_size / 2:
            raise ValueError(
                "radius must be positive and smaller than half the box size."
            )
        if not isinstance(steps_per_frame, int) or steps_per_frame < 1:
            raise ValueError("steps_per_frame must be a positive integer.")
        if interval <= 0:
            raise ValueError("interval must be positive.")
        if num_frames is not None and (
            not isinstance(num_frames, int)
            or isinstance(num_frames, bool)
            or num_frames < 1
        ):
            raise ValueError("num_frames must be a positive integer or None.")


        self.box_size = float(box_size)
        self.radius = float(radius)
        self.steps_per_frame = steps_per_frame
        self.interval = int(interval)
        self.particle_color = particle_color
        self.show_axes = bool(show_axes)
        self.num_frames = num_frames

        print(
                f"GasVisualizer initialized: "
                f"box_size={self.box_size}, "
                f"radius={self.radius}, "
                f"steps_per_frame={self.steps_per_frame}, ",
                f"interval={self.interval}",
            )

    def run(self, positions, update_simulation, filled=None):
        """Display an animated simulation of the particles.

        Parameters
        ---------- 
        positions : numpy.ndarray, shape (N, 2)
            Initial particle coordinates. Each row contains the x and y
            coordinates of one particle.

            The array is retained by reference for subsequent animation
            updates. The callback must therefore update its contents
            in place rather than reassigning the array to a new object.

        update_simulation : callable
            Function called to advance the simulation by one time step.
            It takes no arguments and should update the particle
            positions in place.

            The visualizer calls this function steps_per_frame times
            before displaying each frame.

        filled : array-like of bool, shape (N,), optional
            Determines whether each particle is drawn as a filled or
            hollow circle. True produces a filled circle; False produces
            a hollow circle. If omitted, all particles are filled.

        Returns
        -------
        matplotlib.animation.FuncAnimation
            The animation object. Keeping a reference to this object
            helps ensure that the animation remains active.

        Raises
        ------
        ValueError
            If positions does not have shape (N, 2), contains non-finite
            values, or filled has an incompatible shape.

        TypeError
            If positions is not numeric or update_simulation is not
            callable.

        Notes
        -----
        Particle positions are displayed in the coordinate system of
        the simulation box. This method does not implement physical
        behavior such as motion, collisions, or boundary conditions;
        these belong in update_simulation.
        """
        
        if not isinstance(positions, np.ndarray):
            raise TypeError("positions must be a NumPy array.")
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions must have shape (N, 2).")
        if not np.issubdtype(positions.dtype, np.number):
            raise TypeError("positions must contain numeric values.")
        if not np.all(np.isfinite(positions)):
            raise ValueError("positions must contain only finite values.")
        if not callable(update_simulation):
            raise TypeError("update_simulation must be callable.")

        n = len(positions)

        if filled is None:
            filled = np.ones(n, dtype=bool)
        else:
            filled = np.asarray(filled, dtype=bool)
            if filled.shape != (n,):
                raise ValueError(
                    "filled must have one Boolean value per particle."
                )

        # Set up the figure and square simulation box.
        half = self.box_size / 2
        fig, ax = plt.subplots()

        ax.add_patch(
            Rectangle(
                (-half, -half),
                self.box_size,
                self.box_size,
                fill=False,
                edgecolor="black",
                linewidth=1.5,
            )
        )

        # Leave enough space for particles whose centers approach the walls.
        ax.set_xlim(-half - self.radius, half + self.radius)
        ax.set_ylim(-half - self.radius, half + self.radius)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Ideal Gas Simulator")

        if self.show_axes:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        else:
            ax.set_axis_off()

        # Create one circle patch for each particle.
        particles = []
        for i, position in enumerate(positions):
            circle = Circle(
                position,
                radius=self.radius,
                facecolor=self.particle_color if filled[i] else "none",
                edgecolor=self.particle_color,
                linewidth=1.5,
            )
            ax.add_patch(circle)
            particles.append(circle)

        def animate(_frame):
            """Advance the simulation and update the displayed particles."""
            # Perform the requested number of simulation steps.
            for _ in range(self.steps_per_frame):
                update_simulation()

            # Move each displayed circle to its updated position.
            for circle, position in zip(particles, positions):
                circle.center = position

            # Return the artists that Matplotlib should redraw.
            return particles

        # Create the animation. Frame-data caching is disabled because
        # the simulation advances continuously rather than replaying
        # a stored sequence of frames.
        animation = FuncAnimation(
            fig,
            animate,
            frames=self.num_frames,
            repeat=False,
            interval=self.interval,
            blit=False,
            cache_frame_data=False,
        )

        plt.show(block=True)
        return animation
    
GasVisualizer()