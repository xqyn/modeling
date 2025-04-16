from manim import *

class CircleAnimation(Scene):
    def construct(self):
        # Create a circle with radius 2
        circle = Circle(radius=2, color=BLUE)
        
        # Create a dot to trace the circle's circumference
        dot = Dot(point=circle.point_from_proportion(0), color=RED)
        
        # Create an angle text to show degrees
        angle_text = always_redraw(
            lambda: Text(f"{int(self.t * 360)}°").move_to(UP * 3)
        )
        
        # Add circle and initial dot to scene
        self.add(circle, dot, angle_text)
        
        # Track time/angle progression
        self.t = 0
        
        # Animation function to update dot position
        def update_dot(dt):
            self.t += dt / 4  # Control speed (4 seconds for full circle)
            if self.t > 1:
                self.t -= 1
            dot.move_to(circle.point_from_proportion(self.t))
        
        # Add updater to dot
        dot.add_updater(update_dot)
        
        # Play animation for 4 seconds (one full circle)
        self.wait(4)
        
        # Remove updater to stop animation
        dot.remove_updater(update_dot)