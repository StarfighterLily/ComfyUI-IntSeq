A custom node for ComfyUI that takes integer sequences and creates sigmas for sampling, or images for masks or encoding into latents. The On-Line Encyclopedia of Integer Sequences ([OEIS](https://oeis.org/))
inspired the name, being the first place I went to get numbers to plug into these.

---IntSeq Sigmas:
Inspired by the RES4LYF nodepack's Sigmas From Text node, I wrote this one to simplify the process I was using to remap the integer sequence, sort them, reverse them, and specify the number of integers to use from the sequence.

Sequence - takes a comma separated list of numbers
Count - specifies how many numbers to process; 0 or a number exceeding the length of the sequence will use the whole sequence
New_minimum - remaps the sequence to this lowest number
New_maximum - remaps the sequence to this highest number
Reverse - reverses the sequence
Sort_order - sorts the sequence either from high to low or low to high
NOTE: Count happens first; every option after works on the subset

---IntSeq Plotter:
A simple 2d graph for visualizing the sequence.

Sequence - hopefully you know already
Title - specify a name for the graph, a header label
Xlabel - specify an X axis label
Ylabel - specify a Y axis label

---IntSeq Image:
Create an image from a sequence. Useful for generating masks, various img2img generation, or to encode into a latent.

Width - specify the width of the image
Height - specify the height of the image
Sequence - really shouldn't need to tell you at this point ;)
Method - how the sequence should be visualized
Color_offset - determines the color of the pixel in RGB
Value_min - remaps the sequence to this lowest number
Value_max - remaps the sequence to this highest number
Red_min - as value, for red
Red_max - as value, for red
Green_min - as value, for green
Green_max - as value, for green
Blue_min - as value, for blue
Blue_max - as value, for blue
Angle_scale - determines how much angle should be applied for Angle/Length or Run/Turn method
Length_scale - determines how much length should be applied for Angle/Length or Run/Turn method
Line_width - determines how wide the line should be applied for Angle/Length or Run/Turn method
Start_x - specify the horizontal starting point for all modes but RGB/grayscale
Start_y - specify the verical starting point for all modes but RGB/grayscale
Boundary_behavior - how to deal with drawing off the image edges
-Clamp - hard boundary, nothing offscreen
-Wrap - offscreen draing wraps around to the other side
-Bounce - bounce off the edge back into the imagebounds
-None - no boundary

---IntSeq Wave:
Generate a sequence for various waves, to be plugged into any string-receiving node or text interface.

Type - the type of wave to create
Length - how many values to populate a sequence with
Amplitude - highest value to use
Frequency - how many cycles to make
Phase_offset - phase offset in degrees, moves the wave along X axis
Vertical_offset - vertical offset, moves the wave along Y axis
Slope - slope of the sigmoid wave
Duty cycle - duty cycle of the sawtooth, triangle, and square waves
