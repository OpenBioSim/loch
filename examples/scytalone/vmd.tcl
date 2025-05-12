# Specify the lighting.
light 0 on
light 1 on
light 2 off
light 3 off

# Disable the stage and axes.
axes location off
stage location off

# Use orthographic projection.
display projection orthographic

# Set the rendering method for each molecule.
mol modstyle 0 0 {NewCartoon}
mol modstyle 0 1 {CPK 1.0 0.4 32 16}
mol modstyle 0 2 {CPK 1.0 0.0 32 16}
mol modcolor 0 2 {Occupancy}

# Turn off depth cue.
display depthcue off

# Set the view.
rotate y by -90

# Scale the view.
scale by 0.1
