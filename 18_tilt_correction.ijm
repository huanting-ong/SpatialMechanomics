// 18_tilt_correction.ijm
//
// Tilt correction for post-nanoindentation confocal stacks.
// Brings the tissue surface parallel to the imaging plane so that the
// 12 um projection used for registration is measured perpendicular to the
// surface rather than along the optical axis.
//
// USAGE
//   1. Open the stack in Fiji.
//   2. Run the first Reslice below to view the stack from the left.
//   3. Select the line tool and draw along the tissue surface.
//   4. Read the angle from Analyze > Measure, then enter it in the
//      Rotate line below. THE VALUE BELOW IS AN EXAMPLE, NOT A CONSTANT:
//      it is measured separately for every stack and edited before running.
//   5. Run the Rotate and the second Reslice.
//
// output=0.982 is the voxel depth in um and must match the acquired stack.

run("Reslice [/]...", "output=0.982 start=Left");

//setTool("line");                 // measure the surface tilt here

run("Rotate... ", "angle=0.5 grid=1 interpolation=Bilinear stack");   // <-- per-stack value

run("Reslice [/]...", "output=0.982 start=Top");
