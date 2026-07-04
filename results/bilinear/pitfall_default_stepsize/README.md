# Pitfall: default stepsize on bilinear's degenerate spectrum

bilinear uses B=[[1]], so m=M=1. The vanilla baseline (h=1/sqrt(M)=1.0) and the
Chebyshev/slingshot schedule (which collapses to the same constant magnitude when
m=M) both pick step magnitude 1.0 -- far above the position-step's adaptive clip
cap (min(1, 0.1/a_i) <= 1). Both modes instantly saturate the clip and slam
particles to the domain boundary within a handful of steps, then freeze there.
Vanilla and slingshot look identical here -- but for the wrong reason
(clip-domination, not the sign schedule). See ../calibrated/ for the run that
stays below clip saturation and actually isolates slingshot's effect.
