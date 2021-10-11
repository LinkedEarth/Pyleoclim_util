1. In you test_pca_t3, do not reimport pyleoclim as a package. It's heavy enough.
A: that was an oversight ; now fixed.

1. Parameters in the documentation of MultipleSeries.pca do not match the actual parameters in the function (I assume that standardize is part of pca_kwargs)
A: well spotted; now fixed.

1. Does SSAres actually have a name that we can use for the new class (like PSD o scalogram)
A: I'm not aware of any name for something that stores the output of an SSA method, so I made one up. Would you rather it be called SSA, in analogy with PSD? I was trying to use a different name from the method itself to avoid confusion. Incidentally, the relevant name for PCA is SpatialDecomp, which is designed to hold results from PCA, MSSA, and maybe other spatial methods if we ever get there.

1. screeplot arguments are not all defined in the docstring
A: Indeed, one was missing.  Now added.

1. Why only allow for MTM when we have other spectral functions?
A: All our decomposition methods assume evenly spaced time grids, and MTM is usually the best choice for those. I generalized to allow any method to be invoked here, with MTM as the default.

1. Mode: I understand the python indexing challenge but we should really start at 1 and do the 0-indexing internally
A: Right now, all graphics for decomposition classes (e.g. SSARes, SpatialDecomp) use 1-based indices, but API and code use 0-based indexing. It would create an odd precedent to use 1-based indices internally for those classes but not the others.  
