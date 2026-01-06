# WilsonCollins2019TenSimpleRules.jl

This repository is a Julia reimplementation of the logics and results on the following paper:

Wilson, R. C., & Collins, A. G. (2019). Ten simple rules for the computational modeling of behavioral data. eLife, 8, e49547. https://doi.org/10.7554/eLife.49547

There is already the authors' MATLAB version: 

https://github.com/AnneCollins/TenSimpleRulesModeling

The extent to which I would refer to their MATLAB codes is undecided at the current time.

## about this code

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> WilsonCollins2019TenSimpleRules.jl

It is authored by TT.

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:

   ```sh
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:

```julia
using DrWatson
@quickactivate "WilsonCollins2019TenSimpleRules.jl"
```

which auto-activate the project and enable local path handling from DrWatson.
