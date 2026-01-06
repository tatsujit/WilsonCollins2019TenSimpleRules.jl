################################################################
# 両者であんまり実行速度に差がなさそうだった
################################################################
"""
sampling により non-sleeping arms を決定する
"""
function sample_available_arms!(sys::System)
    # randomly choose n_arms from distributions
    a_arms = sample(sys.rng, 1:sys.env.n_arms, sys.env.n_avail_arms, replace=false)
    sys.env.available_arms = a_arms # [i in a_arms for i in 1:sys.env.n_arms]
end
"""
shuffle して先頭からとることで non-sleeping arms を決定する
"""
function shuffle_available_arms!(sys::System)
    # randomly choose n_arms from distributions by shuffling
    arms = 1:sys.env.n_arms
    shuffled_arms = shuffle(sys.rng, arms)
    sys.env.available_arms = shuffled_arms[1:sys.env.n_avail_arms]
end
