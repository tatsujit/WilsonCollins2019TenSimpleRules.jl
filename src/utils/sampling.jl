"""
一様分布 Uniform(lower, upper) からサンプリングする。
ただし lower == upper の場合は定数 lower を返す。
これによって、パラメータを固定値にすることができる。
"""
function uniform_or_constant(lower, upper)
    # lower と upper が同じ場合は定数を返す
    if lower == upper
        return lower
    else
        return rand(Uniform(lower, upper))
    end
end
