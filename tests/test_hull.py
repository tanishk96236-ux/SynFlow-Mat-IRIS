

def test_calibration_error_is_scatter_and_not_a_standard_error():
    """A claim worth pinning, because a feature was built on its opposite.

    `calibration_error` is the standard deviation of the residuals about the
    fitted correction. It estimates how far one more phase would sit from the
    fit, which is the right uncertainty for one more candidate -- and it
    converges to a fixed value rather than averaging down with sample size.

    Measured on the real thing: KRhO2 re-judged on 40 calibration phases
    instead of 10 went from +0.0311 +/- 0.131 to +0.0387 +/- 0.112. Four
    times the data, 15% less spread. A standard error would have halved.

    So a candidate closer to the hull than this number cannot be resolved by
    calibrating harder, and refine_hull.py says so in its own docstring.

    Demonstrated without a regression on purpose: the claim is about a
    standard deviation against a standard error, which needs no fitting --
    and numpy's lstsq aborts inside MKL on the development machine, so a test
    that used it could not run here at all.
    """
    import statistics

    import numpy as np

    rng = np.random.default_rng(0)
    sigma = 0.15

    def scatter(n):
        return statistics.pstdev(rng.normal(0, sigma, size=n).tolist())

    small, large = scatter(10), scatter(640)

    # 64x the data. A standard error would fall 8-fold; the scatter does not
    # move, because it is estimating sigma rather than averaging it away.
    assert abs(large - sigma) < 0.02, large
    assert abs(small - large) < 0.09, (small, large)
    assert large > sigma / 2, (
        "the spread must not collapse with sample size -- that is the whole "
        "reason a tighter calibration cannot rescue a borderline candidate")


def test_the_calibration_sample_is_representative_not_the_smallest():
    """A larger sample must widen the draw, not change the population.

    calibrate used to sort by cell size and keep the 2*sample smallest, then
    sample inside that. So `sample` chose the population: at 10 the fit saw
    the 20 smallest phases in the chemical system, at 30 the 60 smallest. The
    smallest phases are simple binaries where ORB's error is smaller and more
    uniform than on the complex phases a candidate competes against, so the
    uncertainty came out too tight. Measured on perovskites:

        YTaO3    +0.562 +/- 0.006   ->   +0.594 +/- 0.115
        LaHfO3   +0.258 +/- 0.010   ->   +0.269 +/- 0.067

    An uncertainty that grows nineteenfold when given more data was not
    converging, it was wrong.
    """
    import inspect

    from synflow import hull

    source = inspect.getsource(hull.calibrate)

    # the size-ranked truncation must be gone from the main path
    assert "usable[:max(sample * 2, sample)]" not in source
    # a cost cap is fine; a ranking is not
    assert "max_atoms" in source
    assert "affordable" in source
    # and the sample must be drawn from everything affordable
    assert "random.Random(seed).sample(affordable" in source


def test_the_pool_size_is_reported():
    """So a thin chemical system looks thin instead of looking precise. When
    n_calibration equals calibration_pool the fit used everything there was,
    and asking for a larger sample cannot help."""
    import inspect

    from synflow import hull

    source = inspect.getsource(hull.calibrate)
    assert '"calibration_pool"' in source
    assert '"n_calibration"' in source
