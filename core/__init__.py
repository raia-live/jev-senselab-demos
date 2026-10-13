"""Shared harness for the Jev + SenseLab demos.

Every demo under ``demos/`` is a :class:`core.scenario.Scenario`: a stream of business
cases, a written policy, a hidden rule the policy gets wrong, a change half-way through,
and a review that reports how each decision turned out. The arms in :mod:`core.arms`
decide every case with the same Jev model; they differ only in what fills the memory
slot of Jev's state.
"""
