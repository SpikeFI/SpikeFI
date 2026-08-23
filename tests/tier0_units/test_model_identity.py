"""Tier 0 — FaultModel's identity contract: equal models hash equal (even
with Tensor arguments), and models with different semantics never compare
equal despite sharing target/method/args.
"""


from copy import deepcopy

import pytest
import torch

from spikefi.fault import Fault, FaultModel, FaultRound, FaultSite, FaultTarget
from spikefi.models import (
    DeadNeuron, DeadSynapse, ParametricNeuron, set_value, StuckNeuron,
    StuckSynapse, ThresholdFaultNeuron,
)


@pytest.mark.neuron
def test_equal_models_hash_equal() -> None:
    """a == b implies hash(a) == hash(b), the contract Python's set/dict
    machinery (and FaultRound, which keys on models) relies on."""
    a, b = DeadNeuron(), DeadNeuron()
    assert a == b
    assert hash(a) == hash(b)


@pytest.mark.synapse
def test_equal_models_hash_equal_with_0dim_tensor_arg() -> None:
    """The identity contract holds when an argument is a 0-dim Tensor:
    _hashable converts it to a Python scalar before hashing."""
    a = FaultModel(FaultTarget.WEIGHT, set_value, torch.tensor(0.5))
    b = FaultModel(FaultTarget.WEIGHT, set_value, torch.tensor(0.5))
    assert a == b
    assert hash(a) == hash(b)


@pytest.mark.synapse
def test_equal_models_hash_equal_with_ndim_tensor_arg() -> None:
    """The identity contract holds when an argument is an N-dim Tensor:
    _hashable converts it to a nested list, then to a hashable tuple."""
    a = FaultModel(FaultTarget.WEIGHT, set_value, torch.tensor([0.1, 0.2]))
    b = FaultModel(FaultTarget.WEIGHT, set_value, torch.tensor([0.1, 0.2]))
    assert a == b
    assert hash(a) == hash(b)


@pytest.mark.synapse
def test_equal_models_hash_equal_with_nested_list_arg() -> None:
    """The identity contract holds for a plain nested-list argument, the
    same path _hashable takes for a Tensor's .tolist() result."""
    a = FaultModel(FaultTarget.WEIGHT, set_value, [[0.1, 0.2], [0.3, 0.4]])
    b = FaultModel(FaultTarget.WEIGHT, set_value, [[0.1, 0.2], [0.3, 0.4]])
    assert a == b
    assert hash(a) == hash(b)


@pytest.mark.synapse
def test_extract_finds_fault_by_deepcopied_tensor_arg_model() -> None:
    """A Fault inserted with a Tensor-arg model is found by extract() using
    a deepcopy of that model, proving the hash survives the deepcopy, not
    just object identity."""
    model = FaultModel(FaultTarget.WEIGHT, set_value, torch.tensor([0.1, 0.2]))
    site = FaultSite('SF1', (0, 0, 0, 0))
    round = FaultRound()
    round.insert(Fault(model, site))

    round.extract(Fault(deepcopy(model), site))

    assert ('SF1', model) not in round


@pytest.mark.neuron
@pytest.mark.synapse
def test_same_target_method_args_different_class_not_equal() -> None:
    """DeadNeuron and StuckNeuron(0.) reduce to the same
    (target, method, args), but must not compare equal: the concrete class
    is part of the identity key."""
    assert DeadNeuron() != StuckNeuron(0.)
    assert DeadSynapse() != StuckSynapse(0.)


@pytest.mark.parametric
def test_parametric_model_identity_contract() -> None:
    """ParametricNeuronFaultModel's own _key (param_name, param_method,
    param_args) carries the same equal-implies-equal-hash guarantee."""
    a = ThresholdFaultNeuron(1.5)
    b = ThresholdFaultNeuron(1.5)
    assert a == b
    assert hash(a) == hash(b)


@pytest.mark.parametric
def test_parametric_model_different_param_name_not_equal() -> None:
    """Two parametric models of the *same* class perturbing different
    parameters must not compare equal, even with identical percentages --
    using the same class isolates param_name itself in the identity key,
    rather than incidentally relying on type(self) to tell them apart."""
    assert ParametricNeuron('tauSr', 1.5) != ParametricNeuron('tauRef', 1.5)
