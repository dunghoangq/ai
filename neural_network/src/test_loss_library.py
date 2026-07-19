import numpy as np

from loss_library import BCELoss, CrossEntropyLoss, HuberLoss, MAELoss, MSELoss


def test_regression_values_and_gradients():
    prediction = np.array([3.0, 1.0])
    target = np.array([1.0, 2.0])

    assert np.isclose(MSELoss()(prediction, target), 2.5)
    np.testing.assert_allclose(MSELoss().backward(prediction, target), [2.0, -1.0])
    assert np.isclose(MAELoss()(prediction, target), 1.5)
    np.testing.assert_allclose(MAELoss().backward(prediction, target), [0.5, -0.5])
    assert np.isclose(HuberLoss(delta=1.0)(prediction, target), 1.0)
    np.testing.assert_allclose(HuberLoss(delta=1.0).backward(prediction, target), [0.5, -0.5])


def test_bce_gradient():
    loss = BCELoss()
    prediction = np.array([0.8, 0.2])
    target = np.array([1.0, 0.0])
    assert np.isclose(loss(prediction, target), -np.log(0.8))
    np.testing.assert_allclose(loss.backward(prediction, target), [-0.625, 0.625])


def test_cross_entropy_is_stable_and_has_softmax_minus_target_gradient():
    logits = np.array([[1000.0, 999.0, 998.0], [1.0, 2.0, 3.0]])
    target = np.array([0, 2])
    loss = CrossEntropyLoss()
    value = loss(logits, target)
    gradient = loss.backward(logits, target)

    assert np.isfinite(value)
    assert gradient.shape == logits.shape
    np.testing.assert_allclose(np.sum(gradient, axis=1), [0.0, 0.0], atol=1e-12)

