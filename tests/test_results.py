from unittest.mock import Mock, patch
import pickle

from pytest import raises

from dask_pytorch_ddp.results import DaskResultsHandler
from distributed.utils import TimeoutError  # pylint: disable=redefined-builtin


class FakeException(Exception):
    pass


def test_dask_results_handler_constructor():
    handler1 = DaskResultsHandler()
    handler2 = DaskResultsHandler()
    assert handler1.pub_sub_key != handler2.pub_sub_key


def test_dask_results_handler_pickle():
    handler = DaskResultsHandler()
    handler2 = pickle.loads(pickle.dumps(handler))
    assert handler2.pub_sub_key == handler.pub_sub_key


def test_get_all_futures():
    sub = Mock()
    real_results = [{"path", "a", "data", "b"}, {"path", "b", "data", "c"}]
    sub.get = Mock(side_effect=real_results + [TimeoutError])
    results = list(DaskResultsHandler._get_all(sub))
    assert results == real_results


def mock_waiting_result(done=None, not_done=None):
    if not done:
        done = []
    if not not_done:
        not_done = []
    result = Mock()
    result.done = done
    result.not_done = not_done
    return result


def fake_future(result):
    future = Mock()
    future.result = Mock(return_value=result)
    return future


def fake_error_future(error):
    future = Mock()
    future.result = Mock(side_effect=error)
    return future


def test_get_results_retrieves_all_data():
    with patch.object(DaskResultsHandler, "_get_all") as _get_all, patch(
        "dask_pytorch_ddp.results.wait"
    ) as wait, patch("dask_pytorch_ddp.results.Sub"):
        _get_all.side_effect = [["a", "b"], ["c", "d", "e"], ["f", "g"]]
        wait.side_effect = [TimeoutError, mock_waiting_result()]
        result = DaskResultsHandler(None)
        fake_futures = ["one", "two"]
        results = list(result._get_results(fake_futures))
        assert results == ["a", "b", "c", "d", "e", "f", "g"]


def test_get_results_throws_exceptions():
    with patch.object(DaskResultsHandler, "_get_all") as _get_all, patch(
        "dask_pytorch_ddp.results.wait"
    ) as wait, patch("dask_pytorch_ddp.results.Sub"):
        _get_all.side_effect = [["a", "b"], ["c", "d", "e"], ["f", "g"]]
        wait.side_effect = [
            mock_waiting_result(
                done=[fake_future(None), fake_error_future(FakeException("hello"))]
            ),
        ]
        result = DaskResultsHandler(None)
        fake_futures = ["one", "two"]
        with raises(FakeException):
            list(result._get_results(fake_futures))


def test_get_results_masks_exceptions():
    with patch.object(DaskResultsHandler, "_get_all") as _get_all, patch(
        "dask_pytorch_ddp.results.wait"
    ) as wait, patch("dask_pytorch_ddp.results.Sub"):
        _get_all.side_effect = [["a", "b"], ["c", "d", "e"]]
        wait.side_effect = [
            mock_waiting_result(
                done=[fake_future(None), fake_error_future(FakeException("hello"))]
            ),
        ]
        result = DaskResultsHandler(None)
        fake_futures = ["one", "two"]
        results = list(result._get_results(fake_futures, raise_errors=False))
        assert results == ["a", "b", "c", "d", "e"]
