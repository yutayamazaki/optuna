from __future__ import absolute_import

import gc
import warnings

from optuna.logging import get_logger
from optuna.storages import InMemoryStorage
from optuna.storages import RDBStorage
from optuna.structs import TrialPruned
from optuna.trial import BaseTrial
from optuna import types

if types.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Callable  # NOQA
    from typing import Dict  # NOQA
    from typing import Optional  # NOQA
    from typing import Sequence  # NOQA
    from typing import Tuple  # NOQA
    from typing import Type  # NOQA
    from typing import TypeVar  # NOQA
    from typing import Union  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA
    from optuna.study import Study  # NOQA
    from optuna.trial import Trial  # NOQA

    T = TypeVar('T', float, str)

try:
    from mpi4py.MPI import Comm  # NOQA
    _available = True
except ImportError as e:
    _import_error = e
    _available = False


class _MPIObjectiveFunc(object):
    """A wrapper of an objective function to incorporate Optuna with MPI.

    Note that this class is not supposed to be used by library users.

    Args:
        func:
            A callable that implements objective function.
        comm:
            An MPI communicator.
    """

    def __init__(self, func, comm):
        # type: (Callable[[MPITrial, Comm], float], Comm) -> None

        self.comm = comm
        self.objective = func

    def __call__(self, trial):
        # type: (Trial) -> float

        self.comm.bcast(True)
        return self.objective(MPITrial(trial, self.comm), self.comm)


class MPIStudy(object):
    """A wrapper of :class:`~optuna.study.Study` to incorporate Optuna with MPI.

    .. seealso::
        :class:`~optuna.integration.mpi.MPIStudy` provides the same interface as
        :class:`~optuna.study.Study`. Please refer to :class:`optuna.study.Study` for further
        details.

    Example:

        Optimize an objective function that trains neural network written with MPI.

        .. code::

            comm = mpi4py.MPI.COMM_WORLD
            study = optuna.Study(study_name, storage_url)
            mpi_study = optuna.integration.MPIStudy(study, comm)
            mpi_study.optimize(objective, n_trials=25)

    Args:
        study:
            A :class:`~optuna.study.Study` object.
        comm:
            An MPI communicator.
    """

    def __init__(
            self,
            study,  # type: Study
            comm,  # type: Comm
    ):
        # type: (...) -> None

        if isinstance(study.storage, InMemoryStorage):
            raise ValueError('MPI integration is not available with InMemoryStorage.')

        if isinstance(study.storage, RDBStorage):
            if study.storage.engine.dialect.name == 'sqlite':
                logger = get_logger(__name__)
                logger.warning('SQLite may cause synchronization problems when used with '
                               'MPI integration. Please use other DBs like PostgreSQL.')

        study_names = comm.allgather(study.study_name)
        if len(set(study_names)) != 1:
            raise ValueError('Please make sure an identical study name is shared among workers.')

        super(MPIStudy, self).__setattr__('delegate', study)
        super(MPIStudy, self).__setattr__('comm', comm)

    def optimize(
            self,
            func,  # type: Callable[[MPITrial, Comm], float]
            n_trials=None,  # type: Optional[int]
            timeout=None,  # type: Optional[float]
            catch=(Exception, ),  # type: Union[Tuple[()], Tuple[Type[Exception]]]
    ):
        # type: (...) -> None
        """Optimize an objective function.

        This method provides the same interface as :func:`optuna.study.Study.optimize` except
        the absence of ``n_jobs`` argument.
        """

        if self.comm.rank == 0:
            func_mn = _MPIObjectiveFunc(func, self.comm)
            self.delegate.optimize(func_mn, n_trials=n_trials, timeout=timeout, catch=catch)
            self.comm.bcast(False)
        else:
            while True:
                has_next_trial = self.comm.bcast(None)
                if not has_next_trial:
                    break
                try:
                    func(MPITrial(None, self.comm), self.comm)

                    # We assume that if a node raises an exception,
                    # all other nodes will do the same.
                    #
                    # The responsibility to handle acceptable exceptions (i.e., `TrialPruned` and
                    # `catch`) is in the rank-0 node, so other nodes simply ignore them.
                except TrialPruned:
                    pass
                except catch:
                    pass
                finally:
                    # The following line mitigates memory problems that can be occurred in some
                    # environments (e.g., services that use computing containers such as CircleCI).
                    # Please refer to the following PR for further details:
                    # https://github.com/pfnet/optuna/pull/325.
                    gc.collect()

    def __getattr__(self, attr_name):
        # type: (str) -> Any

        return getattr(self.delegate, attr_name)

    def __setattr__(self, attr_name, value):
        # type: (str, Any) -> None

        setattr(self.delegate, attr_name, value)


class MPITrial(BaseTrial):
    """A wrapper of :class:`~optuna.trial.Trial` to incorporate Optuna with mpi4py.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` object if the caller is rank0 worker,
            :obj:`None` otherwise.
        comm:
            A mpi4py communicator.
    """

    def __init__(self, trial, comm):
        # type: (Optional[Trial], Comm) -> None

        self.delegate = trial
        self.comm = comm

    def suggest_uniform(self, name, low, high):
        # type: (str, float, float) -> float

        def func():
            # type: () -> float

            assert self.delegate is not None
            return self.delegate.suggest_uniform(name, low, high)

        return self._call_with_mpi(func)

    def suggest_loguniform(self, name, low, high):
        # type: (str, float, float) -> float

        def func():
            # type: () -> float

            assert self.delegate is not None
            return self.delegate.suggest_loguniform(name, low, high)

        return self._call_with_mpi(func)

    def suggest_discrete_uniform(self, name, low, high, q):
        # type: (str, float, float, float) -> float

        def func():
            # type: () -> float

            assert self.delegate is not None
            return self.delegate.suggest_discrete_uniform(name, low, high, q)

        return self._call_with_mpi(func)

    def suggest_int(self, name, low, high):
        # type: (str, int, int) -> int

        def func():
            # type: () -> int

            assert self.delegate is not None
            return self.delegate.suggest_int(name, low, high)

        return self._call_with_mpi(func)

    def suggest_categorical(self, name, choices):
        # type: (str, Sequence[T]) -> T

        def func():
            # type: () -> T

            assert self.delegate is not None
            return self.delegate.suggest_categorical(name, choices)

        return self._call_with_mpi(func)

    def report(self, value, step=None):
        # type: (float, Optional[int]) -> None

        if self.comm.rank == 0:
            assert self.delegate is not None
            self.delegate.report(value, step)
        self.comm.barrier()

    def should_prune(self, step=None):
        # type: (Optional[int]) -> bool

        def func():
            # type: () -> bool

            assert self.delegate is not None
            return self.delegate.should_prune(step)

        return self._call_with_mpi(func)

    def set_user_attr(self, key, value):
        # type: (str, Any) -> None

        if self.comm.rank == 0:
            assert self.delegate is not None
            self.delegate.set_user_attr(key, value)
        self.comm.barrier()

    def set_system_attr(self, key, value):
        # type: (str, Any) -> None

        if self.comm.rank == 0:
            assert self.delegate is not None
            self.delegate.set_system_attr(key, value)
        self.comm.barrier()

    @property
    def number(self):
        # type: () -> int

        def func():
            # type: () -> int

            assert self.delegate is not None
            return self.delegate.number

        return self._call_with_mpi(func)

    @property
    def trial_id(self):
        # type: () -> int

        warnings.warn(
            'The use of `MPITrial.trial_id` is deprecated. '
            'Please use `MPITrial.number` instead.', DeprecationWarning)
        return self._trial_id

    @property
    def _trial_id(self):
        # type: () -> int

        def func():
            # type: () -> int

            assert self.delegate is not None
            return self.delegate._trial_id

        return self._call_with_mpi(func)

    @property
    def params(self):
        # type: () -> Dict[str, Any]

        def func():
            # type: () -> Dict[str, Any]

            assert self.delegate is not None
            return self.delegate.params

        return self._call_with_mpi(func)

    @property
    def distributions(self):
        # type: () -> Dict[str, BaseDistribution]

        def func():
            # type: () -> Dict[str, BaseDistribution]

            assert self.delegate is not None
            return self.delegate.distributions

        return self._call_with_mpi(func)

    @property
    def user_attrs(self):
        # type: () -> Dict[str, Any]

        def func():
            # type: () -> Dict[str, Any]

            assert self.delegate is not None
            return self.delegate.user_attrs

        return self._call_with_mpi(func)

    @property
    def system_attrs(self):
        # type: () -> Dict[str, Any]

        def func():
            # type: () -> Dict[str, Any]

            assert self.delegate is not None
            return self.delegate.system_attrs

        return self._call_with_mpi(func)

    def _call_with_mpi(self, func):
        # type: (Callable) -> Any

        if self.comm.rank == 0:
            try:
                result = func()
                self.comm.bcast(result)
                return result
            except Exception as e:
                self.comm.bcast(e)
                raise
        else:
            result = self.comm.bcast(None)
            if isinstance(result, Exception):
                raise result
            return result


def _check_mpi4py_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            'mpi4py is not available. Please install mpi4py to use this feature. '
            'mpi4py can be installed by executing `$ pip install mpi4py`. '
            'For further information, please refer to the installation guide of mpi4py. '
            '(The actual import error is as follows: ' + str(_import_error) + ')')
