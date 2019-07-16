from __future__ import absolute_import

from mpi4py.MPI import Comm  # NOQA
from typing import Any  # NOQA
from typing import Callable  # NOQA
from typing import Optional  # NOQA
from typing import Tuple  # NOQA
from typing import Type  # NOQA
from typing import Union  # NOQA

from optuna.logging import get_logger
from optuna.pruners import BasePruner  # NOQA
from optuna.storages import BaseStorage  # NOQA
from optuna.storages import InMemoryStorage
from optuna.storages import RDBStorage
from optuna.structs import TrialPruned
from optuna.study import Study  # NOQA
from optuna.trial import Trial  # NOQA


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
        # type: (Callable[[Trial, Comm], float], Comm) -> None

        self.comm = comm
        self.objective = func

    def __call__(self, trial):
        # type: (Trial) -> float

        self.comm.bcast((True, trial.trial_id))
        return self.objective(trial, self.comm)


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

        study.pruner = _MPIPruner(pruner=study.pruner, comm=comm)
        super(MPIStudy, self).__setattr__('delegate', study)
        super(MPIStudy, self).__setattr__('comm', comm)

    def optimize(
            self,
            func,  # type: Callable[[Trial, Comm], float]
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
            self.comm.bcast((False, None))
        else:
            while True:
                has_next_trial, trial_id = self.comm.bcast(None)
                if not has_next_trial:
                    break
                trial = Trial(self.delegate, trial_id)
                try:
                    func(trial, self.comm)

                    # We assume that if a node raises an exception,
                    # all other nodes will do the same.
                    #
                    # The responsibility to handle acceptable exceptions (i.e., `TrialPruned` and
                    # `catch`) is in the rank-0 node, so other nodes simply ignore them.
                except TrialPruned:
                    pass
                except catch:
                    pass

    def __getattr__(self, attr_name):
        # type: (str) -> Any

        return getattr(self.delegate, attr_name)

    def __setattr__(self, attr_name, value):
        # type: (str, Any) -> None

        setattr(self.delegate, attr_name, value)


class _MPIPruner(BasePruner):
    def __init__(self, pruner, comm):
        # type: (BasePruner, Comm) -> None

        self.delegate = pruner
        self.comm = comm

    def prune(self, storage, study_id, trial_id, step):
        # type: (BaseStorage, int, int, int) -> bool

        if self.comm.rank == 0:
            try:
                result = self.delegate.prune(storage, study_id, trial_id, step)
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
