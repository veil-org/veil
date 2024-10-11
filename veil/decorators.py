from __future__ import annotations
from typing import Any, Callable, Optional, Tuple
import functools
from typeguard import check_type

import mlflow
from mlflow.entities import Experiment, RunStatus
from mlflow.tracking.fluent import _get_experiment_id, ActiveRun
from mlflow.utils.mlflow_tags import MLFLOW_GIT_COMMIT, MLFLOW_GIT_BRANCH, MLFLOW_GIT_REPO_URL

from veil.types import StringDict, StringList


def _active_experiment_id() -> str:
    return _get_experiment_id()


def _get_repo_info() -> Tuple[str, str, str]:
    import git
    from git.exc import InvalidGitRepositoryError, NoSuchPathError

    repo = None
    repo_uri, sha_commit, branch_name = None, None, None
    try:
        repo = git.Repo(search_parent_directories=True)
    except InvalidGitRepositoryError as e:
        print(f"Invalid Git repository: {e}")
    except NoSuchPathError as e:
        print(f"Invalid Git path: {e}")

    if repo is not None:

        try:
            repo_uri = repo.remotes[0].config_reader.get("url")
            print(f"{MLFLOW_GIT_REPO_URL}={repo_uri}")
        except Exception:
            print(f"{MLFLOW_GIT_REPO_URL} is None")

        try:
            sha_commit = repo.head.object.hexsha
            print(f"{MLFLOW_GIT_COMMIT}={sha_commit}")
        except Exception:
            print(f"{MLFLOW_GIT_COMMIT} is None")

        try:
            branch_name = repo.active_branch.name
            print(f"{MLFLOW_GIT_BRANCH}={branch_name}")
        except Exception:
            print(f"{MLFLOW_GIT_BRANCH} is None")

    return repo_uri, sha_commit, branch_name


class Autologger:
    """ Implements the auto-logging strategy.
    """

    def __init__(
        self,
        is_autolog_enabled: bool = True,
        tracking_uri: str = mlflow.get_tracking_uri(),
        experiment_name: str = Experiment.DEFAULT_EXPERIMENT_NAME,
    ):
        self.is_autolog_enabled = is_autolog_enabled
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

        # members with intended protected access
        self._current_session: Optional[AutologSession] = None

    @property
    def is_autolog_enabled(self) -> bool:
        return self.__is_autolog_enabled

    @is_autolog_enabled.setter
    def is_autolog_enabled(self, value: bool) -> None:
        self.__is_autolog_enabled: bool = check_type(value, bool)

    @property
    def tracking_uri(self) -> str:
        return self.__tracking_uri

    @tracking_uri.setter
    def tracking_uri(self, value: str) -> None:
        self.__tracking_uri: str = check_type(value, str)

    @property
    def experiment_name(self) -> str:
        return self.__experiment_name

    @experiment_name.setter
    def experiment_name(self, value: str) -> None:
        self.__experiment_name: str = check_type(value, str)

    def start_session(
        self,
        name: Optional[str] = None,
        log_tags: StringDict = dict()
    ):
        """Starts a new session.

        Parameters
        ----------
        name : Optional[str], optional
            the experiment name, by default None
        log_tags : StringDict, optional
            the tags to be logged, by default dict()

        Returns
        -------
        AutologSession
            teh autolog session.
        """
        return AutologSession(
            autologger=self,
            name=name,
            log_tags=log_tags
        )

    def run(
        self,
        name: Optional[str] = None,
        log_params: Optional[StringList] = None,
        log_tags: StringDict = dict()
    ):
        """Executes a new run.

        Parameters
        ----------
        name : Optional[str], optional
            the run name, by default None
        log_params : Optional[StringList], optional
            the params to be logged, by default None
        log_tags : StringDict, optional
            the tags to be logged, by default dict()

        Returns
        -------
        Run
            the experiment run.
        """
        return Run(
            autologger=self,
            name=name,
            log_params=log_params,
            log_tags=log_tags
        )


class MlflowIsolated:
    """ Isolates an Mlflow experiment.
    """

    def __init__(
        self,
        autologger: Autologger
    ):
        # members with intended private access
        self.__autologger: Autologger = check_type(autologger, Autologger)

    def __call__(self, func: Callable):
        """
        Execute the decorator as well as the wrapped function
        """
        check_type(func, Callable)

        @functools.wraps(func)
        def isolation_wrapper(*args, **kwargs):
            result: Any = None

            if self.__autologger.is_autolog_enabled and self.__autologger._current_session is not None:

                # 3) switch the run current active run (which is paused) within mlflow with a new one
                past_active_run: ActiveRun = mlflow.active_run()
                if past_active_run:
                    mlflow.end_run(RunStatus.to_string(RunStatus.RUNNING))

                # 1) switch the tracking uri currently used by mlflow to the one in the autologger
                past_tracking_uri: str = mlflow.get_tracking_uri()
                mlflow.set_tracking_uri(self.__autologger.tracking_uri)

                # 2) switch the experiment currently used by mlflow to the one in the autologger
                past_active_experiment_id: str = _active_experiment_id()
                mlflow.set_experiment(
                    experiment_name=self.__autologger.experiment_name)

                # performs the function workload
                result = func(*args, **kwargs)

                # 5) switch back to the previosuly activated experiment
                mlflow.set_experiment(experiment_id=past_active_experiment_id)
                past_active_experiment_id = None

                # 6) switch back to the previously targeted tracking server
                mlflow.set_tracking_uri(past_tracking_uri)
                past_tracking_uri = None

                # 4) switch back to the previously activated run
                if past_active_run:
                    mlflow.start_run(run_id=past_active_run.info.run_id)
                past_active_run = None
            else:
                result = func(*args, **kwargs)

            return result

        return isolation_wrapper


class AutologSession:
    """ Wraps an autolog session.

    Parameters
        ----------
        autologger : Autologger
            the autolog object
        name : Optional[str], optional
            the experiment name, by default None
        log_tags : StringDict, optional
            the tags to be logged, by default dict()
    """

    def __init__(
        self,
        autologger: Autologger,
        name: Optional[str] = None,
        log_tags: StringDict = dict(),
    ):
        self.name = name
        self.log_tags = log_tags

        # members with intended private access
        self.__autologger: Autologger = check_type(autologger, Autologger)
        self.__run_id: Optional[str] = None
        self.__past_session: Optional[AutologSession] = None

    @property
    def autologger(self) -> Autologger:
        return self.__autologger

    @property
    def run_id(self) -> Optional[str]:
        return self.__run_id

    @property
    def name(self) -> Optional[str]:
        return self.__name

    @name.setter
    def name(self, value: Optional[str]) -> None:
        self.__name: Optional[str] = check_type(value, Optional[str])

    @property
    def log_tags(self) -> StringDict:
        return self.__log_tags

    @log_tags.setter
    def log_tags(self, value: StringDict) -> None:
        self.__log_tags: StringDict = check_type(value, StringDict)

    def __enter__(self):
        # switch the session currently used by the autologger to this one
        self.__past_session = self.autologger._current_session
        self.autologger._current_session = self

        @MlflowIsolated(autologger=self.autologger)
        def do_enter():
            # starting a session means managing the context so to:
            if self.autologger.is_autolog_enabled:
                # immediately starts and stops a novel parent run associated with this context
                # note that this run will be resumed within run-annotated functions.
                run: ActiveRun = mlflow.start_run(run_name=self.name)
                mlflow.end_run(status=RunStatus.to_string(RunStatus.RUNNING))
                self.__run_id = run.info.run_id

        do_enter()

    def __exit__(self, exc_type, exc_value, exc_tb):

        @MlflowIsolated(autologger=self.autologger)
        def do_exit():
            # terminating a session means managing the context so to:
            if self.autologger.is_autolog_enabled:

                # immediately starts and stops the parent run associated with this context
                # note that it is terminated with a given status, according to exceptions within the
                # context manager.
                termination_status: RunStatus = RunStatus.FINISHED
                if exc_type:
                    termination_status = RunStatus.FAILED
                mlflow.start_run(
                    run_id=self.autologger._current_session.run_id)
                mlflow.end_run(status=RunStatus.to_string(termination_status))
                self.__run_id = None

        do_exit()

        # switch back the session currently used by the autologger to the previous one
        self.autologger._current_session = self.__past_session
        self.__past_session = None


class Run:
    """ Encapsulates an Mlflow run with auto-logging features.

    Parameters
    ----------
    autologger : Autologger
        the autologger object
    name : Optional[str], optional
        the experiment name, by default None
    log_params : Optional[StringList], optional
        the params to be logged, by default None
    log_tags : StringDict, optional
        the tags to be logged, by default dict()
    """

    def __init__(
        self,
        autologger: Autologger,
        name: Optional[str] = None,
        log_params: Optional[StringList] = None,
        log_tags: StringDict = dict(),
    ):
        # members with intended private access
        self.__autologger: Autologger = check_type(autologger, Autologger)
        self.__name: Optional[str] = check_type(name, Optional[str])
        self.__log_params: Optional[StringList] = check_type(
            log_params, Optional[StringList])
        self.__log_tags: StringDict = check_type(log_tags, StringDict)

    @property
    def autologger(self) -> Autologger:
        return self.__autologger

    @property
    def name(self) -> Optional[str]:
        return self.__name

    @property
    def log_params(self) -> Optional[StringList]:
        return self.__log_params

    @property
    def log_tags(self) -> StringDict:
        return self.__log_tags

    def __call__(self, func: Callable):
        """
        Execute the decorator as well as the wrapped function
        """
        check_type(func, Callable)

        @functools.wraps(func)
        @MlflowIsolated(autologger=self.__autologger)
        def wrapper(*args, **kwargs):
            result: Any = None

            if self.__autologger.is_autolog_enabled and self.__autologger._current_session:

                # resume the parent run
                mlflow.start_run(
                    run_id=self.__autologger._current_session.run_id)

                # retrieves the tags from the context
                tags: StringDict = self.__autologger._current_session.log_tags.copy()

                # ...then overrides them with run-bound tags...
                tags.update(self.__log_tags)

                # ...and eventuallly sets mlflow special tags for .git info
                repo_uri, sha_commit, branch_name = _get_repo_info()
                tags.update({
                    MLFLOW_GIT_REPO_URL: repo_uri,
                    MLFLOW_GIT_COMMIT: sha_commit,
                    MLFLOW_GIT_BRANCH: branch_name,
                })

                # uses the user provided run name instead of function name, if any
                _run_name: str = func.__name__
                if not self.__name is None:
                    _run_name = self.__name

                # starts the child run with a context manager (eventually closing it gracefully in case of exceptions)
                with mlflow.start_run(run_name=_run_name, nested=True) as active_run:

                    # then logs tags and...
                    mlflow.set_tags(tags)

                    # ...the params with which the funciton has been called
                    for k, v in kwargs.items():
                        if self.__log_params is not None and len(self.__log_params) > 0:
                            if k in self.log_params:
                                mlflow.log_param(k, v)
                        else:
                            mlflow.log_param(k, v)

                    # finally the function gets invoked
                    result = func(*args, **kwargs)

                # stops the parent run
                mlflow.end_run()

            else:
                result = func(*args, **kwargs)
            return result

        return wrapper


if __name__ == "__main__":

    import veil
    from veil import run, start_session

    mlflow.set_experiment("Default")
    veil.set_experiment_name(experiment_name="experiment_autolog")

    @run()
    def my_operator():
        print("Hello World!")

    @run()
    def faulty_operator():
        raise ValueError

    with start_session(name="my_parent_faboulous_run"):
        my_operator()

        with start_session(name="my_parent_second_faboulous_run"):
            my_operator()
            mlflow.set_experiment("My Experiment")
            mlflow.start_run(run_name="My Run Name")
            faulty_operator()
            my_operator()

        my_operator()

    my_operator()
