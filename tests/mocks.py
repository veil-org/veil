#import pytest
#import veil

######################
# this python module must contains the fixture you want to use in your unit test
#####################

from __future__ import annotations

import time
from mlflow.entities import LifecycleStage, RunStatus
from mlflow.entities.run import Run, RunData, RunInfo
from typing import Any, Dict, Optional
import pytest

from unittest import mock
from mlflow.entities import Experiment
from mlflow.tracking.fluent import ActiveRun


class GitRepo:

    def __init__(self, *args, **kwargs):
        class ConfigReader: pass
        cr = ConfigReader()
        cr.config_reader = {"url": "mock/url"}

        self.remotes = [cr]

        class Head: pass
        self.head = Head()

        class Object: pass
        self.head.object = Object()
        self.head.object.hexsha = "mock_hexsha"

        class ActiveBranch: pass
        self.active_branch = ActiveBranch()
        self.active_branch.name = "origin"




class GitRepoDetachedHead:

    def __init__(self, *args, **kwargs):
        class ConfigReader: pass
        cr = ConfigReader()
        cr.config_reader = {"url": "mock/url"}

        self.remotes = [cr]

        class Head: pass
        self.head = Head()

        class Object: pass
        self.head.object = Object()
        self.head.object.hexsha = "mock_hexsha"

        @property
        def active_branch(self):
            raise TypeError()


@pytest.fixture
def mock_git_correct_repo():
    mocked_git_repo = GitRepo()
    with mock.patch("git.Repo") as mocked_function:
        mocked_function.return_value = mocked_git_repo
        yield mocked_function


@pytest.fixture
def mock_git_wrong_repo():
    from git.exc import InvalidGitRepositoryError
    def raise_invalid_git_repo(*args, **kwargs): raise InvalidGitRepositoryError()

    with mock.patch("git.Repo") as mocked_function:
        mocked_function.side_effect = raise_invalid_git_repo
        yield mocked_function


@pytest.fixture
def mock_git_detached_head():
    mocked_git_repo_detached = GitRepoDetachedHead()
    with mock.patch("git.Repo") as mocked_function:
        mocked_function.return_value = mocked_git_repo_detached
        yield mocked_function



class TrackingServer:

    def __init__(self):
        self.__current_experiment_id:int = 0
        self.__active_experiment:Optional[Experiment] = None
        self.__experiments:Dict[str, Experiment] = dict()
        self.__current_run_id:int = 0
        self.__current_run:ActiveRun = None
        self.__runs:Dict[str, ActiveRun] = dict()
        self.set_experiment(experiment_name=Experiment.DEFAULT_EXPERIMENT_NAME)

    def start_run(self, run_name:str=None, run_id:int=None, nested:bool=True) -> ActiveRun:

        run:Run = None
        if run_id is not None:
            run = self.__runs.get(run_id)

        if run is None:
            self.__current_run_id += 1

            run = Run(
                run_info = RunInfo(
                    experiment_id = self.__current_experiment_id,
                    run_id = str(self.__current_run_id),
                    run_uuid = self.__current_run_id,
                    run_name = run_name,
                    status = RunStatus.RUNNING,
                    start_time = time.time(),
                    end_time = time.time(),
                    user_id=1,
                    lifecycle_stage = LifecycleStage.ACTIVE
                ),
                run_data = RunData()
            )

            self.__runs[self.__current_experiment_id] = run    

        self.__current_run = ActiveRun(run)
        return self.__current_run
            
    def set_experiment(self, experiment_name: Optional[str] = None, experiment_id: Optional[str] = None) -> Experiment:

        experiment:Experiment = None
        if experiment_name is not None:
            experiment = self.__experiments.get(experiment_name)
        else:
            for e in self.__experiments.values():
                if e.experiment_id == experiment_id:
                    experiment = e
                    break

        if experiment is None:
            self.__current_experiment_id += 1

            experiment = Experiment(
                experiment_id = self.__current_experiment_id,
                name = experiment_name,
                artifact_location = None,
                lifecycle_stage = LifecycleStage.ACTIVE
            )

            self.__experiments[experiment_name] = experiment

        self.__active_experiment = experiment
        return experiment
    
    
    def end_run(self, status:str = RunStatus.to_string(RunStatus.FINISHED)) -> None:
        current_run:ActiveRun = self.active_run()
        if current_run:
            current_run.data.status = status
            self.__current_run = None

    def set_tag(self, key: str, value: Any) -> None:
        run_data:RunData = self.active_run().data
        if run_data:
            run_data.tags[key] = value

    def set_tags(self, tags: Dict[str, Any]) -> None:
        run_data:RunData = self.active_run().data
        if run_data:
            run_data.tags.update(tags)

    def log_param(self, key: str, value: Any) -> None:
        run_data:RunData = self.active_run().data
        if run_data:
            run_data.params[key] = value

    def active_run(self) -> ActiveRun:
        return self.__current_run
    
    def active_experiment(self) -> Experiment:
        return self.__active_experiment




mocked_server:TrackingServer = TrackingServer()




@pytest.fixture(autouse=True)
def mock_set_experiment():
    global mocked_server
    with mock.patch("mlflow.set_experiment") as mocked_function:
        mocked_function.side_effect = mocked_server.set_experiment
        yield mocked_function


@pytest.fixture(autouse=True)
def mock_start_run():
    global mocked_server
    with mock.patch("mlflow.start_run") as mocked_function:
        mocked_function.side_effect = mocked_server.start_run
        yield mocked_function


@pytest.fixture(autouse=True)
def mock_end_run():
    with mock.patch("mlflow.end_run") as mocked_function:
        mocked_function.side_effect = mocked_server.end_run
        yield mocked_function


@pytest.fixture(autouse=True)
def mock_active_run():
    with mock.patch("mlflow.active_run") as mocked_function:
        mocked_function.side_effect = mocked_server.active_run
        yield mocked_function
        

@pytest.fixture(autouse=True)
def mock_set_tags():
    with mock.patch("mlflow.set_tags") as mocked_function:
        mocked_function.side_effect = mocked_server.set_tags
        yield mocked_function


@pytest.fixture(autouse=True)
def mock_log_param():
    with mock.patch("mlflow.log_param") as mocked_function:
        mocked_function.side_effect = mocked_server.log_param
        yield mocked_function