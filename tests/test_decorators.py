from typing import Dict, List, Optional, Set
from unittest.mock import Mock
import mlflow
from mlflow.tracking.fluent import ActiveRun
from mlflow.entities import Experiment, RunData, RunInfo
import pytest
from typeguard import TypeCheckError

from veil.decorators import Run, AutologSession, Autologger, MlflowIsolated, _get_repo_info
from veil.types import StringDict, StringList

from tests.mocks import (
    mock_git_correct_repo, 
    mock_git_wrong_repo, 
    mock_git_detached_head, 
    mocked_server, 
    mock_active_run, 
    mock_start_run, 
    mock_set_experiment, 
    mock_log_param, 
    mock_end_run, 
    mock_set_tags,
)


class TestAutologger:
    """
    Test suite designed for methods belonging to the
    veil.decorators.Autologger class.
    """

    #
    # section: Autologger.__init__
    #
    @pytest.mark.parametrize("illegal_value", [None, 1, "wrong"])
    def test_init_type_check_error_on_illegal_is_autolog_enabled(self, illegal_value) -> None:
        """
        Checks whether Autologger.__init__ raises a TypeCheckError when
        is_autolog_enabled is of illegal type.
        """
        with pytest.raises(TypeCheckError):
            Autologger(
                is_autolog_enabled = illegal_value
            )



    @pytest.mark.parametrize("illegal_value", [None, 1, True])
    def test_init_type_check_error_on_illegal_tracking_uri(self, illegal_value) -> None:
        """
        Checks whether Autologger.__init__ raises a TypeCheckError when
        tracking_uri is of illegal type.
        """
        with pytest.raises(TypeCheckError):
            Autologger(
                tracking_uri = illegal_value
            )



    @pytest.mark.parametrize("illegal_value", [None, 1, True])
    def test_init_type_check_error_on_illegal_experiment_name(self, illegal_value) -> None:
        """
        Checks whether Autologger.__init__ raises a TypeCheckError when
        experiment_name is of illegal type.
        """
        with pytest.raises(TypeCheckError):
            Autologger(
                experiment_name = illegal_value
            )



    def test_init_correctness_on_default_arguments(self) -> None:
        """
        Checks whether Autologger.__init__ returns an Autologger instance
        that is coherent with default arguments.
        """
        autologger: Autologger = Autologger()
        assert(autologger.is_autolog_enabled == True)
        assert(autologger.tracking_uri == mlflow.get_tracking_uri())
        assert(autologger.experiment_name == Experiment.DEFAULT_EXPERIMENT_NAME)



    def test_init_correctness_on_custom_arguments(self) -> None:
        """
        Checks whether Autologger.__init__ returns an Autologger instance
        that is coherent with custom arguments.
        """
        is_autolog_enabled:bool = False
        tracking_uri:str = "prova"
        experiment_name:str = "prova"
        autologger: Autologger = Autologger(
            is_autolog_enabled=is_autolog_enabled,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name
        )
        assert(autologger.is_autolog_enabled == is_autolog_enabled)
        assert(autologger.tracking_uri == tracking_uri)
        assert(autologger.experiment_name == experiment_name)
    

    #
    # section: Autologger.is_autolog_enabled (getter/setter)
    #
    @pytest.mark.parametrize("illegal_value", [None, 1])
    def test_is_autolog_enabled_correctness_on_illegal_value(self, illegal_value) -> None:
        """
        Checks whether Autologger.is_autolog_enabled raises a TypeCheckError when
        calling it with a value of illegal type.
        """
        autologger: Autologger = Autologger()
        with pytest.raises(TypeCheckError):
            autologger.is_autolog_enabled = illegal_value



    @pytest.mark.parametrize("legal_value", [True, False])
    def test_is_autolog_enabled_correctness_on_legal_value(self, legal_value) -> None:
        """
        Checks whether the getter associated to Autologger.is_autolog_enabled returns
        a value that is coherent with the setter.
        """
        autologger: Autologger = Autologger()
        autologger.is_autolog_enabled = legal_value
        assert(autologger.is_autolog_enabled == legal_value)



    #
    # section: Autologger.tracking_uri (getter/setter)
    #
    @pytest.mark.parametrize("illegal_value", [None, 1])
    def test_tracking_uri_correctness_on_illegal_value(self, illegal_value) -> None:
        """
        Checks whether Autologger.tracking_uri raises a TypeCheckError when
        calling it with a value of illegal type.
        """
        autologger: Autologger = Autologger()
        with pytest.raises(TypeCheckError):
            autologger.tracking_uri = illegal_value



    @pytest.mark.parametrize("legal_value", ["prova"])
    def test_tracking_uri_on_legal_value(self, legal_value) -> None:
        """
        Checks whether the getter associated to Autologger.tracking_uri returns
        a value that is coherent with the setter.
        """
        autologger: Autologger = Autologger()
        autologger.tracking_uri = legal_value
        assert(autologger.tracking_uri == legal_value)



    #
    # section: Autologger.experiment_name (getter/setter)
    #
    @pytest.mark.parametrize("illegal_value", [None, 1])
    def test_experiment_name_correctness_on_illegal_value(self, illegal_value) -> None:
        """
        Checks whether Autologger.experiment_name raises a TypeCheckError when
        calling it with a value of illegal type.
        """
        autologger: Autologger = Autologger()
        with pytest.raises(TypeCheckError):
            autologger.experiment_name = illegal_value



    @pytest.mark.parametrize("legal_value", ["prova"])
    def test_experiment_name_on_legal_value(self, legal_value) -> None:
        """
        Checks whether the getter associated to Autologger.experiment_name returns
        a value that is coherent with the setter.
        """
        autologger: Autologger = Autologger()
        autologger.experiment_name = legal_value
        assert(autologger.experiment_name == legal_value)



    #
    # section: Autologger.start_session
    #
    @pytest.mark.parametrize("illegal_value", [1])
    def test_start_session_correctness_on_illegal_name(self, illegal_value) -> None:
        """
        Checks whether Autologger.start_session raises a TypeCheckError when
        calling it with a name of illegal type.
        """
        autologger: Autologger = Autologger()
        with pytest.raises(TypeCheckError):
            autologger.start_session(
                name = illegal_value
            )



    @pytest.mark.parametrize("illegal_value", [None, 1, "prova"])
    def test_start_session_correctness_on_illegal_log_tags(self, illegal_value) -> None:
        """
        Checks whether Autologger.start_session raises a TypeCheckError when
        calling it with a log_tags of illegal type.
        """
        autologger: Autologger = Autologger()
        with pytest.raises(TypeCheckError):
            autologger.start_session(
                log_tags = illegal_value
            )
    

    def test_start_session_correctness_on_default_arguments(self) -> None:
        """
        Checks whether Autologger.start_session returns an AutologSession
        instance that is coherent with default arguments.
        """
        autologger: Autologger = Autologger()
        session: AutologSession = autologger.start_session()

        assert(session.name == None)
        assert(session.log_tags == dict())



    def test_start_session_correctness_on_custom_arguments(self) -> None:
        """
        Checks whether Autologger.start_session returns an AutologSession
        instance that is coherent with custom arguments.
        """
        name: Optional[str] = "prova"
        log_tags: StringDict = {"prova":"prova"}

        autologger: Autologger = Autologger()
        session: AutologSession = autologger.start_session(
            name = name,
            log_tags = log_tags
        )
        assert(session.name == name)
        assert(session.log_tags == log_tags)
    


    #
    # section: Autologger.run
    #
    @pytest.mark.parametrize("illegal_value", [1])
    def test_run_correctness_on_illegal_name(self, illegal_value) -> None:
        """
        Checks whether Autologger.run raises a TypeCheckError when
        calling it with a name of illegal type.
        """
        autologger: Autologger = Autologger()
        with pytest.raises(TypeCheckError):
            autologger.run(
                name = illegal_value
            )



    @pytest.mark.parametrize("illegal_value", [1, "prova"])
    def test_run_correctness_on_illegal_log_params(self, illegal_value) -> None:
        """
        Checks whether Autologger.run raises a TypeCheckError when
        calling it with a log_params of illegal type.
        """
        autologger: Autologger = Autologger()
        with pytest.raises(TypeCheckError):
            autologger.run(
                log_params = illegal_value
            )



    @pytest.mark.parametrize("illegal_value", [1, "prova", {1:"dict"}, {"dict":1}])
    def test_run_correctness_on_illegal_log_tags(self, illegal_value) -> None:
        """
        Checks whether Autologger.run raises a TypeCheckError when
        calling it with a log_tags of illegal type.
        """
        autologger: Autologger = Autologger()
        with pytest.raises(TypeCheckError):
            autologger.run(
                log_tags = illegal_value
            )

    
    def test_run_correctness_on_default_arguments(self) -> None:
        """
        Checks whether Autologger.run returns an Run instance
        that is coherent with default arguments.
        """
        autologger: Autologger = Autologger()
        run: Run = autologger.run()

        assert(run.name == None)
        assert(run.log_params == None)
        assert(run.log_tags == dict())



    def test_run_correctness_on_custom_arguments(self) -> None:
        """
        Checks whether Autologger.run returns an Run instance
        that is coherent with custom arguments.
        """
        name: Optional[str] = "prova"
        log_params: StringList = ["prova"]
        log_tags: StringDict = {"prova":"prova"}

        autologger: Autologger = Autologger()
        session: Run = autologger.run(
            name = name,
            log_params = log_params,
            log_tags = log_tags
        )
        assert(session.name == name)
        assert(session.log_params == log_params)
        assert(session.log_tags == log_tags)





class TestMlflowIsolated:
    """
    Test suite designed for methods belonging to the
    veil.decorators.MlflowIsolated class.
    """

    #
    # section: MlflowIsolated.__init__
    #
    @pytest.mark.parametrize("illegal_value", [None, 1, "wrong"])
    def test_init_type_check_error_on_illegal_autologger(self, illegal_value) -> None:
        """
        Checks whether MlflowIsolated.__init__ raises a TypeCheckError when
        autologger is of illegal type.
        """
        with pytest.raises(TypeCheckError):
            MlflowIsolated(
                autologger = illegal_value
            )



    #
    # section: MlflowIsolated.__call__
    #      
    @pytest.mark.parametrize("illegal_value", [None, 1, "wrong"])
    def test_call_type_check_error_on_illegal_func(self, illegal_value) -> None:
        """
        Checks whether MlflowIsolated.__call__ raises a TypeCheckError when
        called with a value of illegal type.
        """
        delegate:MlflowIsolated = MlflowIsolated(autologger = Autologger())
        with pytest.raises(TypeCheckError):
            delegate(illegal_value)



    def test_call_correctness_on_autolog_enabled_inside_session(
        self, 
        mock_start_run:Mock,
        mock_end_run:Mock,
        mock_set_experiment:Mock
    ) -> None:
        """
        Checks whether calling a method annotated with MlflowIsolated
        inside an autolog session and with autolog enabled, actually 
        does temporarily switch the tracking server uri and the currently
        used experiment.
        """
        experiment_name:str = "experiment_name"
        tracking_uri:str = "tracking_uri"
        run_name:str = "run_name"
        autologger:Autologger = Autologger(
            experiment_name = experiment_name,
            tracking_uri = tracking_uri,
            is_autolog_enabled = True
        )

        @MlflowIsolated(autologger=autologger)
        def annotated_function():
            assert(mlflow.get_tracking_uri() == tracking_uri)
            assert(mocked_server.active_experiment().name == experiment_name)

        with autologger.start_session(name=run_name):
            annotated_function()
        
        assert(mlflow.get_tracking_uri() != tracking_uri)
        assert(mocked_server.active_experiment().name != experiment_name)



    def test_call_correctness_on_autolog_enabled_outside_session(
        self, 
        mock_start_run:Mock,
        mock_end_run:Mock,
        mock_set_experiment:Mock
    ) -> None:
        """
        Checks whether calling a method annotated with MlflowIsolated
        outside an autolog session and with autolog enabled, actually 
        does not log anything.
        """
        experiment_name:str = "experiment_name"
        tracking_uri:str = "tracking_uri"
        autologger:Autologger = Autologger(
            experiment_name = experiment_name,
            tracking_uri = tracking_uri,
            is_autolog_enabled = True
        )

        @MlflowIsolated(autologger=autologger)
        def annotated_function():
            mock_start_run.assert_not_called()
            mock_end_run.assert_not_called()
            mock_set_experiment.assert_not_called()

        annotated_function()



    def test_call_correctness_on_autolog_disabled_outside_session(
        self, 
        mock_start_run:Mock,
        mock_end_run:Mock,
        mock_set_experiment:Mock
    ) -> None:
        """
        Checks whether calling a method annotated with MlflowIsolated
        outside an autolog session and with autolog disabled, actually 
        does not log anything.
        """
        experiment_name:str = "experiment_name"
        tracking_uri:str = "tracking_uri"
        autologger:Autologger = Autologger(
            experiment_name = experiment_name,
            tracking_uri = tracking_uri,
            is_autolog_enabled = False
        )

        @MlflowIsolated(autologger=autologger)
        def annotated_function():
            mock_start_run.assert_not_called()
            mock_end_run.assert_not_called()
            mock_set_experiment.assert_not_called()

        annotated_function()

        



class TestAutologSession:
    """
    Test suite designed for methods belonging to the
    veil.decorators.AutologSession class.
    """

    #
    # section: AutologSession.__init__
    #
    @pytest.mark.parametrize("illegal_value", [None, 1, "value"])
    def test_init_correctness_on_illegal_autologger(self, illegal_value) -> None:
        """
        Checks whether AutologSession.__init__ raises a TypeCheckError when
        autologger is of illegal type.
        """
        with pytest.raises(TypeCheckError):
            AutologSession(
                autologger = illegal_value
            )



    @pytest.mark.parametrize("illegal_value", [1])
    def test_init_correctness_on_illegal_name(self, illegal_value) -> None:
        """
        Checks whether AutologSession.__init__ raises a TypeCheckError when
        name is of illegal type.
        """
        with pytest.raises(TypeCheckError):
            AutologSession(
                autologger = Autologger(),
                name = illegal_value   
            )



    @pytest.mark.parametrize("illegal_value", [None, 1, "prova", {1:"dict"}, {"dict":1}])
    def test_init_correctness_on_illegal_log_tags(self, illegal_value) -> None:
        """
        Checks whether AutologSession.__init__ raises a TypeCheckError when
        log_tags is of illegal type.
        """
        with pytest.raises(TypeCheckError):
            AutologSession(
                autologger = Autologger(),
                log_tags = illegal_value   
            )

    
    
    def test_init_correctness_on_default_arguments(self) -> None:
        """
        Checks whether AutologSession.__init__ returns an AutologSession instance
        that is coherent with default arguments.
        """
        session: AutologSession = AutologSession(
            autologger = Autologger()
        )

        assert(session.name == None)
        assert(session.log_tags == dict())



    def test_init_correctness_on_custom_arguments(self) -> None:
        """
        Checks whether AutologSession.__init__ returns an AutologSession instance
        that is coherent with custom arguments.
        """
        name: Optional[str] = "prova"
        log_tags: StringDict = {"prova":"prova"}

        session: Run = AutologSession(
            autologger = Autologger(),
            name = name,
            log_tags = log_tags
        )
        assert(session.name == name)
        assert(session.log_tags == log_tags)



    #
    # section: AutologSession.run_id (getter - IS IT REALLY NECESSARY?)
    #



    #
    # section: AutologSession.name (getter/setter)
    #
    @pytest.mark.parametrize("illegal_value", [1])
    def test_name_correctness_on_illegal_value(self, illegal_value) -> None:
        """
        Checks whether AutologSession.name raises a TypeCheckError when
        calling it with a value of illegal type.
        """
        session:AutologSession = Autologger().start_session()
        with pytest.raises(TypeCheckError):
            session.name = illegal_value



    @pytest.mark.parametrize("legal_value", ["prova"])
    def test_name_correctness_on_legal_value(self, legal_value) -> None:
        """
        Checks whether the getter associated to AutologSession.name returns
        a value that is coherent with the setter.
        """
        session:AutologSession = Autologger().start_session()
        session.name = legal_value
        assert(session.name == legal_value)



    #
    # section: AutologSession.log_tags (getter/setter)
    #
    @pytest.mark.parametrize("illegal_value", [None, 1])
    def test_log_tags_correctness_on_illegal_value(self, illegal_value) -> None:
        """
        Checks whether AutologSession.log_tags raises a TypeCheckError when
        calling it with a value of illegal type.
        """
        session:AutologSession = Autologger().start_session()
        with pytest.raises(TypeCheckError):
            session.log_tags = illegal_value



    @pytest.mark.parametrize("legal_value", [{"prova":"prova"}])
    def test_log_tags_correctness_on_legal_value(self, legal_value) -> None:
        """
        Checks whether the getter associated to AutologSession.log_tags returns
        a value that is coherent with the setter.
        """
        session:AutologSession = Autologger().start_session()
        session.log_tags = legal_value
        assert(session.log_tags == legal_value)



    #
    # section: AutologSession.__enter__/__exit__
    #
    def test_ctx_manager_correctness_on_autolog_enabled(self,
        mock_start_run:Mock,
        mock_end_run:Mock,
        mock_set_experiment:Mock) -> None:
        experiment_name:str = "experiment_name"
        tracking_uri:str = "tracking_uri"
        run_name:str = "run_name"
        """
        Checks whether calling the context manager (e.g. calling __enter__
        and __exit__ in sequence) with autolog enabled actually logs
        and terminates a novel parent run to the tracking server.
        """

        autologger:Autologger = Autologger(
            experiment_name = experiment_name,
            tracking_uri = tracking_uri,
            is_autolog_enabled = True
        )

        session:AutologSession = AutologSession(
            autologger = autologger,
            name = run_name
        )

        with session:
            mock_start_run.assert_called_with(run_name=run_name)
            assert(mock_end_run.call_count == 1)
            run_id:int = session.run_id
        mock_start_run.assert_called_with(run_id = run_id)
        assert(mock_end_run.call_count == 2)
        


    def test_ctx_manager_correctness_on_autolog_disabled(self, 
        mock_start_run:Mock,
        mock_end_run:Mock,
        mock_set_experiment:Mock) -> None:
        """
        Checks whether calling the context manager (e.g. calling __enter__
        and __exit__ in sequence) with autolog disabled actually does not
        log anything to the tracking server.
        """

        run_name:str = "run_name"
        session:AutologSession = AutologSession(
            autologger = Autologger(is_autolog_enabled = False),
            name = run_name
        )

        with session:
            pass

        mock_start_run.assert_not_called()
        mock_end_run.assert_not_called()
        mock_set_experiment.assert_not_called()






class TestRun:
    """
    Test case suite designed for testing methods belonging to
    veil.decorators.Run class.
    """

    #
    # section: Run.__init__ tests
    #

    @pytest.mark.parametrize("illegal_value", [None, 1, "wrong"])
    def test_init_type_check_error_on_illegal_autologger(self, illegal_value) -> None:
        """
        Checks whether Run.__init__ raises a TypeCheckError when
        autologger is of illegal type.
        """
        with pytest.raises(TypeCheckError):
            Run(
               autologger = illegal_value
            )



    @pytest.mark.parametrize("illegal_value", [1])
    def test_init_type_check_error_on_illegal_name(self, illegal_value) -> None:
        """
        Checks whether Run.__init__ raises a TypeCheckError when
        name is of illegal type.
        """
        with pytest.raises(TypeCheckError):
            Run(
                autologger = Autologger(),
                name = illegal_value
            )



    @pytest.mark.parametrize("illegal_value", [1])
    def test_init_type_check_error_on_illegal_log_params(self, illegal_value) -> None:
        """
        Checks whether Run.__init__ raises a TypeCheckError when
        log_params is of illegal type.
        """
        with pytest.raises(TypeCheckError):
            Run(
                autologger = Autologger(),
                log_params = illegal_value
            )



    @pytest.mark.parametrize("illegal_value", [None, 1])
    def test_init_type_check_error_on_illegal_log_tags(self, illegal_value) -> None:
        """
        Checks whether Run.__init__ raises a TypeCheckError when
        log_tags is of illegal type.
        """
        with pytest.raises(TypeCheckError):
            Run(
                autologger = Autologger(),
                log_tags = illegal_value
            )



    def test_init_correctness_on_default_arguments(self) -> None:
        """
        Checks whether Run.__init__ returns a Run instance
        that is coherent with default arguments.
        """
        decorator: Run = Run(autologger = Autologger())
        assert(decorator.name == None)
        assert(decorator.log_params == None)
        assert(decorator.log_tags == dict())



    def test_init_correctness_on_custom_arguments(self) -> None:
        """
        Checks whether Run.__init__ returns a Run instance
        that is coherent with custom arguments.
        """
        name: Optional[str] = "prova"
        log_params: Optional[StringList] = ["prova"]
        log_tags: StringDict = {"prova":"prova"}
        decorator: Run = Run(
            autologger = Autologger(),
            name = name,
            log_params = log_params,
            log_tags = log_tags
        )
        assert(decorator.name == name)
        assert(decorator.log_params == log_params)
        assert(decorator.log_tags == log_tags)
    


    #
    # section: Run.autologger (getter) tests
    #
    def test_autologger_correctness(self) -> None:
        """
        Checks whether Run.autologger returns a value consistent
        with the associated __init__ arg.
        """
        param:Autologger = Autologger()
        run:Run = Run(
            autologger=param
        )
        assert(run.autologger == param)


    #
    # section: Run.name (getter) tests
    #
    def test_name_correctness(self) -> None:
        """
        Checks whether Run.name returns a value consistent
        with the associated __init__ arg.
        """
        param:Optional[str] = "value"
        run:Run = Run(
            autologger=Autologger(), 
            name = param
        )
        assert(run.name == param)
        


    #
    # section: Run.log_params (getter) tests
    #
    def test_log_params_correctness(self) -> None:
        """
        Checks whether Run.log_params returns a value consistent
        with the associated __init__ arg.
        """
        param:StringList = []
        run:Run = Run(
            autologger=Autologger(), 
            log_params = param
        )
        assert(run.log_params == param)
        


    #
    # section: Run.log_tags (getter) tests
    #
    def test_log_tags_correctness(self) -> None:
        """
        Checks whether Run.log_tags returns a value consistent
        with the associated __init__ arg.
        """
        param:StringDict = {}
        run:Run = Run(
            autologger=Autologger(), 
            log_tags = param
        )
        assert(run.log_tags == param)
        


    #
    # section: Run.__call__ (getter) tests
    #
    @pytest.mark.parametrize("illegal_value", [None, 1, "wrong"])
    def test_call_type_check_error_on_illegal_func(self, illegal_value) -> None:
        """
        Checks whether Run.__call__ raises a TypeCheckError when
        called with a value of illegal type.
        """
        decorator: Run = Run(autologger = Autologger())
        with pytest.raises(TypeCheckError):
            decorator(illegal_value)



    @pytest.mark.parametrize("run_name", [None, "run_name"])
    def test_call_correctness_on_autolog_enabled_within_session(self,
        mock_start_run:Mock,
        mock_end_run:Mock,
        mock_set_experiment:Mock,
        run_name:Optional[str]) -> None:
        experiment_name:str = "experiment_name"
        tracking_uri:str = "tracking_uri"

        autologger:Autologger = Autologger(
            experiment_name = experiment_name,
            tracking_uri = tracking_uri,
            is_autolog_enabled = True
        )

        @Run(autologger = autologger)
        def annotated_function():
            assert(mocked_server.active_experiment().name == experiment_name)
            assert(mocked_server.active_run().info.run_name == annotated_function.__name__ or run_name)

        with autologger.start_session(name=run_name):
            annotated_function()



    def test_call_correctness_on_autolog_enabled_no_session(self,
        mock_start_run:Mock,
        mock_end_run:Mock,
        mock_set_experiment:Mock) -> None:
        experiment_name:str = "experiment_name"
        tracking_uri:str = "tracking_uri"
        run_name:str = "run_name"

        autologger:Autologger = Autologger(
            experiment_name = experiment_name,
            tracking_uri = tracking_uri,
            is_autolog_enabled = True
        )

        @Run(autologger = autologger)
        def annotated_function():
            mock_start_run.assert_not_called()
            mock_end_run.assert_not_called()
            mock_set_experiment.assert_not_called()

        annotated_function()



    def test_call_correctness_on_autolog_disabled_within_session(self, 
        mock_start_run:Mock,
        mock_end_run:Mock,
        mock_active_run:Mock,
        mock_log_param:Mock,
        mock_set_tags:Mock,
        mock_set_experiment:Mock) -> None:

        autologger:Autologger = Autologger(is_autolog_enabled = False)

        @Run(autologger = autologger)
        def annotated_function():
            mock_start_run.assert_not_called()
            mock_end_run.assert_not_called()
            mock_set_experiment.assert_not_called()
            mock_active_run.assert_not_called()
            mock_log_param.assert_not_called()
            mock_set_tags.assert_not_called()

        with autologger.start_session():
            annotated_function()



    def test_call_correctness_on_autolog_disabled_no_session(self, 
        mock_start_run:Mock,
        mock_end_run:Mock,
        mock_active_run:Mock,
        mock_log_param:Mock,
        mock_set_tags:Mock,
        mock_set_experiment:Mock) -> None:

        autologger:Autologger = Autologger(is_autolog_enabled = False)

        @Run(autologger = autologger)
        def annotated_function():
            mock_start_run.assert_not_called()
            mock_end_run.assert_not_called()
            mock_set_experiment.assert_not_called()
            mock_active_run.assert_not_called()
            mock_log_param.assert_not_called()
            mock_set_tags.assert_not_called()

        annotated_function()
            


    @pytest.mark.parametrize("log_params", [[], ["a", "b"], ["a", "f", "g"]])
    @pytest.mark.parametrize("args, kwargs",  [
        ([1,2,3,4,5], {}),
        ([1,2], {"c":3, "d":4, "e":5}),
        ([], {"a":1, "b":2, "c":3, "d":4, "e":5})
    ])
    def test_call_correctness_on_log_params(
        self,
        mock_log_param:Mock,
        log_params,
        args:List[str],
        kwargs:Dict[str, int]
    ) -> None:
        """
        Checks that calling the Run decorator with log_params argument
        logs a consistent number of parameters according to the content
        of log_params itself and to the number of keyword arguments of 
        the decorated function (under different conditions).
        """
        autologger:Autologger = Autologger(is_autolog_enabled = True)

        @Run(autologger = autologger, log_params=log_params)
        def annotated_function(a, b, c, d, e):
            if len(log_params) == 0:
                assert(mock_log_param.call_count == len(kwargs))
            else:
                common_args:Set[str] = set(kwargs.keys()).intersection(log_params)
                assert(mock_log_param.call_count == len(common_args))

        with autologger.start_session():
            annotated_function(*args, **kwargs)



    @pytest.mark.parametrize("session_tags", [{}, {"a":"1"}])
    @pytest.mark.parametrize("run_tags", [{}, {"b":"1"}, {"a":"2"}, {"a":"2", "b":"1"}])
    def test_call_correctness_on_log_tags(
        self,
        session_tags:StringDict,
        run_tags:StringDict
    ) -> None:
        """
        Checks that calling the Run decorator with log_tags argument
        logs a consistent number of tags according to the content
        of log_params itself and to the session-bound tags.
        """
        autologger:Autologger = Autologger(is_autolog_enabled = True)

        @Run(autologger = autologger, log_tags=run_tags)
        def annotated_function():
            data:RunData = mlflow.active_run().data

            actual_tags:StringDict = session_tags.copy()
            actual_tags.update(run_tags)

            # we avoid the side-effecting of automatically logged git-related tags
            # by injecting them into actual_tags
            diff_tags = dict(set(data.tags.items()) - set(actual_tags.items()))
            actual_tags.update(diff_tags)

            assert(data.tags == actual_tags)

        with autologger.start_session(log_tags=session_tags):
            annotated_function()


class TestGitRepo:

    def test_git_repo_info_on_correct_repo(
        self, 
        mock_git_correct_repo: Mock,
    ):
        repo_uri, sha_commit, branch_name = _get_repo_info()

        assert (repo_uri is not None)
        assert (sha_commit is not None)
        assert (branch_name is not None)

    def test_git_repo_info_on_invalid_repo(
        self, 
        mock_git_wrong_repo: Mock,
    ):
        repo_uri, sha_commit, branch_name = _get_repo_info()

        assert (repo_uri is None)
        assert (sha_commit is None)
        assert (branch_name is None)    

    def test_git_repo_info_on_detached_branch(
        self, 
        mock_git_detached_head: Mock,
    ):
        repo_uri, sha_commit, branch_name = _get_repo_info()

        assert (repo_uri is not None)
        assert (sha_commit is not None)
        assert (branch_name is None)
