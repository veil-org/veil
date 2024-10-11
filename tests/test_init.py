"""
???
"""
from typing import Optional
import pytest
from typeguard import TypeCheckError

import veil
from veil.decorators import Run, AutologSession
from veil.types import StringDict, StringList
from tests.mocks import mocked_server, mock_active_run, mock_start_run, mock_set_experiment, mock_log_param, mock_end_run, mock_set_tags



#
# section: __init__.py:is/set_autolog_enabled tests
#
class TestSetIsAutologEnabled:

    @pytest.mark.parametrize("illegal_value", [None, 1])
    def test_set_autolog_enabled_correctness_on_illegal_value(self, illegal_value) -> None:
        """
        Checks whether veil.set_autolog_enabled raises a TypeCheckError when
        calling it with a value of illegal type.
        """
        with pytest.raises(TypeCheckError):
            veil.set_autolog_enabled(illegal_value)



    @pytest.mark.parametrize("legal_value", [True, False])
    def test_set_autolog_enabled_correctness_on_legal_value(self, legal_value) -> None:
        veil.set_autolog_enabled(legal_value)
        assert(veil.is_autolog_enabled() == legal_value)



#
# section: __init__.py:get/set_experiment_name
#
class TestSetGetExperimentName:

    @pytest.mark.parametrize("illegal_value", [None, 1])
    def test_set_experiment_name_correctness_on_illegal_value(self, illegal_value) -> None:
        """
        Checks whether veil.set_experiment_name raises a TypeCheckError when
        calling it with a value of illegal type.
        """
        with pytest.raises(TypeCheckError):
            veil.set_experiment_name(illegal_value)



    @pytest.mark.parametrize("legal_value", ["prova"])
    def test_set_experiment_name_correctness_on_legal_value(self, legal_value) -> None:
        veil.set_experiment_name(legal_value)
        assert(veil.get_experiment_name() == legal_value)



#
# section: __init__.py:get/set_experiment_name
#
class TestSetGetExperimentName:

    @pytest.mark.parametrize("illegal_value", [None, 1])
    def test_set_tracking_uri_correctness_on_illegal_value(self, illegal_value) -> None:
        """
        Checks whether veil.set_tracking_uri raises a TypeCheckError when
        calling it with a value of illegal type.
        """
        with pytest.raises(TypeCheckError):
            veil.set_tracking_uri(illegal_value)



    @pytest.mark.parametrize("legal_value", ["prova"])
    def test_set_tracking_uri_correctness_on_legal_value(self, legal_value) -> None:
        veil.set_tracking_uri(legal_value)
        assert(veil.get_tracking_uri() == legal_value)



#
# section: __init__.py:start_session
#
class TestStartSession:

    @pytest.mark.parametrize("illegal_value", [1])
    def test_start_session_correctness_on_illegal_name(self, illegal_value) -> None:
        """
        Checks whether veil.start_session raises a TypeCheckError when
        calling it with a name of illegal type.
        """
        with pytest.raises(TypeCheckError):
            veil.start_session(
                name = illegal_value   
            )



    @pytest.mark.parametrize("illegal_value", [None, 1, "prova", {1:"dict"}, {"dict":1}])
    def test_start_session_correctness_on_illegal_log_tags(self, illegal_value) -> None:
        """
        Checks whether veil.start_session raises a TypeCheckError when
        calling it with a log_tags of illegal type.
        """
        with pytest.raises(TypeCheckError):
            veil.start_session(
                log_tags = illegal_value   
            )

    
    
    def test_start_session_correctness_on_default_arguments(self) -> None:
        """
        Checks whether veil.start_session returns an AutologSession 
        instance coherent with default arguments
        """
        session: AutologSession = veil.start_session()

        assert(session.name == None)
        assert(session.log_tags == dict())



    def test_start_session_correctness_on_custom_arguments(self) -> None:
        """
        Checks whether veil.start_session returns an AutologSession 
        instance coherent with custom arguments
        """
        name: Optional[str] = "prova"
        log_tags: StringDict = {"prova":"prova"}

        session: AutologSession = veil.start_session(
            name = name,
            log_tags = log_tags
        )
        assert(session.name == name)
        assert(session.log_tags == log_tags)



#
# section: __init__.py:run
#
class TestRun:

    @pytest.mark.parametrize("illegal_value", [1])
    def test_run_correctness_on_illegal_name(self, illegal_value) -> None:
        """
        Checks whether veil.run raises a TypeCheckError when
        calling it with a name of illegal type.
        """
        with pytest.raises(TypeCheckError):
            veil.run(
                name = illegal_value
            )



    @pytest.mark.parametrize("illegal_value", [1, "prova"])
    def test_run_correctness_on_illegal_log_params(self, illegal_value) -> None:
        """
        Checks whether veil.run raises a TypeCheckError when
        calling it with a log_params of illegal type.
        """
        with pytest.raises(TypeCheckError):
            veil.run(
                log_params = illegal_value
            )



    @pytest.mark.parametrize("illegal_value", [1, "prova", {1:"dict"}, {"dict":1}])
    def test_run_correctness_on_illegal_log_tags(self, illegal_value) -> None:
        """
        Checks whether veil.run raises a TypeCheckError when
        calling it with a log_tags of illegal type.
        """
        with pytest.raises(TypeCheckError):
            veil.run(
                log_tags = illegal_value
            )

    
    def test_run_correctness_on_default_arguments(self) -> None:
        """
        Checks whether veil.run returns a Run instance
        coherent with default arguments
        """
        run: Run = veil.run()

        assert(run.name == None)
        assert(run.log_params == None)
        assert(run.log_tags == dict())



    def test_run_correctness_on_custom_arguments(self) -> None:
        """
        Checks whether veil.run returns a Run instance
        coherent with custom arguments
        """
        name: Optional[str] = "prova"
        log_params: StringList = ["prova"]
        log_tags: StringDict = {"prova":"prova"}

        session: Run = veil.run(
            name = name,
            log_params = log_params,
            log_tags = log_tags
        )
        assert(session.name == name)
        assert(session.log_params == log_params)
        assert(session.log_tags == log_tags)