from agent.conversation import Conversation


def test_initial_state_has_only_system_message():
    c = Conversation("you are helpful")
    assert len(c.messages) == 1
    assert c.messages[0]["role"] == "system"
    assert c.messages[0]["content"] == "you are helpful"


def test_add_user_appends_message():
    c = Conversation("system")
    c.add_user("hello")
    assert len(c.messages) == 2
    assert c.messages[-1]["role"] == "user"
    assert c.messages[-1]["content"] == "hello"


def test_add_assistant_appends_message():
    c = Conversation("system")
    c.add_user("hello")
    c.add_assistant("hi there")
    assert len(c.messages) == 3
    assert c.messages[-1]["role"] == "assistant"
    assert c.messages[-1]["content"] == "hi there"


def test_messages_build_in_correct_order():
    c = Conversation("system")
    c.add_user("q1")
    c.add_assistant("a1")
    c.add_user("q2")
    roles = [m["role"] for m in c.messages]
    assert roles == ["system", "user", "assistant", "user"]


def test_pop_last_user_removes_user_message():
    c = Conversation("system")
    c.add_user("hello")
    c.pop_last_user()
    assert len(c.messages) == 1
    assert c.messages[0]["role"] == "system"


def test_pop_last_user_does_not_remove_assistant_message():
    c = Conversation("system")
    c.add_user("hello")
    c.add_assistant("hi")
    c.pop_last_user()  # last message is assistant, not user
    assert len(c.messages) == 3


def test_pop_last_user_is_safe_when_only_system_message():
    c = Conversation("system")
    c.pop_last_user()  # should not raise
    assert len(c.messages) == 1


def test_replace_system_prompt_updates_first_message():
    c = Conversation("old prompt")
    c.replace_system_prompt("new prompt")
    assert c.messages[0]["content"] == "new prompt"


def test_replace_system_prompt_preserves_history():
    c = Conversation("old prompt")
    c.add_user("hello")
    c.add_assistant("hi")
    c.replace_system_prompt("new prompt")
    assert len(c.messages) == 3
    assert c.messages[1]["content"] == "hello"
    assert c.messages[2]["content"] == "hi"


def test_clear_resets_to_single_system_message():
    c = Conversation("system")
    c.add_user("hello")
    c.add_assistant("hi")
    c.clear("fresh start")
    assert len(c.messages) == 1
    assert c.messages[0]["role"] == "system"
    assert c.messages[0]["content"] == "fresh start"


def test_messages_property_returns_list():
    c = Conversation("system")
    assert isinstance(c.messages, list)
