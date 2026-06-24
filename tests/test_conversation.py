from agent.conversation import Conversation


def test_initial_state_has_only_system_message():
    c = Conversation("you are helpful")
    assert len(c.recent_messages()) == 1
    assert c.recent_messages()[0]["role"] == "system"
    assert c.recent_messages()[0]["content"] == "you are helpful"


def test_add_user_appends_message():
    c = Conversation("system")
    c.add_user("hello")
    assert len(c.recent_messages()) == 2
    assert c.recent_messages()[-1]["role"] == "user"
    assert c.recent_messages()[-1]["content"] == "hello"


def test_add_assistant_appends_message():
    c = Conversation("system")
    c.add_user("hello")
    c.add_assistant("hi there")
    assert len(c.recent_messages()) == 3
    assert c.recent_messages()[-1]["role"] == "assistant"
    assert c.recent_messages()[-1]["content"] == "hi there"


def test_messages_build_in_correct_order():
    c = Conversation("system")
    c.add_user("q1")
    c.add_assistant("a1")
    c.add_user("q2")
    roles = [m["role"] for m in c.recent_messages()]
    assert roles == ["system", "user", "assistant", "user"]


def test_pop_last_user_removes_user_message():
    c = Conversation("system")
    c.add_user("hello")
    c.pop_last_user()
    assert len(c.recent_messages()) == 1
    assert c.recent_messages()[0]["role"] == "system"


def test_pop_last_user_does_not_remove_assistant_message():
    c = Conversation("system")
    c.add_user("hello")
    c.add_assistant("hi")
    c.pop_last_user()  # last message is assistant, not user
    assert len(c.recent_messages()) == 3


def test_pop_last_user_is_safe_when_only_system_message():
    c = Conversation("system")
    c.pop_last_user()  # should not raise
    assert len(c.recent_messages()) == 1


def test_replace_system_prompt_updates_first_message():
    c = Conversation("old prompt")
    c.replace_system_prompt("new prompt")
    assert c.recent_messages()[0]["content"] == "new prompt"


def test_replace_system_prompt_preserves_history():
    c = Conversation("old prompt")
    c.add_user("hello")
    c.add_assistant("hi")
    c.replace_system_prompt("new prompt")
    assert len(c.recent_messages()) == 3
    assert c.recent_messages()[1]["content"] == "hello"
    assert c.recent_messages()[2]["content"] == "hi"


def test_clear_resets_to_single_system_message():
    c = Conversation("system")
    c.add_user("hello")
    c.add_assistant("hi")
    c.clear("fresh start")
    assert len(c.recent_messages()) == 1
    assert c.recent_messages()[0]["role"] == "system"
    assert c.recent_messages()[0]["content"] == "fresh start"


def test_recent_messages_returns_list():
    c = Conversation("system")
    assert isinstance(c.recent_messages(), list)


# ── sliding window ────────────────────────────────────────────────────────────


def test_recent_messages_returns_all_when_under_limit():
    c = Conversation("system")
    c.add_user("q1")
    c.add_assistant("a1")
    assert len(c.recent_messages(max_turns=10)) == 3  # system + 2


def test_recent_messages_truncates_oldest_turns():
    c = Conversation("system")
    for i in range(5):
        c.add_user(f"q{i}")
        c.add_assistant(f"a{i}")
    # 5 pairs = 10 messages + system; max_turns=2 keeps last 2 pairs
    result = c.recent_messages(max_turns=2)
    assert len(result) == 5  # system + 4 messages (2 pairs)
    assert result[0]["role"] == "system"
    assert result[1]["content"] == "q3"
    assert result[2]["content"] == "a3"
    assert result[3]["content"] == "q4"
    assert result[4]["content"] == "a4"


def test_recent_messages_always_includes_system_prompt():
    c = Conversation("always here")
    for i in range(20):
        c.add_user(f"q{i}")
        c.add_assistant(f"a{i}")
    result = c.recent_messages(max_turns=1)
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "always here"
    assert len(result) == 3  # system + 1 pair


def test_recent_messages_custom_max_turns():
    c = Conversation("system")
    for i in range(10):
        c.add_user(f"q{i}")
        c.add_assistant(f"a{i}")
    result = c.recent_messages(max_turns=3)
    # system + 3 pairs = 7 messages
    assert len(result) == 7
    assert result[0]["role"] == "system"
