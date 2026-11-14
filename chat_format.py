"""Token layout for chat tuning.

GPT-2's tokenizer has no role markers, but the model's vocabulary is padded from 50257 to 50304
rows, so three of the unused rows are given meanings here. A conversation is one document:

    <eot> <user> hello <assistant> Hi there! <end> <user> ... <assistant> ... <end>

The text between markers is ordinary GPT-2 tokens. `<eot>` (50256) is the document separator the
trainer already knows; attention never crosses it, so it only ever appears at the start.
"""
EOT_ID = 50256
USER_ID = 50257
ASSISTANT_ID = 50258
END_ID = 50259
CHAT_VOCAB = END_ID + 1  # tokens a chat model may emit

# Each new token's rows start as the average of existing tokens with a related meaning, so tuning
# has less to learn. A string means "every token it encodes to"; an int is a token id.
SEED_TOKENS = {
    USER_ID: ["Human", " Human", "User", " User", "Question"],
    ASSISTANT_ID: ["AI", " AI", "Assistant", " Assistant", "Answer"],
    END_ID: [EOT_ID],
}


def render_conversation(messages, encode):
    """Token ids for a whole conversation. `messages` is a list of {"role", "content"}."""
    ids = [EOT_ID]
    for message in messages:
        text = encode(message["content"].strip())
        if message["role"] == "user":
            ids += [USER_ID, *text]
        else:
            ids += [ASSISTANT_ID, *text, END_ID]
    return ids


def render_prompt(messages, encode):
    """Token ids for a conversation that ends on a user turn, ready for the model to reply."""
    return render_conversation(messages, encode) + [ASSISTANT_ID]


def assistant_target_mask(inputs):
    """True at positions whose next token belongs to an assistant reply, including its <end>."""
    import torch
    index = torch.arange(len(inputs), device=inputs.device)
    never = torch.full_like(index, -1)
    last_assistant = torch.cummax(torch.where(inputs == ASSISTANT_ID, index, never), 0).values
    other = (inputs == USER_ID) | (inputs == END_ID) | (inputs == EOT_ID)
    last_other = torch.cummax(torch.where(other, index, never), 0).values
    return last_assistant > last_other
