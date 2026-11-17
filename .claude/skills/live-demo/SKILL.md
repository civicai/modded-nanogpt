---
name: live-demo
description: Run CivAI's live "building an AI" demonstration. Use when asked, often by voice during a presentation, to start a pretraining run or an instruction-tuning (chat-tuning) run for the little nanoGPT model, to show training results on the display, or to make a page, interface or tab for the demo (a training monitor, a document-completion interface, a chat interface, a voice mode).
---

# Live demo

You are on stage with a presenter in front of a non-technical audience. They talk; you train a small
language model on rented GPUs and write small web pages that appear on the screen behind them. The
screen is the salt page `/p/nanogpt-live`. Each page you publish becomes a tab there, and the
audience watches your code being typed into the page's editor before the tab opens.

## How to behave

- **Act at once.** Do not ask questions, plan aloud or explore the repository. Everything you need is
  on this page.
- **Say very little.** One short sentence before you start and one when you are done, with no lists
  and no summary of what you built: the presenter is mid-sentence and the audience can see the page.
  Never paste code into the conversation.
- **Be quick.** A page should be published within about a minute of being asked for. Simple and
  working beats clever.
- **Use plain words** in anything the audience will read: "pre-training", "step", "how wrong its
  guesses are". Avoid jargon such as "loss", "tokens", "checkpoint", "inference".
- Run every command from the repository root.

## Training

The GPUs are normally reserved before the talk, so these start within a second or two. If they are
not reserved, the same commands still work, after a wait for GPUs and about three minutes of loading.

| Asked for | Command |
|---|---|
| Start pretraining | `uvx --with modal python live.py pretrain --no-follow` |
| Start instruction tuning (chat tuning) | `uvx --with modal python live.py tune --no-follow` |
| Start instruction tuning with the model named, say, Pip | `uvx --with modal python live.py tune --name Pip --no-follow` |
| What is happening | `uvx --with modal python live.py status` |

- Pretraining takes about 3 minutes and instruction tuning about 30 seconds. Do not wait for them:
  start the run, say it has started, and stop. The page shows progress by itself.
- Tuning can only start once pretraining has finished. If it has not, `tune` queues it to start then.
- Each run can be done once per trainer. `live.py restart` loads a fresh trainer (3 minutes), and
  `live.py release` gives the GPUs back. Only do either when asked.

## Making a tab

1. Write one HTML file in `demo/tabs/`, for example `demo/tabs/monitor.html`. It holds markup, one
   `<style>` block and one `<script>` block. No `<html>`, `<head>` or `<body>` tags.
2. Publish it: `python3 demo/tab.py publish demo/tabs/monitor.html --title "Training monitor"`
3. The command prints whether the page ran your code without errors. If it printed errors, fix the
   file and publish again under the same title, which replaces the tab.

Write every page yourself, from scratch, each time: that you are writing it live is the point of the
demo. Do not look for earlier versions in `demo/tabs/` or anywhere else.

To change a tab later, edit its file and publish again under the same title.

### Rules for the page

- Light background, dark text, no emojis. The page already styles `body` (18px text, 24px padding),
  headings, `button`, `input`, `textarea` and `select`. Colors that fit: navy `#0d2a45`, blue
  `#23527f`, red accent `#e36344`, grays `#f9fafb`, `#e5e7eb`, `#4b5563`.
- It is shown on a laptop, zoomed in, beside a sidebar, to people sitting back from it: body text at
  18px or more. The page may be as narrow as 640px and as short as 430px, so lay it out to fit that
  without scrolling where you can.
- It runs in a sandbox. No libraries, no CDNs, no images, no network requests, no `localStorage` or
  cookies. Plain HTML, CSS and JavaScript, with inline SVG for charts.
- Put text from the model on the page with `textContent`, never `innerHTML`.
- Everything the page needs from outside comes from the global `demo` object below.

### The `demo` object

```js
// Training state. The listener is called at once and then about once a second.
demo.onTraining((training) => { ... });
demo.training();                        // the same object, right now

// Document completion by the pretrained model. Resolves with the full text.
await demo.complete(prompt, { maxTokens: 120, temperature: 0.8, onText: (piece, textSoFar) => {} });

// One reply from the chat-tuned model. messages = [{ role: 'user' | 'assistant', content }], ending on a user turn.
await demo.chat(messages, { maxTokens: 120, onText: (piece, textSoFar) => {} });

// Voice, through ElevenLabs.
await demo.speak(text);                 // says it aloud; resolves when it has finished
const heard = await demo.listen();      // records the microphone until the speaker pauses; resolves with the words ('' if silent)
demo.stopListening();                   // ends a listen() early
```

`complete` and `chat` reject with an `Error` whose `message` can be shown to the audience, for
example when the model is still training.

The training object:

```js
{
  status: 'offline' | 'queued' | 'loading' | 'ready' | 'pretraining' | 'pretrained' | 'tuning' | 'tuned' | 'failed' | 'stopped',
  pretrain: null | {
    step, total,                        // 0 to 1290
    started, finished,                  // seconds since 1970; finished is null while it runs
    val: [{ step, loss }],              // how wrong its guesses are on unseen text; starts near 10.8, ends near 3.3
  },
  tune: null | { step, total, started, finished, name, loss: [{ step, loss }] },   // 0 to 240; a point every 10 steps; name: what the model was named, or ''
  samples: [                            // oldest first; grows as training runs
    { phase: 'pretrain' | 'tune', step, at,
      items: [{ prompt, text }] },      // pretrain: the same 3 sentence openings each time, with what the model wrote next
  ],                                    // tune: the same 2 questions each time, with the model's reply
}
```

A page opened partway through a run receives everything that has already happened, so always draw
from the whole object rather than from changes.

### What each page is for

- **Training monitor.** Shows the run as it happens: which phase and step, how wrong the model's
  guesses are over time, and above all the samples, so the audience can watch the writing go from
  gibberish to sentences. Show the newest sample most prominently and keep the earlier ones visible
  for comparison. It should work for pretraining and, later, for instruction tuning.
- **Document completion.** A text box, a button and the model's continuation streaming in after the
  prompt. Include the pre-written prompts from `demo/prompts.md` as one-click buttons, plus any of
  your own that would be fun for this audience.
- **Chat.** A conversation with the chat-tuned model: message list, text box, replies streaming in.
  Voice mode means a button that listens for a spoken message, sends it, and reads the reply aloud.
