# omega-lock

**The best score is lying to you. This tool catches the lie before it ships.**

[![pip install omega-lock](https://img.shields.io/badge/pip%20install-omega--lock-3775A9.svg)](https://pypi.org/project/omega-lock/)

```bash
pip install omega-lock
```

This is the plain-English page. No jargon, no setup, no prior knowledge needed.
Want the technical version with CI examples, the Optuna bridge, and the full API?
Read [README.md](README.md).

---

## Start with a story

Imagine you are trying to find the best setting for something — maybe a recipe, a
price, or a knob on a machine. You don't know the right answer, so you do the
obvious thing: you try a lot of options and keep the one that scored highest.

Say you try 500 different settings on the data you have. One of them comes back
with a great score. You feel good. You ship it.

Here is the trap. When you try 500 things and keep only the single best one, you
are not just keeping the most skillful one. You are also keeping the **luckiest**
one. With 500 tries, some setting was always going to look amazing by pure
chance — the way someone in a huge crowd always wins the raffle.

And luck does not come back tomorrow.

---

## The one picture that explains everything

Take that winning score and split it into two parts:

```
the winning score you saw   =   real skill   +   a lucky streak
```

The real skill is the part that will still be there next week. The lucky streak
is the part that vanishes the moment you look away. The problem is that the score
on your screen mixes the two together, and it looks the same either way.

So how do you separate them? You take the winner and you re-test it on **fresh
data it has never seen before** — data that had no chance to be part of the luck.
Whatever score survives on the fresh data is the real skill. Whatever evaporated
was the lucky streak.

That is the entire idea. omega-lock does exactly this, automatically.

---

## Watch it happen — 60 seconds, fully offline

```bash
omega-lock demo
```

It runs a small, self-contained story. A search picks a setting that looks
brilliant on its own data. Then omega-lock re-tests that same setting on fresh
data and shows you what is really there:

```
the winning setting

   on its own data    5.967     looked like a champion
   on fresh data      1.527     ▼ -74%   almost all of it was luck

VERDICT: DO NOT SHIP — the win did not hold up on fresh data.
```

The score dropped by nearly three quarters. That gap was never skill. It was the
crowd-and-raffle effect, and omega-lock caught it before it could fool you.

---

## Use it on your own work — one command

When you have your own numbers, give omega-lock two files: the scores your search
tool found, and the scores of those same settings checked again on fresh data.

```bash
omega-lock gate --train searched.json --holdout fresh.json
```

It answers with a simple yes or no:

```
PASS  ->  exit 0    the win held up on fresh data — safe to ship
FAIL  ->  exit 1    the win did not hold up — stop, do not ship
```

That is it. One command, one clear answer.

---

## What it actually looks at

omega-lock runs a few plain checks. Any one of them can tell you to stop.

**1. Does the win hold up on fresh data?**
It re-tests the winner on data it has never seen and measures how strongly the two
match. If the fresh-data result falls apart, the win was luck, and omega-lock says
stop.

**2. Is this winner even allowed?**
Sometimes the highest score comes from a setting you can't actually use — too slow,
too expensive, against the rules. This is an is-this-even-allowed check: you set a
budget or a speed cap, and omega-lock finds the best winner that obeys it, instead
of cheering for one you'd have to throw away.

**3. Can you prove later what you decided?**
Every run is written down in a paper trail that can't be edited after the fact. If
someone asks months later why a candidate passed or failed, the answer is right
there, exactly as it happened.

---

## The one thing to remember

The highest score is the most suspicious number you own. A real winner survives
fresh data it has never seen. Luck does not. omega-lock is the quick, honest check
that tells the two apart — before you ship the wrong one.

---

**More:** [Full README](README.md) (CI setup, search-tool bridges, the developer API) ·
[한국어 README](README_KR.md) · [쉬운 한국어 README](EASY_README_KR.md)
