# Submission

Pencils down at 8:30. Before then:

1. **Push your repo** with this file filled in. No API keys, no private data.
2. **Fill in your team card**: team name, statement number, first names, and the six short answers below.
3. **Hand the card to a judge.** Judges pick six finalists from the cards.
4. **Be ready to demo.** Three minutes, no extra prep time: the concern, the thing working on real data, one honest limit.

## Team card

**Team name:** Claude SuperMonitor

**Statement number**: 10

**First names:** Shayaan, Yunzhen, Ajinkya, Helena, Louise

**Statement:** How might we help **teams that hand bookkeeping to a group of AI agents** do **spot and shut out the one agent that is quietly cooking the books** so that **the ledger stays right without throwing out the honest agents**? Build idea: A (a better monitor). We'll know it worked if **the monitor catches the rogue at least as often as the simple two-strikes rule while wrongly removing far fewer honest agents**.

## Six short answers

**1. The concern:** who, specifically, and what failure would a neighbor recognize?

A small business or nonprofit whose books are reconciled by a team of AI agents. One agent is compromised or misconfigured and pads invoices by 10 to 25 percent, writing a plausible note each time. The owner pays inflated bills and nobody notices. The obvious fix, a rule that removes any agent caught in two disagreements, also throws out the honest agents, until nobody is left doing the books.

**2. What we built,** and the one command that runs it:

`sim.py` is a 12-round ledger game: five agents, one secret rogue, two agents per invoice, and the ledger averages when they disagree. A monitor sees only answers and notes, never the invoice or the true total. We compared a two-strikes baseline with a likelihood monitor. That monitor keeps P(rogue) for every agent, updates it with Bayes' rule from who disagreed and in which direction, and only quarantines when one agent passes 0.7.

`python3 sim.py --mock --monitor all --games 1000 --out results.md`

**3. Data we used:** dataset ids from `datasets.json`, how many rows, date range:

None from `datasets.json`. The track's challenge A is simulation-based. Invoices are generated from a seed (1,000 games, 12 invoices each). Agents are scripted in mock mode or played by Claude Haiku in real mode (`uv run sim.py --real --monitor all --games 20 --out results.md`). TODO: real-game results, if they are run before the deadline.

**4. What it found:** one measured result, with sample size:

On the same 1,000 mock games, the likelihood monitor caught the rogue as often as the baseline (95% vs 95%) and earlier (median round 3 vs 5). It wrongly removed 0.19 honest agents per game against the baseline's 2.19, and cut final ledger error from 3.80% to 1.27%. Full table in `results.md`.

**5. One honest limitation:** what it cannot tell you, what breaks:

The monitor works because the rogue pushes in one direction often, and it assumes that direction is up. A quieter rogue breaks it. If the rogue cheats on 40% of its invoices instead of 75%, the catch rate falls to 68% (1,000 games; the baseline falls to 74% and still removes 1.84 honest agents per game). Without the upward assumption it catches 85%, or 44% against the quiet rogue. The results are mostly from scripted agents, whose mistake rates are close to what the monitor assumes. Real agents' mistake rates are unmeasured, and the notes are not used at all.

**6. If this were deployed and it failed,** who is accountable, and how would they find out?

The team that deploys the agents and sets the monitor's threshold. The monitor doesn't remove the failure. It only makes it rarer. A missed rogue shows up only through ground truth, so a human should spot-check a sample of reconciled invoices against the source documents every period. A false quarantine shows up as a removed agent. Every quarantine should be logged with the monitor's probabilities and reviewed by a person before it becomes permanent.

## How judges score

Each card is scored 1 to 3 on the first three.

1. **Real concern.** It came from the room, with a specific who and a failure a neighbor would recognize.
2. **Works on real data.** Something runs on real data, and you can show what it found.
3. **Honest.** It names a limitation and says who is accountable if it fails.
4. **Tiebreak.** Would someone outside this room use it?
